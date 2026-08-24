"""Initial training script — SOMA Task 1.4. Spec: WORLD_MODEL.md "INITIAL TRAINING".
Run: uv run python -m backend.train | import backend.train.train_prediction_network.
DEVIATIONS (QUICK_REFERENCE.md protocol): (1) vessel_proximity label computed in
normalized space: euclid(state[768:770], next[779+i*4:+2]) < next[779+i*4+2] + 0.01
— spec wanted workspace radius + 2mm; workspace width 0.2m makes 0.002/0.2 = 0.01,
numerically identical. (2) damage BCE inputs also clamped to [1e-7, 1-1e-7]
(spec clamps vessel head only) — sigmoid can emit exactly 0.0/1.0 in f32 → NaN (D-2).
"""

import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — canonical torch alias
from torch.optim.lr_scheduler import CosineAnnealingLR

from backend.prediction_net import PredictionNetwork

MAX_EPOCHS, EXTENDED_EPOCHS, EXTENDED_LR = 50, 100, 1e-4
LR, ETA_MIN, BATCH_SIZE = 1e-3, 1e-4, 256
EARLY_STOP_PATIENCE, MIN_SAMPLES, SPLIT_SEED = 10, 1000, 42
TARGET_VAL_TISSUE_MSE, TARGET_VAL_DAMAGE_ACC = 0.05, 0.85
IMPROVEMENT_EPSILON, BCE_CLAMP = 1e-3, 1e-7  # epsilon: min val_mse drop / BCE clamp
LOSS_W_TISSUE, LOSS_W_DAMAGE, LOSS_W_VESSEL = 0.6, 0.3, 0.1


@dataclass(frozen=True, slots=True)
class TrainingData:
    """Full dataset as dense arrays plus precomputed head targets."""

    states: np.ndarray            # (N, 806) f32
    actions: np.ndarray           # (N, 12)  f32
    tissue_targets: np.ndarray    # (N, 256) f32 ← next_state[0:256]
    damage_labels: np.ndarray     # (N,)     f32 ← any vessel damaged flag
    proximity_labels: np.ndarray  # (N, 5)   f32 ← EE-near-vessel flags


@dataclass(frozen=True, slots=True)
class _TrainContext:
    """Bundle so helper functions stay ≤3 params."""

    net: PredictionNetwork
    optimizer: torch.optim.Optimizer
    device: torch.device


def _write_event_safe(event_type: str, **payload: object) -> None:
    # Lazy import so train.py runs standalone pre-Task-1.1.
    try:
        from backend.event_log import write_event

        write_event(event_type, **payload)
    except Exception:
        pass  # standalone mode — event log not built yet


def _init_wandb(n_samples: int) -> ModuleType | None:
    """Best-effort W&B init; training must run without it."""
    try:
        import wandb

        wandb.init(project="soma-surgical", name=f"initial-train-{int(time.time())}",
                   config={"lr": LR, "epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE,
                           "state_dim": 806, "action_dim": 12,
                           "n_samples": n_samples})
        return wandb
    except Exception as exc:
        print(f"wandb unavailable ({exc}) — training without dashboard")
        return None


def _wandb_log(wandb_mod: ModuleType | None, payload: dict[str, object]) -> None:
    if wandb_mod is None:
        return
    try:
        wandb_mod.log(payload)
    except Exception:
        pass  # never let telemetry kill training


def _load_training_data(db_path: str) -> TrainingData | None:
    """samples table → TrainingData, or None (message printed) if missing/short/NaN."""
    path = Path(db_path)
    if not path.exists():
        print(f"[train] {db_path} not found — skipping initial training")
        return None

    states, actions, next_states, skipped = [], [], [], 0
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT state, action, next_state FROM samples")
        for blob_s, blob_a, blob_ns in rows:
            s, a, ns = (np.frombuffer(b, dtype=np.float32)
                        for b in (blob_s, blob_a, blob_ns))
            if s.size != 806 or a.size != 12 or ns.size != 806:
                skipped += 1
                continue
            states.append(s), actions.append(a), next_states.append(ns)
    finally:
        con.close()
    if skipped:
        print(f"[train] skipped {skipped} malformed rows (wrong BLOB size)")
    if len(states) < MIN_SAMPLES:
        print(f"[train] only {len(states)} valid samples (< {MIN_SAMPLES}) "
              f"in {db_path} — skipping initial training")
        return None

    st, ac, ns = np.stack(states), np.stack(actions), np.stack(next_states)

    # RISKS.md D-2: NaN from epoch 1 batch 1 is a DATA problem. Filter here.
    finite = (np.isfinite(st).all(axis=1) & np.isfinite(ac).all(axis=1)
              & np.isfinite(ns).all(axis=1))
    dropped = int((~finite).sum())
    if dropped:
        print(f"[train] dropped {dropped} rows containing NaN/Inf (D-2 guard)")
    st, ac, ns = st[finite], ac[finite], ns[finite]
    if len(st) < MIN_SAMPLES:
        print(f"[train] < {MIN_SAMPLES} clean samples after NaN filter — skipping")
        return None

    # Head targets. Damage flag: vessel slot i damaged bit at 779+i*4+3.
    # Proximity: see DEVIATIONS in module docstring.
    ee_xy = st[:, 768:770][:, None, :]                        # (N, 1, 2)
    slots = ns[:, 779:799].reshape(-1, 5, 4)
    dist = np.linalg.norm(ee_xy - slots[:, :, 0:2], axis=2)   # (N, 5)
    return TrainingData(
        states=st, actions=ac,
        tissue_targets=ns[:, 0:256].copy(),
        damage_labels=(ns[:, 779 + 3 :: 4] > 0.5).any(axis=1).astype(np.float32),
        proximity_labels=(dist < slots[:, :, 2] + 0.01).astype(np.float32))


def _batch_tensors(
    data: TrainingData, idx: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one on-device batch. Every tensor moves to device (RISKS D-3)."""
    x = np.column_stack([data.states[idx], data.actions[idx]])
    x_t = torch.from_numpy(x).float().to(device)
    y_tissue = torch.from_numpy(data.tissue_targets[idx]).float().to(device)
    y_damage = torch.from_numpy(data.damage_labels[idx]).float().unsqueeze(1).to(device)
    y_vessel = torch.from_numpy(data.proximity_labels[idx]).float().to(device)
    return x_t, y_tissue, y_damage, y_vessel


def _run_epoch(ctx: _TrainContext, data: TrainingData, order: np.ndarray) -> float:
    """One training pass over `order`; returns mean batch loss."""
    ctx.net.train()
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start : start + BATCH_SIZE]
        x, y_tissue, y_damage, y_vessel = _batch_tensors(data, idx, ctx.device)
        tissue_pred, damage_pred, vessel_pred, _ = ctx.net(x)
        loss = (LOSS_W_TISSUE * F.mse_loss(tissue_pred, y_tissue)
                + LOSS_W_DAMAGE * F.binary_cross_entropy(
                    damage_pred.clamp(BCE_CLAMP, 1 - BCE_CLAMP), y_damage)
                + LOSS_W_VESSEL * F.binary_cross_entropy(
                    vessel_pred.clamp(BCE_CLAMP, 1 - BCE_CLAMP), y_vessel))
        ctx.optimizer.zero_grad()
        loss.backward()
        ctx.optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def _evaluate(ctx: _TrainContext, data: TrainingData) -> tuple[float, float]:
    """Returns (val_tissue_mse, val_damage_accuracy)."""
    ctx.net.eval()
    mse_sum, correct = 0.0, 0.0
    for start in range(0, len(data.states), BATCH_SIZE):
        idx = np.arange(start, min(start + BATCH_SIZE, len(data.states)))
        x, y_tissue, y_damage, _ = _batch_tensors(data, idx, ctx.device)
        tissue_pred, damage_pred, _, _ = ctx.net(x)
        mse_sum += float(F.mse_loss(tissue_pred, y_tissue, reduction="sum").item())
        correct += float(((damage_pred > 0.5) == (y_damage > 0.5)).sum().item())
    return mse_sum / (len(data.states) * 256), correct / len(data.states)


def _save_checkpoint(
    model_path: Path, ctx: _TrainContext, meta: dict[str, object]
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": ctx.net.state_dict(),
                "optimizer_state_dict": ctx.optimizer.state_dict(),
                "epoch": meta["epoch"], "val_tissue_mse": meta["val_tissue_mse"],
                "val_damage_acc": meta["val_damage_acc"]}, model_path)
    model_path.with_name("training_meta.json").write_text(json.dumps(meta, indent=2))


def train_prediction_network(
    db_path: str = "backend/data/training.db",
    out_dir: str = "backend/models",
) -> dict[str, object] | None:
    """Train from training.db → summary dict, or None when skipped."""
    data = _load_training_data(db_path)
    if data is None:
        return None

    _write_event_safe("training_started", payload_n_samples=len(data.states))
    wandb_mod = _init_wandb(len(data.states))

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(indices := list(range(len(data.states))))
    split = int(len(indices) * 0.1)
    val_idx, train_idx = np.array(indices[:split]), np.array(indices[split:])
    print(f"[train] {len(train_idx)} train / {len(val_idx)} val samples "
          f"(split seed={SPLIT_SEED})")

    net = PredictionNetwork()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=ETA_MIN)
    ctx = _TrainContext(net=net, optimizer=optimizer, device=net.device)
    model_path = Path(out_dir) / "prediction_network.pt"

    best_val_mse, best_val_acc, patience = float("inf"), 0.0, 0
    epoch, total_epochs, extended = 0, MAX_EPOCHS, False
    shuffle_rng = np.random.default_rng(SPLIT_SEED)
    epoch_durations: list[float] = []

    while True:
        if epoch == MAX_EPOCHS and not extended:
            if best_val_mse < TARGET_VAL_TISSUE_MSE \
                    and best_val_acc > TARGET_VAL_DAMAGE_ACC:
                break  # targets met by saved checkpoint
            extended, total_epochs = True, EXTENDED_EPOCHS
            for group in optimizer.param_groups:
                group["lr"] = EXTENDED_LR
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=ETA_MIN)
            print(f"[train] targets unmet after {MAX_EPOCHS} epochs — "
                  f"extending to {EXTENDED_EPOCHS} at lr={EXTENDED_LR}")
        if epoch >= total_epochs:
            break

        t0 = time.perf_counter()
        order = train_idx.copy()
        shuffle_rng.shuffle(order)
        train_loss = _run_epoch(ctx, data, order)
        val_mse, val_acc = _evaluate(ctx, data)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        prev_best_mse = best_val_mse
        epoch_durations.append(time.perf_counter() - t0)
        remaining = float(np.mean(epoch_durations)) * (total_epochs - epoch - 1)

        print(f"Epoch {epoch + 1}/{total_epochs} | loss={train_loss:.3f} | "
              f"val_mse={val_mse:.3f} | dam_acc={val_acc * 100:.1f} | "
              f"lr={lr_now:.6f} | {int(remaining // 60)}m"
              f"{int(remaining % 60):02d}s remaining")
        _wandb_log(wandb_mod, {"epoch": epoch + 1, "train_loss": train_loss,
                               "val_tissue_mse": val_mse,
                               "val_damage_acc": val_acc, "lr": lr_now})

        # Checkpoint on EVERY improvement; patience resets only on a
        # meaningful (> IMPROVEMENT_EPSILON) improvement (WORLD_MODEL.md).
        if val_mse < best_val_mse:
            best_val_mse, best_val_acc = val_mse, val_acc
            _save_checkpoint(model_path, ctx, {
                "epoch": epoch + 1, "val_tissue_mse": val_mse,
                "val_damage_acc": val_acc, "n_train": len(train_idx),
                "n_val": len(val_idx), "device": str(net.device),
                "targets_met": False, "timestamp": time.time()})
        patience = (0 if val_mse < prev_best_mse - IMPROVEMENT_EPSILON
                    else patience + 1)
        if patience >= EARLY_STOP_PATIENCE:
            print(f"[train] early stop: no val_mse improvement for {patience} epochs")
            break
        if val_mse < TARGET_VAL_TISSUE_MSE and val_acc > TARGET_VAL_DAMAGE_ACC:
            print("[train] success targets met — stopping early")
            break
        epoch += 1

    # Reload best weights so the in-memory net matches the saved checkpoint.
    ckpt = torch.load(model_path, map_location=net.device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    targets_met = bool(best_val_mse < TARGET_VAL_TISSUE_MSE
                       and best_val_acc > TARGET_VAL_DAMAGE_ACC)
    summary: dict[str, object] = {
        "epochs_run": epoch + 1, "val_tissue_mse": best_val_mse,
        "val_damage_acc": best_val_acc, "targets_met": targets_met,
        "model_path": str(model_path)}
    print(f"[train] done: {summary}")
    _write_event_safe("training_completed", payload_val_mse=best_val_mse,
                      payload_dam_acc=best_val_acc, payload_targets_met=targets_met)
    _wandb_log(wandb_mod, {"final_val_tissue_mse": best_val_mse})
    if wandb_mod is not None:
        try:
            wandb_mod.finish()
        except Exception:
            pass
    return summary


if __name__ == "__main__":
    train_prediction_network()
