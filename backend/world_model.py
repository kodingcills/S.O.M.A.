"""WorldModel — SOMA Task 1.5.

Singleton wrapper owning the PredictionNetwork weights, the BeliefState,
the replay buffer, and the two asyncio locks that make fine-tuning safe on
a single event loop.

Spec: docs/specs/WORLD_MODEL.md "WORLD MODEL CLASS", "fine_tune",
"_fine_tune_sync", "Replay Buffer".
Invariants:
  - WORLD MODEL SINGLETON INVARIANT (ARCHITECTURE.md): constructed exactly
    once, in main.py. No other module may build one.
  - EVENT LOOP NON-BLOCKING INVARIANT (ARCHITECTURE.md): every CPU-heavy
    step runs via asyncio.to_thread; _fine_tune_sync is a REGULAR def and
    must never contain an await ("no running event loop" pitfall).
  - Lock order contract: fine_tune acquires _finetune_lock THEN
    _predict_lock. No other path holds both locks, so no deadlock is
    possible; predict paths take _predict_lock only.

Imports: event_log, belief_state, prediction_net per MODULE IMPORT RULES.
"""

from __future__ import annotations

import asyncio
import threading
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:  # W&B optional — system operates correctly without it
    wandb = None

from backend.belief_state import BeliefState
from backend.event_log import write_event
from backend.prediction_net import PredictedState, PredictionNetwork

# Constants (QUICK_REFERENCE.md thresholds table)
FINE_TUNE_EPOCHS = 5        # targeted update, not full retraining
FINE_TUNE_BATCH_SIZE = 64   # smaller than initial-training batches
FINE_TUNE_LR = 1e-4         # 10x smaller than initial lr — avoid overwriting
REPLAY_MAX = 100_000        # FIFO eviction beyond this

# Loss weights (WORLD_MODEL.md): tissue drives error map + MPC value fn;
# damage is the safety signal with asymmetric cost.
LOSS_W_TISSUE = 0.6
LOSS_W_DAMAGE = 0.3

REPLAY_RATIO_NEW_TO_OLD = 4  # n_old = len(new) * 4 -> 80% old / 20% new mix


class WorldModel:
    """Owns network + belief + replay buffer for the whole process.

    Constructed exactly once in main.py and passed as a parameter everywhere
    else (SINGLETON INVARIANT). A second instance would carry untrained
    weights and a blank error map — MPC planning and orchestrator spawning
    would silently operate on garbage.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        # Anchor to module dir — CWD differs between `uvicorn backend.main:app`
        # (repo root) and `uvicorn main:app` (backend/). Silent wrong-cwd load
        # would boot on random weights with no error.
        self.model_path = (
            Path(model_path) if model_path is not None
            else Path(__file__).resolve().parent / "models" / "prediction_network.pt"
        )
        self.network = PredictionNetwork()
        self.belief = BeliefState()

        self._predict_lock = asyncio.Lock()    # held ~ms per predict call
        self._finetune_lock = asyncio.Lock()   # held for entire fine_tune()
        # threading mirror of _predict_lock: update() runs sync (MPC thread),
        # asyncio.Lock cannot be acquired from sync context.
        self._training_guard = threading.Lock()

        # (state_806, action_12, next_806) tuples, FIFO eviction at REPLAY_MAX
        self._replay_buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # Adam persisted across calls: momentum/adaptive-lr state encodes which
        # parameters move often vs rarely — discarding it between fine-tunes
        # throws away accumulated gradient statistics (WORLD_MODEL.md).
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=FINE_TUNE_LR)

        self.load(self.model_path)

    # -- inference -----------------------------------------------------------

    def predict(
        self, state_vec: np.ndarray, action_vec: np.ndarray
    ) -> PredictedState:
        """Synchronous single-sample inference.

        CONTRACT: the caller owns _predict_lock acquisition (MPC path wraps
        its candidate loop in `async with wm._predict_lock` and offloads this
        call via asyncio.to_thread). This method never locks or awaits.
        """
        return self.network.predict(state_vec, action_vec)

    async def predict_locked(
        self, state_vec: np.ndarray, action_vec: np.ndarray
    ) -> PredictedState:
        """Convenience locked inference for callers without an open lock scope."""
        async with self._predict_lock:
            return await asyncio.to_thread(self.network.predict, state_vec, action_vec)

    # -- online update (MPC loop, sync thread context) ------------------------

    def update(
        self,
        state_vec: np.ndarray,
        action_vec: np.ndarray,
        next_state_vec: np.ndarray,
    ) -> None:
        """One-step bookkeeping after an executed action. Synchronous.

        Called by MPCAgent right after env.step(): refreshes the error map
        from predicted-vs-actual tissue, folds the observation into the
        belief, and files the transition into the replay buffer.
        """
        if not self._training_guard.acquire(blocking=False):
            return  # fine-tune in flight: predict would flip train->eval mid-loop
        try:
            predicted = self.network.predict(state_vec, action_vec)
        finally:
            self._training_guard.release()
        actual_integrity = next_state_vec[0:256].reshape(16, 16)
        self.belief.update_prediction_error(predicted.tissue, actual_integrity)
        self.belief.update_from_observation(next_state_vec)
        self._add_to_replay([state_vec], [action_vec], [next_state_vec])

    # -- fine-tuning -----------------------------------------------------------

    async def fine_tune(
        self, new_samples: list[tuple], epochs: int = FINE_TUNE_EPOCHS
    ) -> float:
        """Train on new_samples mixed with replay; returns mean step loss.

        Lock choreography (see class docstring for ordering rationale):
          1. _finetune_lock serializes concurrent fine-tunes for their full
             duration.
          2. _predict_lock is held ONLY around the weight-update window so
             MPC predictions pause just for the swap, not the whole training.
          3. Version bump happens AFTER releasing _predict_lock but INSIDE
             _finetune_lock — predictions resume immediately on the new
             weights while version bookkeeping stays race-free.
        """
        async with self._finetune_lock:
            async with self._predict_lock:
                # threading.Lock is fine across await (same-task release);
                # update() checks it non-blocking from the MPC path.
                with self._training_guard:
                    loss = await asyncio.to_thread(
                        self._fine_tune_sync, new_samples, epochs
                    )

            self.belief.world_model_version += 1

            try:
                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        "fine_tune_loss": loss,
                        "world_model_version": self.belief.world_model_version,
                        "global_error": float(self.belief.prediction_error_map.mean()),
                    })
            except Exception:
                pass  # W&B optional — never break fine-tuning over it

            try:
                write_event(
                    "world_model_updated",
                    payload_version=self.belief.world_model_version,
                    payload_loss=loss,
                )
            except Exception:
                pass

            try:
                self._save()
            except Exception:
                pass

        return loss

    def _fine_tune_sync(self, new_samples: list[tuple], epochs: int) -> float:
        """Training loop. REGULAR def — runs in a worker thread via
        asyncio.to_thread. Any await here raises 'no running event loop'
        (RISKS pitfall); all side effects happen in the async caller.
        """
        n_new = len(new_samples)
        n_old = min(n_new * REPLAY_RATIO_NEW_TO_OLD, len(self._replay_buffer))
        old_samples = random.sample(self._replay_buffer, n_old)
        batch = list(new_samples) + old_samples
        random.shuffle(batch)

        self.network.train()
        total_loss = 0.0
        steps = 0

        for _epoch in range(epochs):
            for i in range(0, len(batch), FINE_TUNE_BATCH_SIZE):
                chunk = batch[i : i + FINE_TUNE_BATCH_SIZE]
                states, actions, nexts = zip(*chunk)

                x = (
                    torch.from_numpy(
                        np.column_stack([np.stack(states), np.stack(actions)])
                    )
                    .float()
                    .to(self.network.device)
                )
                y_tissue = (
                    torch.from_numpy(np.stack([n[0:256] for n in nexts]))
                    .float()
                    .to(self.network.device)
                )
                # damaged flag lives at index [779 + slot*4 + 3] of each next-state
                y_damage = torch.tensor(
                    [
                        float(any(n[779 + j * 4 + 3] > 0.5 for j in range(5)))
                        for n in nexts
                    ],
                    dtype=torch.float32,
                ).unsqueeze(1).to(self.network.device)

                tissue_pred, damage_pred, _vessel_pred, _instrument_pred = self.network(x)
                loss = (
                    LOSS_W_TISSUE * F.mse_loss(tissue_pred, y_tissue)
                    + LOSS_W_DAMAGE
                    * F.binary_cross_entropy(
                        damage_pred.clamp(1e-7, 1 - 1e-7), y_damage
                    )
                )
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                total_loss += float(loss.item())
                steps += 1

        self.network.eval()
        if new_samples:
            states, actions, nexts = zip(*new_samples)
            self._add_to_replay(states, actions, nexts)
        return total_loss / max(1, steps)  # mean step loss across epochs*batches

    # -- persistence -----------------------------------------------------------

    def load(self, path: str | Path | None = None) -> None:
        """Restore weights + optimizer state from checkpoint, if present.

        Missing checkpoint keeps random-initialized weights and
        world_model_version == 0; initial training (train.py / main.py
        startup) fills the gap before any consumer relies on predictions.
        """
        p = Path(path) if path is not None else self.model_path
        if p.exists():
            checkpoint = torch.load(str(p), map_location=self.network.device)
            self.network.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.network.eval()
        else:
            try:
                write_event("system_ready", warning="No saved model found")
            except Exception:
                pass
            self.network.eval()

    def _save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "world_model_version": self.belief.world_model_version,
            },
            str(self.model_path),
        )

    # -- replay buffer -----------------------------------------------------------

    def _add_to_replay(
        self,
        state_vecs,
        action_vecs,
        next_vecs,
    ) -> None:
        """Append transitions as defensive copies; FIFO-evict past REPLAY_MAX."""
        for s, a, n in zip(state_vecs, action_vecs, next_vecs):
            self._replay_buffer.append(
                (
                    np.asarray(s, dtype=np.float32).copy(),
                    np.asarray(a, dtype=np.float32).copy(),
                    np.asarray(n, dtype=np.float32).copy(),
                )
            )
        if len(self._replay_buffer) > REPLAY_MAX:
            del self._replay_buffer[: len(self._replay_buffer) - REPLAY_MAX]
