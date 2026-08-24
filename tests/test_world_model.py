"""Tests for backend/world_model.py — SOMA Task 1.5.

Named tests per WORLD_MODEL.md TESTING PROTOCOL + ARCHITECTURE.md invariants:
  test_singleton_grep                  — WORLD MODEL SINGLETON INVARIANT — MUST PASS
  test_nonblocking                     — EVENT LOOP NON-BLOCKING INVARIANT — MUST PASS
  test_fine_tune_updates_version_and_saves — version bump + checkpoint round-trip
  test_replay_ratio                    — n_old == min(len(new)*4, len(buffer))
  test_update_updates_error_map        — update() wires predict → error map → belief
  test_predict_returns_predicted_state — sync + locked inference contract

Suite budget < 60s: small sample counts, CPU-forced devices where timing matters.
"""

import asyncio
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from backend.prediction_net import PredictionNetwork, PredictedState
from backend.world_model import WorldModel

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Synthetic fixtures — same pattern as test_belief_state.make_state_vec
# ---------------------------------------------------------------------------

def make_state_vec(seed: int = 0) -> np.ndarray:
    """Synthetic (806,) float32 vector matching the SIMULATION.md index table."""
    rng = np.random.RandomState(seed)
    v = np.zeros(806, dtype=np.float32)
    v[0:256] = rng.uniform(0.3, 1.0, 256).astype(np.float32)            # integrity
    v[256:512] = rng.uniform(0.0, 1.0, 256).astype(np.float32)          # vascularity
    v[512:768] = (rng.uniform(0, 1, 256) > 0.8).astype(np.float32)      # bleeding
    v[768:771] = rng.uniform(0.05, 0.95, 3).astype(np.float32)          # EE x,y,z
    v[771:775] = rng.uniform(-1.0, 1.0, 4).astype(np.float32)           # quaternion
    v[775] = 0.25                                                        # gripper
    v[776] = 0.4                                                         # target col/15
    v[777] = 0.6                                                         # target row/15
    v[778] = 0.0                                                         # not reached
    # vessel slots: damaged flag at [779 + i*4 + 3] varies so BCE sees both classes
    for i in range(5):
        v[779 + i * 4 : 782 + i * 4] = rng.uniform(0.0, 1.0, 3).astype(np.float32)
        v[782 + i * 4] = float(rng.uniform() > 0.5)
    v[799] = 1.0                                                         # DOF 1 active
    return v


def make_action(rng: np.random.RandomState) -> np.ndarray:
    a = np.zeros(12, dtype=np.float32)
    a[0:3] = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
    a[7 + int(rng.randint(0, 4))] = 1.0                                  # mode one-hot
    a[11] = rng.uniform(0.2, 1.0)                                        # magnitude
    return a


def make_sample(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed + 500_000)
    return (make_state_vec(seed), make_action(rng), make_state_vec(seed + 9_000))


def make_world_model(tmp_path: Path, name: str = "wm.pt") -> WorldModel:
    return WorldModel(model_path=tmp_path / name)


def force_cpu(wm: WorldModel) -> None:
    """Pin network to CPU so MPS contention never flakes the timing test."""
    wm.network.device = torch.device("cpu")
    wm.network.to(wm.network.device)


def seed_replay(wm: WorldModel, n: int = 50, seed: int = 0) -> None:
    """Populate the replay buffer through the public update() path."""
    rng = np.random.RandomState(seed)
    for i in range(n):
        wm.update(make_state_vec(seed=i), make_action(rng), make_state_vec(seed=10_000 + i))


# ---------------------------------------------------------------------------
# test:world_model:singleton — SINGLETON INVARIANT (ARCHITECTURE.md) — MUST PASS
# ---------------------------------------------------------------------------

def test_singleton_grep():
    """grep -r 'WorldModel(' backend/ must be empty after excluding main.py,
    test files, and comment lines. Any hit means a second instantiation site."""
    result = subprocess.run(
        ["grep", "-rn", "WorldModel(", "backend/"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    hits = [
        line
        for line in result.stdout.splitlines()
        if "main.py" not in line
        and "/test_" not in line
        and not line.split(":", 2)[-1].lstrip().startswith("#")
    ]
    assert hits == [], f"SINGLETON INVARIANT violated:\n" + "\n".join(hits)


# ---------------------------------------------------------------------------
# test:world_model:nonblocking — EVENT LOOP NON-BLOCKING INVARIANT — MUST PASS
# ---------------------------------------------------------------------------

def test_nonblocking(tmp_path, monkeypatch):
    """PORT of the ARCHITECTURE.md timing test.

    A tick task fires every 100ms on the event loop while fine_tune runs.
    At least 8/10 ticks must land BEFORE fine_tune returns — proves training
    was offloaded via asyncio.to_thread instead of blocking the loop.

    Determinism: raw CPU training on an M-series machine can finish in well
    under a second, which would starve even a correct implementation of ticks.
    We wrap _fine_tune_sync with a fixed 1.2s sleep (executed inside whatever
    thread to_thread chose) so the race window is deterministic: correct
    offload keeps the loop free (all 10 ticks fire); a regression that calls
    _fine_tune_sync directly blocks the loop and lands ~0 ticks.
    """
    wm = make_world_model(tmp_path)
    force_cpu(wm)
    seed_replay(wm, n=50)

    real_sync = wm._fine_tune_sync

    def slow_sync(new_samples, epochs):
        time.sleep(1.2)  # GIL released during sleep — loop stays responsive
        return real_sync(new_samples, epochs)

    monkeypatch.setattr(wm, "_fine_tune_sync", slow_sync)

    samples = [make_sample(i) for i in range(300)]
    tick_times: list[float] = []

    async def count_ticks():
        for _ in range(10):
            await asyncio.sleep(0.1)
            tick_times.append(time.monotonic())

    async def scenario():
        tick_task = asyncio.create_task(count_ticks())
        loss = await wm.fine_tune(samples)
        ft_end = time.monotonic()
        await tick_task
        return loss, ft_end

    loss, ft_end = asyncio.run(scenario())

    assert isinstance(loss, float)
    ticks_during = sum(1 for t in tick_times if t <= ft_end)
    assert ticks_during >= 8, (
        f"Event loop blocked during fine_tune: only {ticks_during}/10 ticks "
        f"received before fine_tune returned"
    )


# ---------------------------------------------------------------------------
# test:world_model:version_increments (+ checkpoint round-trip)
# ---------------------------------------------------------------------------

def test_fine_tune_updates_version_and_saves(tmp_path):
    wm = make_world_model(tmp_path)
    force_cpu(wm)
    seed_replay(wm, n=50)
    assert wm.belief.world_model_version == 0

    loss = asyncio.run(wm.fine_tune([make_sample(i) for i in range(30)], epochs=1))

    assert isinstance(loss, float) and loss >= 0.0
    assert wm.belief.world_model_version == 1

    ckpt = tmp_path / "wm.pt"
    assert ckpt.exists()
    loaded = torch.load(str(ckpt), map_location="cpu")
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert loaded["world_model_version"] == 1

    fresh_net = PredictionNetwork()
    fresh_net.load_state_dict(loaded["model_state_dict"])  # reloadable


# ---------------------------------------------------------------------------
# test:world_model:replay_ratio — 80/20 old/new mix, capped by buffer size
# ---------------------------------------------------------------------------

def test_replay_ratio(tmp_path, monkeypatch):
    wm = make_world_model(tmp_path)
    force_cpu(wm)
    seed_replay(wm, n=40)

    captured: list[tuple[int, int]] = []
    real_sample = random.sample

    def spy(population, k):
        captured.append((len(population), k))
        return real_sample(population, k)

    monkeypatch.setattr(random, "sample", spy)

    new = [make_sample(1000 + i) for i in range(10)]
    asyncio.run(wm.fine_tune(new, epochs=1))

    assert captured, "random.sample never called — replay mixing missing"
    population_len, k = captured[0]
    assert population_len == 40          # drawn from the replay buffer
    assert k == min(10 * 4, 40) == 40    # 4:1 old:new ratio, capped by buffer


# ---------------------------------------------------------------------------
# test:world_model:update_wires_error_map
# ---------------------------------------------------------------------------

def test_update_updates_error_map(tmp_path):
    wm = make_world_model(tmp_path)
    s = make_state_vec(seed=1)
    a = make_action(np.random.RandomState(2))
    nxt = make_state_vec(seed=3)  # different integrity pattern than prediction

    wm.update(s, a, nxt)

    assert np.any(wm.belief.prediction_error_map > 0)
    np.testing.assert_allclose(
        wm.belief.integrity, nxt[0:256].reshape(16, 16), atol=1e-6
    )
    assert len(wm._replay_buffer) == 1
    assert wm.belief.episode_count == 1


# ---------------------------------------------------------------------------
# Inference contract: sync predict (caller owns lock) + locked convenience
# ---------------------------------------------------------------------------

def test_predict_returns_predicted_state(tmp_path):
    wm = make_world_model(tmp_path)
    s = make_state_vec(seed=5)
    a = make_action(np.random.RandomState(6))

    pred = wm.predict(s, a)
    assert isinstance(pred, PredictedState)
    assert pred.tissue.shape == (16, 16)
    assert 0.0 <= pred.damage_prob <= 1.0

    pred_locked = asyncio.run(wm.predict_locked(s, a))
    assert isinstance(pred_locked, PredictedState)
    assert pred_locked.tissue.shape == (16, 16)
