"""Task 1.4 named tests — PredictionNetwork.

References:
- docs/specs/WORLD_MODEL.md "PREDICTION NETWORK" (architecture, PredictedState)
- docs/specs/ARCHITECTURE.md LAYERNORM INVARIANT (BatchNorm prohibited)
"""

import time

import numpy as np
import torch

from backend.prediction_net import PredictedState, PredictionNetwork


def test_layernorm() -> None:
    """test:prediction_net:layernorm — BatchNorm1d must not exist anywhere
    in the module tree (LAYERNORM INVARIANT, ARCHITECTURE.md)."""
    net = PredictionNetwork()
    has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in net.modules())
    assert not has_bn, "BatchNorm found in PredictionNetwork — use LayerNorm"


def test_forward_shapes() -> None:
    """Batch forward: 4 heads with exact shapes; sigmoid heads bounded [0,1];
    instrument head unbounded (no activation — grid-edge gradients)."""
    net = PredictionNetwork()
    net.eval()
    x = torch.randn(4, 818).to(net.device)  # D-3: inputs must ride the device
    with torch.no_grad():
        tissue, damage, vessel, instrument = net(x)

    assert tissue.shape == (4, 256)
    assert damage.shape == (4, 1)
    assert vessel.shape == (4, 5)
    assert instrument.shape == (4, 4)

    assert torch.all(tissue >= 0) and torch.all(tissue <= 1), "tissue sigmoid"
    assert torch.all(damage >= 0) and torch.all(damage <= 1), "damage sigmoid"
    assert torch.all(vessel >= 0) and torch.all(vessel <= 1), "vessel sigmoid"


def test_param_count() -> None:
    """Expected ~615K parameters (WORLD_MODEL.md); allow ±15%."""
    net = PredictionNetwork()
    n = sum(p.numel() for p in net.parameters())
    assert 520_000 < n < 710_000, f"expected ~615K params, got {n}"


def test_inference_latency() -> None:
    """Single-sample forward < 25ms (lenient CI bound; MPS/CPU)."""
    net = PredictionNetwork()
    net.eval()
    x = torch.randn(1, 818).to(net.device)  # D-3: inputs must ride the device
    with torch.no_grad():
        net(x)  # warmup — first call compiles device kernels
        start = time.perf_counter()
        for _ in range(20):
            net(x)
        elapsed_s = time.perf_counter() - start
    per_call_ms = elapsed_s / 20 * 1000
    assert per_call_ms < 25, f"forward pass {per_call_ms:.1f}ms exceeds 25ms"


def test_predict_returns_predictedstate() -> None:
    """predict(state_806, action_12) → PredictedState with exact field
    shapes/dtypes per WORLD_MODEL.md."""
    net = PredictionNetwork()
    rng = np.random.RandomState(42)
    state_vec = rng.randn(806).astype(np.float32)
    action_vec = rng.randn(12).astype(np.float32)

    pred = net.predict(state_vec, action_vec)

    assert isinstance(pred, PredictedState)
    assert pred.tissue.shape == (16, 16)
    assert pred.tissue.dtype == np.float32
    assert isinstance(pred.damage_prob, float)
    assert 0.0 <= pred.damage_prob <= 1.0
    assert pred.vessel_risk.shape == (5,)
    assert pred.vessel_risk.dtype == np.float32
    assert pred.pred_ee_norm.shape == (4,)
    assert pred.pred_ee_norm.dtype == np.float32
