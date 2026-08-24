"""PredictionNetwork — SOMA Task 1.4.

Learned dynamics model f_theta(s_t, a_t) -> s_{t+1}.
Spec: docs/specs/WORLD_MODEL.md "PREDICTION NETWORK".
Invariant: LAYERNORM ONLY — BatchNorm is prohibited
(docs/specs/ARCHITECTURE.md, LAYERNORM INVARIANT).

Imports: torch / numpy / dataclasses only. No backend imports.
Save/load lives in WorldModel (Task 1.5), not here.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class PredictedState:
    """One-step-ahead prediction for a single (state, action) pair.

    tissue       — predicted integrity grid, (16,16) float32, range (0,1)
    damage_prob  — P(vessel_damaged_this_step), range (0,1)
    vessel_risk  — P(too_close_to_vessel_i) per slot, (5,) float32, (0,1)
    pred_ee_norm — [x, y, z, orientation] normalized; denormalize on use:
                   ee_x = v[0]*0.2-0.1, ee_z = v[2]*0.3
    """

    tissue: np.ndarray
    damage_prob: float
    vessel_risk: np.ndarray
    pred_ee_norm: np.ndarray


class PredictionNetwork(nn.Module):
    """Encoder(818→128) + 4 heads. Input is concat(state_806, action_12)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(818, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.tissue_head = nn.Sequential(nn.Linear(128, 256), nn.Sigmoid())
        self.damage_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.vessel_head = nn.Sequential(nn.Linear(128, 5), nn.Sigmoid())
        # No activation on instrument head: Sigmoid saturates near 0/1 and
        # kills gradients for EE positions at grid edges (WORLD_MODEL.md).
        self.instrument_head = nn.Linear(128, 4)

        # Device selection once; every consumer must move tensors here too
        # (RISKS.md D-3: model .to(device) without input .to(device) silently
        # runs on CPU).
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """x: (batch, 818) float32 → (tissue, damage, vessel, instrument)."""
        h = self.encoder(x)
        return (
            self.tissue_head(h),      # (B, 256)
            self.damage_head(h),      # (B, 1)
            self.vessel_head(h),      # (B, 5)
            self.instrument_head(h),  # (B, 4)
        )

    def predict(
        self, state_vec: np.ndarray, action_vec: np.ndarray
    ) -> PredictedState:
        """Single-sample synchronous inference.

        Sets eval() mode internally — caller must NOT set mode.
        Called by WorldModel.predict() under _predict_lock.
        """
        self.eval()
        with torch.no_grad():
            x = (
                torch.from_numpy(np.concatenate([state_vec, action_vec]))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
            tissue, damage, vessel, instrument = self(x)
        return PredictedState(
            tissue=tissue.squeeze().cpu().numpy().reshape(16, 16).astype(np.float32),
            damage_prob=float(damage.squeeze().cpu()),
            vessel_risk=vessel.squeeze().cpu().numpy().astype(np.float32),
            pred_ee_norm=instrument.squeeze().cpu().numpy().astype(np.float32),
        )
