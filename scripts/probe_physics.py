#!/usr/bin/env python3
"""SOMA physics validation probes (run after initial training).

Pass >=2/3 to proceed to live operation. 0/3 = investigate circular-input
invariant first (docs/RISKS.md I-1).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from backend.prediction_net import PredictionNetwork
from backend.simulation import SurROLTissueEnv, SimConfig


def main() -> int:
    net = PredictionNetwork()
    net.to("cpu")
    net.device = torch.device("cpu")
    ckpt = torch.load("backend/models/prediction_network.pt", map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    env = SurROLTissueEnv(SimConfig(seed=42))
    try:
        env.reset()
        s = env.get_state_vector()
        vasc = s[256:512].reshape(16, 16)

        def fwd(state, action):
            with torch.no_grad():
                x = torch.from_numpy(
                    np.concatenate([state, action])
                ).float().unsqueeze(0)
                t, d, _, inst = net(x)
            return t.squeeze().numpy().reshape(16, 16), float(d), inst.squeeze().numpy()

        hi = np.unravel_index(vasc.argmax(), (16, 16))
        lo = np.unravel_index(vasc.argmin(), (16, 16))

        def move_toward(col, row):
            a = np.zeros(12, dtype=np.float32)
            a[7] = 1.0
            a[0] = np.clip((col / 15.0) - s[768], -1, 1)
            a[1] = np.clip((row / 15.0) - s[769], -1, 1)
            a[11] = 0.8
            return a

        t_hi, d_hi, _ = fwd(s, move_toward(hi[1], hi[0]))
        t_lo, d_lo, _ = fwd(s, move_toward(lo[1], lo[0]))
        base = s[0:256].reshape(16, 16)

        p1 = bool((base[hi] - t_hi[hi]) > (base[lo] - t_lo[lo]))
        p2 = bool(d_hi > d_lo)

        tgt = s[776:778]
        ee = s[768:770]
        d = tgt - ee
        n = float(np.linalg.norm(d)) + 1e-8

        def toward(sign):
            a = np.zeros(12, dtype=np.float32)
            a[7] = 1.0
            a[0:2] = d / n * 0.1 * sign
            a[11] = 0.7
            return a

        _, _, inst_t = fwd(s, toward(1.0))
        _, _, inst_a = fwd(s, toward(-1.0))
        p3 = bool(np.linalg.norm(inst_t[:2] - tgt)
                  < np.linalg.norm(inst_a[:2] - tgt))

        results = {
            "vascularity_damage": p1,
            "damage_calibration": p2,
            "progress_monotone": p3,
        }
        for name, ok in results.items():
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        passed = sum(results.values())
        print(f"{passed}/3 probes passed")
        if passed == 0:
            print("CRITICAL: run test:belief:circular_input first (RISKS I-1)")
        return 0 if passed >= 2 else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
