"""MPC planning agent — SOMA Task 1.8.

Model-predictive action selection using the WorldModel as the one-step
dynamics model. Spec: docs/specs/WORLD_MODEL.md "MPC PLANNING AGENT".

Direct-call law exception (ARCHITECTURE.md): MPCAgent -> WorldModel.predict()
every sim step (~50ms budget) and WorldModel.update() after each step.
Everything else goes through write_event().

Non-blocking (EVENT LOOP NON-BLOCKING INVARIANT): every CPU-bound call
(network.predict, env.step, env.reset) runs via asyncio.to_thread(); the
loop yields with asyncio.sleep(0.05) as the mandatory last line of every
iteration — including after a transient error, so a persistently failing
step can never spin the loop hot.

DEVIATIONS from WORLD_MODEL.md literal code (all locked by tests/test_mpc.py
or the Task 1.8 build order):
  1. ±45° flank candidates scale only the xyz deltas by STEP. The spec's
     `make_move(...) * STEP` multiplies the whole 12-vector, collapsing the
     one-hot mode slot to 0.1 and magnitude to 0.07 — test_generates_
     exactly_8_candidates asserts one-hot modes, so delta-only scaling wins.
  2. reward / task_failed come from env.step()'s return tuple + info dict,
     not the env._last_reward / env._task_failed attributes — SurRoLTissueEnv
     recomputes its dense reward after tissue effects and returns it fresh;
     reading the return value avoids stale-attribute coupling.
  3. Exploration randomness uses an instance np.random.default_rng() created
     once in __init__ — never np.random.* globals (SEED REPRODUCIBILITY
     INVARIANT hygiene: no global RNG state touches).
  4. WorldModel.update() contention policy: fine-tune holds locks for
     seconds while update()'s window is milliseconds; a skipped belief
     refresh costs one EMA sample. Print-and-proceed, never crash the task.
"""

from __future__ import annotations

import asyncio

import numpy as np

from backend.event_log import write_event
from backend.prediction_net import PredictedState
from backend.simulation import action_to_surrol

# Constants (QUICK_REFERENCE.md): MPC loop sleep 0.05s = 20 steps/sec max.
STEP = 0.1             # normalized-space step ≈ 1.5 grid cells / 15
MPC_LOOP_SLEEP_S = 0.05


class MPCAgent:
    """Greedy-over-predictions planner: sample 8 candidate actions, score
    each with the world model's one-step prediction, execute the argmax."""

    def __init__(self, world_model, env) -> None:
        # Duck-typed by design: production passes the WorldModel singleton +
        # primary SurROLTissueEnv; unit tests pass contract-honoring stubs.
        self._world_model = world_model
        self._env = env
        self._step_count = 0
        self._rng = np.random.default_rng()  # instance RNG — no global seed
        self._reported_errors: set[str] = set()

    # ------------------------------------------------------------------
    # Candidate generation (WORLD_MODEL.md structure, deviation #1 applied)
    # ------------------------------------------------------------------

    def _generate_candidates(self, state_vec: np.ndarray) -> list[np.ndarray]:
        ee = state_vec[768:770]    # normalized EE xy
        tgt = state_vec[776:778]   # normalized target col,row
        diff = tgt - ee
        d = diff / (np.linalg.norm(diff) + 1e-8)  # unit vector toward target

        def make_move(
            dx: float, dy: float, dz: float = 0.0,
            mode: int = 0, magnitude: float = 0.7,
        ) -> np.ndarray:
            a = np.zeros(12, dtype=np.float32)
            a[0], a[1], a[2] = dx, dy, dz
            a[7 + mode] = 1.0  # MOVE=0, GRASP=1, CAUTERIZE=2, WAIT=3
            a[11] = magnitude
            return a

        cos45, sin45 = 0.707, 0.707
        candidates = [
            make_move(d[0] * STEP, d[1] * STEP),                    # direct
            make_move(d[0] * STEP, d[1] * STEP, magnitude=0.4),     # gentle
            # flank +45°
            make_move((d[0] * cos45 - d[1] * sin45) * STEP,
                      (d[0] * sin45 + d[1] * cos45) * STEP),
            # flank −45°
            make_move((d[0] * cos45 + d[1] * sin45) * STEP,
                      (-d[0] * sin45 + d[1] * cos45) * STEP),
        ]

        # Cauterize if any bleeding is active; else a slight dz approach
        if state_vec[512:768].max() > 0.5:
            candidates.append(make_move(0.0, 0.0, mode=2, magnitude=1.0))
        else:
            candidates.append(make_move(d[0] * STEP, d[1] * STEP, dz=0.05))

        rx, ry = self._rng.uniform(-STEP, STEP, size=2)
        candidates.extend([
            make_move(float(rx), float(ry)),   # random exploration
            make_move(0.0, 0.0, dz=STEP),      # depth probe
            make_move(0.0, 0.0, mode=3),       # WAIT
        ])
        return candidates  # exactly 8

    # ------------------------------------------------------------------
    # Value function (WORLD_MODEL.md, verbatim weights)
    # ------------------------------------------------------------------

    def _compute_value(
        self, current_state_vec: np.ndarray, predicted: PredictedState
    ) -> float:
        curr_ee = current_state_vec[768:770]    # normalized xy
        tgt = current_state_vec[776:778]        # normalized xy
        pred_ee = predicted.pred_ee_norm[:2]
        curr_dist = float(np.linalg.norm(curr_ee - tgt))
        pred_dist = float(np.linalg.norm(pred_ee - tgt))
        progress = max(0.0, curr_dist - pred_dist) / 0.1  # positive = closer

        damage_penalty = max(
            0.0,
            float(current_state_vec[0:256].mean()) - float(predicted.tissue.mean()),
        )
        vessel_penalty = float(predicted.vessel_risk.mean())
        hard_penalty = 1.0 if predicted.damage_prob > 0.5 else 0.0

        return (
            0.4 * progress
            - 0.3 * damage_penalty
            - 0.2 * vessel_penalty
            - 100.0 * hard_penalty  # dominates all other terms
        )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        while True:
            try:
                state_vec = self._env.get_state_vector()

                # Plan under the shared predict lock so a fine-tune weight
                # swap can't interleave mid-candidate-loop. predict() is
                # CPU-bound → to_thread (EVENT LOOP NON-BLOCKING INVARIANT).
                async with self._world_model._predict_lock:
                    candidates = self._generate_candidates(state_vec)
                    values = [
                        self._compute_value(
                            state_vec,
                            await asyncio.to_thread(
                                self._world_model.network.predict, state_vec, a
                            ),
                        )
                        for a in candidates
                    ]
                best_action = candidates[int(np.argmax(values))]

                surrol_action = action_to_surrol(
                    best_action, int(self._world_model.belief.active_dof)
                )
                _obs, reward, done, info = await asyncio.to_thread(
                    self._env.step, surrol_action
                )
                next_state_vec = self._env.get_state_vector()

                # Deviation #4: skip-on-contention, never crash the task.
                try:
                    self._world_model.update(state_vec, best_action, next_state_vec)
                except Exception as exc:  # noqa: BLE001 — expected contention path
                    print(f"[mpc] update skipped: {exc}", flush=True)

                # Telemetry only — event log may be uninitialized in unit
                # tests; a lost step record must never kill the loop.
                try:
                    write_event(
                        "simulation_step",
                        sim_id="primary",
                        step=self._step_count,
                        reward=float(reward),
                        tissue_mean=float(next_state_vec[0:256].mean()),
                        vessel_damaged=bool(next_state_vec[779 + 3 :: 4].max() > 0.5),
                        target_reached=bool(next_state_vec[778] > 0.5),
                        task_failed=bool(
                            info.get("task_failed") if isinstance(info, dict) else False
                        ),
                        active_dof=int(self._world_model.belief.active_dof),
                        ee_pos=next_state_vec[768:771].tolist(),
                    )
                except Exception:  # noqa: BLE001 — fire-and-forget telemetry
                    pass

                if done:
                    await asyncio.to_thread(self._env.reset)
                    self._step_count = 0
                else:
                    self._step_count += 1

            except Exception as exc:  # noqa: BLE001 — long-lived task guard:
                # transient startup/env errors must never kill the task
                # (ARCHITECTURE.md graceful degradation). Once per error TYPE
                # keeps a persistent failure from flooding the console.
                err_type = type(exc).__name__
                if err_type not in self._reported_errors:
                    self._reported_errors.add(err_type)
                    print(f"[mpc] transient: {exc}", flush=True)

            # MANDATORY LAST LINE of loop body — yield even after errors.
            await asyncio.sleep(MPC_LOOP_SLEEP_S)
