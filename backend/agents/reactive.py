"""ReactiveAgent — SOMA Task 1.7b.

Greedy surgical policy for the comparison simulation. Spec:
docs/specs/ORCHESTRATION.md "REACTIVE AGENT" (near-verbatim transcription).

REACTIVE_AGENT_ISOLATION INVARIANT (ARCHITECTURE.md): zero imports from
world_model, prediction_net, belief_state, orchestrator, anthropic —
grep-enforced by tests/test_agents_capability_reactive.py. This agent must
have no access to the world model to be a valid comparison baseline.
"""

from __future__ import annotations

import asyncio

import numpy as np

from backend.event_log import write_event
from backend.simulation import SurROLTissueEnv, action_to_surrol


class ReactiveAgent:
    """Greedy surgical policy for the comparison simulation.

    Moves directly toward the surgical target. Cauterizes when bleeding.
    Has no model of vessel locations relative to its approach path.
    This agent will damage vessels — that is the intended behavior.

    Attributes:
        _env: The comparison SurRoL environment. Never the primary sim.
    """

    def __init__(self, env: SurROLTissueEnv) -> None:
        self._env = env  # comparison_sim only. No world_model parameter.

    async def run(self) -> None:
        """Runs the reactive policy indefinitely until task cancellation.

        # LOOP: reactive_policy
        # Pre-condition:  comparison_sim environment initialized and reset.
        # Invariant:      simulation_step event written after every env.step().
        # Termination:    cancelled via task.cancel() on application shutdown.
        # Yield:          await asyncio.sleep(0.05) releases event loop each step.
        """
        await asyncio.to_thread(self._env.reset)
        step = 0

        while True:
            state_vec = self._env.get_state_vector()
            action_vec = self._select_action(state_vec)
            surrol_action = action_to_surrol(action_vec, self._env.config.dof)

            _, reward, done, _ = await asyncio.to_thread(
                self._env.step, surrol_action
            )
            write_event(
                "simulation_step",
                sim_id="comparison",
                step=step,
                reward=float(reward),
                tissue_mean=float(state_vec[0:256].mean()),
                vessel_damaged=bool(state_vec[779 + 3 :: 4].max() > 0.5),
                target_reached=bool(state_vec[778] > 0.5),
                task_failed=bool(getattr(self._env, "_task_failed", False)),
                active_dof=int(self._env.config.dof),
                ee_pos=state_vec[768:771].tolist(),
            )
            step += 1

            if done:
                await asyncio.to_thread(self._env.reset)
                step = 0

            await asyncio.sleep(0.05)  # yield event loop — see LOOP comment

    def _select_action(self, state_vec: np.ndarray) -> np.ndarray:
        """Greedy action: move toward target; cauterize if bleeding.

        Args:
            state_vec: Current environment state vector, shape (806,).

        Returns:
            Action vector shape (12,) with one-hot action mode set.
        """
        ee = state_vec[768:770]    # normalized EE x, y
        tgt = state_vec[776:778]   # normalized target col, row
        bleed = state_vec[512:768].sum()

        vec = np.zeros(12, dtype=np.float32)

        if bleed > 0:
            # Cauterize at current position when bleeding is active.
            vec[9] = 1.0   # CAUTERIZE mode
            vec[11] = 1.0
        else:
            direction = tgt - ee
            dist = float(np.linalg.norm(direction)) + 1e-8
            unit = direction / dist
            speed = 0.1 if dist > 0.1 else 0.05  # slow near target
            vec[0:2] = unit * speed
            vec[7] = 1.0   # MOVE mode
            vec[11] = 0.7 if dist > 0.1 else 0.4

        return vec
