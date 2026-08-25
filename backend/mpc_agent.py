from __future__ import annotations

import asyncio
from typing import Final

import numpy as np

from backend import rerun_logger
from backend.event_log import write_event
from backend.simulus.instrumentation import InstrumentedActionOutput
from backend.simulus.soma_bridge import achievements_to_dof

MPC_LOOP_SLEEP_S: Final = 0.05
JSD_MAX: Final = float(np.log(2.0))


class MPCAgent:
    def __init__(self, world_model, driver) -> None:
        self._world_model = world_model
        self._driver = driver

    @staticmethod
    def _score(output: InstrumentedActionOutput) -> float:
        reward = float(output.reward_expectation)
        normalized_jsd = float(np.clip(float(output.J_ua) / JSD_MAX, 0.0, 1.0))
        return reward - 0.5 * normalized_jsd

    async def run_cycle(self) -> None:
        stamped = await asyncio.to_thread(
            self._driver.evaluate_stamped_candidates
        )
        selected = max(stamped, key=lambda candidate: self._score(candidate.output))
        step = await asyncio.to_thread(
            self._driver.step_with_action, int(selected.output.action_idx)
        )
        achievements = np.asarray(
            self._driver.env.achievements(), dtype=bool
        )
        write_event(
            "simulation_step",
            sim_id="primary",
            step=int(step.step),
            action=int(step.action),
            reward=float(step.reward),
            done=bool(step.done),
            stats=dict(step.stats),
            achievements=achievements.tolist(),
            n_achievements=int(achievements.sum()),
            active_dof=achievements_to_dof(achievements),
        )
        if step.done:
            await asyncio.to_thread(self._driver.reset, 42)
        telemetry = getattr(rerun_logger, "log_craftax_step", None)
        if callable(telemetry):
            try:
                telemetry(
                    "primary",
                    int(step.step),
                    int(step.action),
                    float(step.reward),
                    dict(step.stats),
                    achievements.tolist(),
                    stamped,
                )
            except (OSError, RuntimeError):
                return

    async def run(self) -> None:
        while True:
            await self.run_cycle()
            await asyncio.sleep(MPC_LOOP_SLEEP_S)
