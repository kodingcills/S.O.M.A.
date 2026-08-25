from __future__ import annotations

import asyncio
from typing import Final, TypedDict

import numpy as np

from backend.event_log import write_event
from backend.simulus.soma_bridge import achievements_to_dof

AGENT_TIMEOUT_SECONDS = 600
DOF_THRESHOLDS: Final = {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.04, 5: 0.03}


class CapabilityInput(TypedDict):
    agent_id: str
    target_dof: int


class CapabilityOutput(TypedDict):
    agent_id: str
    target_dof: int
    samples_collected: int
    fine_tune_loss: float
    success: bool


class CapabilityAgent:
    def __init__(
        self,
        agent_id: str,
        world_model,
        driver,
        *,
        target_dof: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.target_dof = (
            int(target_dof)
            if target_dof is not None
            else int(agent_id.split("_", maxsplit=1)[1])
        )
        self._world_model = world_model
        self._driver = driver

    def _trigger_error(self) -> float:
        belief = self._world_model.belief
        if belief.regional_override is not None:
            return float(np.mean(tuple(belief.regional_override.values())))
        return float(belief.prediction_error_map.mean())

    async def run(self) -> CapabilityOutput:
        try:
            async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
                belief = self._world_model.belief
                previous_dof = int(belief.active_dof)
                achievement_dof = achievements_to_dof(
                    self._driver.env.achievements()
                )
                unlocked_dof = min(self.target_dof, achievement_dof)
                success = achievement_dof >= self.target_dof and unlocked_dof >= previous_dof
                if unlocked_dof > previous_dof:
                    belief.active_dof = unlocked_dof
                    write_event(
                        "capability_unlocked",
                        agent_id=self.agent_id,
                        new_dof=unlocked_dof,
                        previous_dof=previous_dof,
                        trigger_error=self._trigger_error(),
                        threshold=DOF_THRESHOLDS[previous_dof],
                    )
                write_event(
                    "agent_completed",
                    agent_id=self.agent_id,
                    samples_collected=0,
                    fine_tune_loss=0.0,
                )
                return CapabilityOutput(
                    agent_id=self.agent_id,
                    target_dof=unlocked_dof,
                    samples_collected=0,
                    fine_tune_loss=0.0,
                    success=success,
                )
        except TimeoutError:
            write_event("agent_failed", agent_id=self.agent_id, error="timeout")
            return CapabilityOutput(
                agent_id=self.agent_id,
                target_dof=self.target_dof,
                samples_collected=0,
                fine_tune_loss=0.0,
                success=False,
            )
