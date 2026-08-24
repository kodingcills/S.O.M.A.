"""CapabilityAgent — SOMA Task 1.7b.

Unlocks a DOF level, collects focused training data demonstrating the new
DOF, and fine-tunes the world model. Spec: docs/specs/ORCHESTRATION.md
"CAPABILITY AGENT" (run() is a near-verbatim transcription).

Does not run an episode loop. Does not call Claude API.

DOF unlock is NOT rolled back on failure (spec-intentional): if fine-tuning
times out, the simulation keeps the new DOF while the world model has high
error on it — the orchestrator detects this and spawns exploration agents.
The system self-corrects.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TypedDict

import numpy as np

from backend.event_log import write_event
from backend.simulation import (
    SimConfig,
    SurROLTissueEnv,
    action_to_surrol,
    create_env,
)

try:
    import wandb
except ImportError:  # pragma: no cover - wandb optional in dev envs
    wandb = None

# Constants (QUICK_REFERENCE.md). No shared constants module exists yet —
# defined locally per build instruction; orchestrator must keep values in sync.
AGENT_TIMEOUT_SECONDS = 600  # 10-minute hard kill
DOF_THRESHOLDS: dict[int, float] = {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.04, 5: 0.03}

Sample = tuple[np.ndarray, np.ndarray, np.ndarray]  # (state806, action12, next_state806)


class CapabilityInput(TypedDict):
    agent_id: str    # format: 'cap_{new_dof}'
    target_dof: int


class CapabilityOutput(TypedDict):
    agent_id: str
    target_dof: int
    samples_collected: int
    fine_tune_loss: float
    success: bool


class CapabilityAgent:
    """Unlocks one DOF, demonstrates it with 1,000 guided steps, fine-tunes."""

    def __init__(
        self,
        agent_id: str,
        world_model,
        primary_sim: SurROLTissueEnv,
        comparison_sim: SurROLTissueEnv,
        *,
        target_dof: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        # agent_id contract is 'cap_{dof}' — derive unless given explicitly.
        self.target_dof = (
            target_dof if target_dof is not None else int(agent_id.split("_")[1])
        )
        self._world_model = world_model
        self._primary_sim = primary_sim
        self._comparison_sim = comparison_sim

    # ------------------------------------------------------------------
    # Data collection (guided random policy exercising the new DOF)
    # ------------------------------------------------------------------

    async def collect_dof_samples(
        self, env: SurROLTissueEnv, n: int = 1000
    ) -> list[Sample]:
        """Collect n (state, action, next_state) triples exercising the new DOF.

        Deterministic via RandomState(42 + target_dof) — never global seeds.
        Components gated on the unlocked level: x-delta from DOF 2, rpy delta
        from DOF 3, alternating gripper from DOF 4; z movement always active.
        """
        rng = np.random.RandomState(42 + self.target_dof)
        samples: list[Sample] = []

        await asyncio.to_thread(env.reset)

        for step in range(n):
            state_vec = env.get_state_vector()

            vec = np.zeros(12, dtype=np.float32)
            vec[0] = rng.uniform(-0.3, 0.3) if self.target_dof >= 2 else 0.0
            vec[2] = rng.uniform(-0.3, 0.3)          # z movement always
            if self.target_dof >= 3:
                vec[3:6] = rng.uniform(-0.3, 0.3, size=3)
            if self.target_dof >= 4:
                vec[6] = 1.0 if step % 2 == 0 else 0.0   # gripper alternate
            vec[7] = 1.0                              # MOVE mode one-hot
            vec[11] = rng.uniform(0.3, 0.9)

            await asyncio.to_thread(
                env.step, action_to_surrol(vec, env.config.dof)
            )
            next_state_vec = env.get_state_vector()
            samples.append((state_vec, vec, next_state_vec))

        return samples

    # ------------------------------------------------------------------
    # Full run (ORCHESTRATION.md "CAPABILITY AGENT", verbatim structure)
    # ------------------------------------------------------------------

    async def run(self) -> CapabilityOutput:
        """Unlocks a DOF, collects focused training data, and fine-tunes.

        Returns:
            CapabilityOutput with success flag.
        """
        try:
            async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
                # Update DOF atomically on both simulations + belief.
                # SPEC DEVIATION: SimConfig is frozen; replace the config
                # instead of assigning .dof (env.config is a plain attribute).
                self._primary_sim.config = dataclasses.replace(
                    self._primary_sim.config, dof=self.target_dof
                )
                self._comparison_sim.config = dataclasses.replace(
                    self._comparison_sim.config, dof=self.target_dof
                )
                self._world_model.belief.active_dof = self.target_dof

                write_event(
                    "capability_unlocked",
                    agent_id=self.agent_id,
                    new_dof=self.target_dof,
                    previous_dof=self.target_dof - 1,
                    trigger_error=float(
                        self._world_model.belief.prediction_error_map.mean()
                    ),
                    threshold=DOF_THRESHOLDS[self.target_dof - 1],
                )

                # Collect DOF-focused data using a distinct seed for variety.
                # seed = 42 + dof gives 6 deterministic but distinct anatomies.
                dof_env = await create_env(SimConfig(
                    seed=42 + self.target_dof,
                    dof=self.target_dof,
                    n_vessels=self._primary_sim.config.n_vessels,
                ))
                try:
                    samples = await self.collect_dof_samples(dof_env, n=1000)
                finally:
                    await asyncio.to_thread(dof_env.close)

                # More epochs than exploration fine-tune: new DOF is completely
                # unseen by the network and requires a larger update.
                loss = await self._world_model.fine_tune(samples, epochs=10)
                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        "active_dof": self.target_dof,
                        "unlock_error": float(
                            self._world_model.belief.prediction_error_map.mean()
                        ),
                    })

            write_event("agent_completed", agent_id=self.agent_id,
                        samples_collected=len(samples), fine_tune_loss=loss)
            return CapabilityOutput(agent_id=self.agent_id,
                                    target_dof=self.target_dof,
                                    samples_collected=len(samples),
                                    fine_tune_loss=loss, success=True)

        except TimeoutError:
            write_event("agent_failed", agent_id=self.agent_id, error="timeout")
            return CapabilityOutput(agent_id=self.agent_id,
                                    target_dof=self.target_dof,
                                    samples_collected=0, fine_tune_loss=0.0,
                                    success=False)
