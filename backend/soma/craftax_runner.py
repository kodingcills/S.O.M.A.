"""Run the continuously audited Craftax control loop.

Performance deviation: the runner uses ``driver.evaluate_stamped_candidates``
for one batched forward pass per iteration. Its outputs are numerically
identical to sequential ``instrumented.evaluate_all_actions`` evaluation, as
locked by ``tests/test_instrumentation.py::test_batched_matches_sequential``.
"""

from __future__ import annotations

import asyncio
from typing import Final

import numpy as np

from backend.craftax_driver import CraftaxDriver
from backend.event_log import write_event
from backend.mpc_agent import MPCAgent
from backend.simulus.instrumentation import SimulusInstrumented
from backend.soma.audit import audit_decision_point
from backend.soma.state_machine import SomaStateMachine

RUNNER_SEED: Final = 42


async def run_craftax_loop(
    driver: CraftaxDriver,
    instrumented: SimulusInstrumented,
    state_machine: SomaStateMachine,
    n_audit_every: int = 10,
) -> None:
    """Continuously evaluate, execute, audit, and report Craftax episodes."""
    del instrumented
    await asyncio.to_thread(driver.reset, RUNNER_SEED)
    episode_return = 0.0

    while True:
        stamped = await asyncio.to_thread(driver.evaluate_stamped_candidates)
        chosen = max(stamped, key=lambda candidate: MPCAgent._score(candidate.output))
        jsd_mean = float(np.mean([candidate.output.J_ua for candidate in stamped]))
        step = await asyncio.to_thread(
            driver.step_with_action, int(chosen.output.action_idx)
        )
        episode_return += step.reward

        if step.step % n_audit_every == 0:
            record = audit_decision_point(stamped, chosen, step)
            state_machine.record(record.predicted_reward_expectation, step.reward)

        if state_machine.should_transition_to_gap():
            info = state_machine.advance_phase()
            write_event(
                "graph_node_added",
                node_id=info.node_id,
                phase=info.phase,
                n_z_u=info.n_z_u,
                n_harmful=info.n_harmful,
            )

        write_event(
            "craftax_step",
            episode=int(driver.episode_generation),
            step=int(step.step),
            jsd_mean=jsd_mean,
            score=float(MPCAgent._score(chosen.output)),
            action=int(step.action),
            reward=float(step.reward),
            done=bool(step.done),
        )

        if step.done:
            achievements = int(
                np.asarray(driver.env.achievements(), dtype=bool).sum()
            )
            write_event(
                "craftax_episode",
                episode=int(driver.episode_generation),
                score=float(episode_return),
                achievements=achievements,
                steps=int(step.step),
            )
            await asyncio.to_thread(driver.reset, RUNNER_SEED)
            episode_return = 0.0

        await asyncio.sleep(0)
