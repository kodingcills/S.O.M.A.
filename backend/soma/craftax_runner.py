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
from backend.soma.collision import CollisionAnalyzer, CollisionResult
from backend.soma.state_machine import SomaStateMachine

RUNNER_SEED: Final = 42


async def run_craftax_loop(
    driver: CraftaxDriver,
    instrumented: SimulusInstrumented,
    state_machine: SomaStateMachine,
    n_audit_every: int = 10,
) -> None:
    collision_analyzer = CollisionAnalyzer(instrumented)
    missed_results = []
    null_results = []
    await asyncio.to_thread(driver.reset, RUNNER_SEED)
    episode_return = 0.0

    while True:
        stamped = await asyncio.to_thread(driver.evaluate_stamped_candidates)
        chosen = max(stamped, key=lambda candidate: MPCAgent._score(candidate.output))
        jsd_mean = float(np.mean([candidate.output.J_ua for candidate in stamped]))
        pre_state, pre_key = await asyncio.to_thread(driver.env.snapshot)
        step = await asyncio.to_thread(
            driver.step_with_action, int(chosen.output.action_idx)
        )
        episode_return += step.reward

        audit_fields = {}
        if step.step % n_audit_every == 0:
            record = await asyncio.to_thread(
                audit_decision_point,
                stamped,
                chosen,
                step,
                driver.env,
                pre_state,
                pre_key,
                jsd_mean=jsd_mean,
            )
            state_machine.record_harm(record.harmful)

            jsd_missed = record.harmful and record.jsd_mean < 0.015

            b_ua = collision_analyzer.extract_b_ua(chosen.output)

            if b_ua is None:
                for s in stamped:
                    extracted = collision_analyzer.extract_b_ua(s.output)
                    if extracted is not None:
                        b_ua = extracted
                        chosen = s
                        break

            collision_result = CollisionResult(False, 0.0, None, 0.0)
            if b_ua is not None:
                collision_analyzer.add_to_database(
                    b_ua,
                    record.chosen_action_idx,
                    max(record.q_p_values.values()),
                )

                collision_result = collision_analyzer.is_collision_candidate(
                    b_missed=b_ua,
                    a_missed_optimal=record.chosen_action_idx,
                )

                payload = {
                    "collision_candidate": collision_result.is_candidate,
                    "min_distance": collision_result.min_distance,
                    "jsd_missed": jsd_missed,
                }
            else:
                payload = {}

            if jsd_missed:
                missed_results.append(collision_result)
            else:
                null_results.append(collision_result)

            if len(missed_results) % 5 == 0 and len(missed_results) >= 5:
                enrichment = collision_analyzer.compute_enrichment(
                    missed_results, null_results
                )
                import json

                from pathlib import Path

                Path("backend/research").mkdir(exist_ok=True)
                Path("backend/research/h2_result.json").write_text(
                    json.dumps(enrichment, indent=2)
                )
                write_event(
                    "graph_node_added",
                    payload=json.dumps(
                        {
                            "phase": "h2_update",
                            "label": f"H2: {enrichment['verdict']} "
                            f"(n_missed={len(missed_results)})",
                            "enrichment_ratio": enrichment["enrichment_ratio"],
                            "verdict": enrichment["verdict"],
                            "finding": enrichment["finding"],
                        }
                    ),
                )

            audit_fields = {
                "regret": record.regret,
                "harmful": record.harmful,
            }

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
            **audit_fields,
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