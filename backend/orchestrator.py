"""LangGraph orchestrator — SOMA Task 1.6.

Spec: docs/specs/ORCHESTRATION.md "LANGGRAPH GRAPH" (near-verbatim) +
AGENTS.md "CONSTANTS". Reads WorldModel.belief every cycle (Architecture Law
Exception #2), decides unlock vs explore vs wait, fans out ExplorationAgents
via Send, and tracks circuit-breaker state across cycles in graph state.

Task 1.9 invocation (validated by tests/test_orchestrator.py wiring smokes):
the graph NEVER reaches END — collect_status loops back to orchestrator by
design. Drive it in bounded bursts from main.py:

    graph = build_soma_graph(world_model, sim_manager)
    config = {"recursion_limit": 100, "configurable": {"thread_id": "soma"}}
    while True:
        try:
            await graph.ainvoke(initial_state(), config=config)
        except GraphRecursionError:
            continue  # checkpointed state persists via thread_id

DEVIATIONS from spec (all recorded for MASTER_STATE.md):
  1. Extra state field `agent_region` + module-level `_AGENT_REGIONS` /
     `_PROCESSED_OUTCOMES`. Spawn nodes return Send objects and therefore
     cannot write state, so agent→region lives in a module registry mirrored
     into state each pass (single writer per superstep — no conflicts).
  2. Spec's `completed_agents: []` reducer reset is omitted: under
     `operator.add` appending [] is a no-op, so the lists are treated as
     cumulative; drains use the full done-set and collect_status dedupes
     outcome processing via `_PROCESSED_OUTCOMES`.
  3. Duplicate-region exclusion uses `_AGENT_REGIONS` instead of the spec's
     `region in aid` substring check — agent ids are hex (`exp_{6hex}`) and
     can never contain a region name, so the spec check never fires.
  4. `exploration_agent_runner` is registered as a node (LangGraph requires
     Send targets to exist at compile time) and lazy-imports ExplorationAgent
     (Task 1.7a). Until it lands, runners record agent_failed so the circuit
     breaker sees real outcomes instead of ghost agents.
  5. `prediction_error_high` events are not in the frontend registry;
     the canvas ignores them silently (spec-mandated write, kept).
  6. Spec shows nodes returning a bare `list[Send]`; langgraph 1.2.11
     rejects that (InvalidUpdateError) and requires `Command(go_to=...)`.
     spawn_exploration_node keeps the documented list[Send] contract for
     direct callers/tests; a thin adapter inside build_soma_graph converts
     it to Command at the graph boundary.
  7. Spec's static edge spawn_explore→collect_status is omitted: with
     Command(goto=[Send...]) langgraph 1.x follows BOTH routes, double-
     scheduling collect_status into the same superstep as orchestrator
     (concurrent writes to LastValue keys → InvalidUpdateError). The
     no-candidate case is covered by the fallback Send("collect_status").
"""

# allow: SIZE_OK — single-file mandate (FILE OWNERSHIP MAP: orchestrator.py
# owns graph + route + nodes); ~85 of 302 lines are spec/task-mandated
# deviation log, invariant references, and Task 1.9 invocation docs.

from __future__ import annotations

import asyncio
import operator
import time
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send

from backend.event_log import write_event

# ---------------------------------------------------------------------------
# Constants (QUICK_REFERENCE.md — change one, update both)
# ---------------------------------------------------------------------------

EXPLORE_THRESHOLD = 0.15      # regional error that triggers agent spawn
MAX_CONCURRENT_AGENTS = 4     # exploration agents max (6 total sims ≤ 8 limit)
ORCHESTRATOR_SLEEP_S = 2.0    # seconds between decision cycles
DOF_THRESHOLDS: dict[int, float] = {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.04, 5: 0.03}
CIRCUIT_BREAKER_THRESHOLD = 3   # consecutive failures before breaker opens
CIRCUIT_BREAKER_BACKOFF_S = 300  # 5-minute region cooldown once open
EXPLORE_N_STEPS = 100           # steps per exploration episode (spawn payload)

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class SomaOrchestratorState(TypedDict):
    regional_errors: dict[str, float]
    global_mean_error: float
    active_dof: int
    active_agents: list[str]
    cycle_count: int
    # Reducer channels: parallel Send branches append without clobbering.
    completed_agents: Annotated[list[str], operator.add]
    failed_agents: Annotated[list[str], operator.add]
    unlock_requested: bool
    target_dof: int
    region_failure_counts: dict[str, int]
    region_backoff_until: dict[str, float]
    agent_region: dict[str, str]  # deviation 1


class ExplorationSpawn(TypedDict):
    """Payload routed to one exploration_agent_runner branch via Send."""

    agent_id: str  # format: 'exp_{6 hex chars}'
    region: str
    error_before: float


# Process-lifetime registries (deviation 1). Safe to lose on restart:
# MemorySaver drops graph state on process death anyway.
_AGENT_REGIONS: dict[str, str] = {}
_PROCESSED_OUTCOMES: set[str] = set()


def initial_state() -> SomaOrchestratorState:
    return SomaOrchestratorState(
        regional_errors={},
        global_mean_error=1.0,
        active_dof=1,
        active_agents=[],
        cycle_count=0,
        completed_agents=[],
        failed_agents=[],
        unlock_requested=False,
        target_dof=1,
        region_failure_counts={},
        region_backoff_until={},
        agent_region={},
    )


# ---------------------------------------------------------------------------
# Routing (pure function of state)
# ---------------------------------------------------------------------------


def route_orchestrator(state: SomaOrchestratorState) -> str:
    """Priority 1: DOF unlock. Priority 2: exploration. Default: wait."""
    unlock_thresh = DOF_THRESHOLDS.get(state["active_dof"], 1.0)
    already_unlocking = any(a.startswith("cap_") for a in state["active_agents"])
    if (
        state["global_mean_error"] < unlock_thresh
        and state["active_dof"] < 6
        and not already_unlocking
    ):
        return "unlock_dof"

    now = time.time()
    candidates = [
        (error, region)
        for region, error in state["regional_errors"].items()
        if error >= EXPLORE_THRESHOLD
        and not any(region in aid for aid in state["active_agents"])
        and now >= state["region_backoff_until"].get(region, 0.0)
        and state["region_failure_counts"].get(region, 0) < CIRCUIT_BREAKER_THRESHOLD
    ]
    if candidates and len(state["active_agents"]) < MAX_CONCURRENT_AGENTS:
        return "spawn_explore"

    return "wait"


# ---------------------------------------------------------------------------
# Nodes — module-level where no injected deps are needed
# ---------------------------------------------------------------------------


async def spawn_exploration_node(
    state: SomaOrchestratorState,
) -> list[Send]:
    """Fan out one Send per spawnable region, highest error first.

    AGENT_SPAWNED_BEFORE_LOOP INVARIANT: agent_spawned is written HERE,
    before the Send issues the agent's task — never inside the agent.
    """
    now = time.time()
    busy_regions = {_AGENT_REGIONS.get(aid, "") for aid in state["active_agents"]}
    candidates = sorted(
        [
            (error, region)
            for region, error in state["regional_errors"].items()
            if error >= EXPLORE_THRESHOLD
            and region not in busy_regions  # deviation 3
            and now >= state["region_backoff_until"].get(region, 0.0)
            and state["region_failure_counts"].get(region, 0) < CIRCUIT_BREAKER_THRESHOLD
        ],
        reverse=True,  # highest error first
    )

    slots = MAX_CONCURRENT_AGENTS - len(state["active_agents"])
    sends: list[Send] = []
    for error, region in candidates[:slots]:
        agent_id = f"exp_{uuid4().hex[:6]}"
        _AGENT_REGIONS[agent_id] = region
        write_event(
            "agent_spawned",
            agent_id=agent_id,
            parent_id="orchestrator",
            region=region,
            error_before=error,
            planned_steps=EXPLORE_N_STEPS,
        )
        sends.append(
            Send(
                "exploration_agent_runner",
                ExplorationSpawn(agent_id=agent_id, region=region, error_before=error),
            )
        )

    return sends or [Send("collect_status", state)]


async def exploration_agent_runner(payload: ExplorationSpawn) -> dict[str, list[str]]:
    """Runs one ExplorationAgent episode; moves the id to done-channels.

    Lazy import (deviation 4): keeps the graph compilable before Task 1.7a.
    """
    agent_id = payload["agent_id"]
    region = payload["region"]
    try:
        from backend.agents.exploration import ExplorationAgent
    except ImportError:
        write_event(
            "agent_failed",
            agent_id=agent_id,
            region=region,
            error="exploration module not implemented yet",
        )
        return {"failed_agents": [agent_id]}

    try:
        agent = ExplorationAgent(
            agent_id=agent_id, region=region, error_before=payload["error_before"]
        )
        result = await agent.run()
    except Exception as exc:
        # Boundary catch: a crashed agent must degrade to failed_agents,
        # never take down the orchestrator loop.
        write_event("agent_failed", agent_id=agent_id, region=region, error=str(exc))
        return {"failed_agents": [agent_id]}

    if bool(result.get("success", False)):
        return {"completed_agents": [agent_id]}
    return {"failed_agents": [agent_id]}


async def collect_status_node(
    state: SomaOrchestratorState,
) -> SomaOrchestratorState:
    """Applies circuit-breaker bookkeeping, then sleeps one cycle interval."""
    failure_counts = dict(state["region_failure_counts"])
    backoff_until = dict(state["region_backoff_until"])
    agent_region = {**state["agent_region"], **_AGENT_REGIONS}

    for agent_id in state["failed_agents"]:
        if agent_id in _PROCESSED_OUTCOMES:
            continue
        _PROCESSED_OUTCOMES.add(agent_id)
        region = agent_region.get(agent_id)
        if region:
            failure_counts[region] = failure_counts.get(region, 0) + 1
            if failure_counts[region] >= CIRCUIT_BREAKER_THRESHOLD:
                backoff_until[region] = time.time() + CIRCUIT_BREAKER_BACKOFF_S
                write_event(
                    "prediction_error_high",
                    region=region,
                    error="circuit_breaker_open",
                    failure_count=failure_counts[region],
                )

    for agent_id in state["completed_agents"]:
        if agent_id in _PROCESSED_OUTCOMES:
            continue
        _PROCESSED_OUTCOMES.add(agent_id)
        region = agent_region.get(agent_id)
        if region:
            failure_counts.pop(region, None)
            backoff_until.pop(region, None)

    await asyncio.sleep(ORCHESTRATOR_SLEEP_S)

    return {
        "region_failure_counts": failure_counts,
        "region_backoff_until": backoff_until,
        "agent_region": agent_region,
    }


# ---------------------------------------------------------------------------
# Graph assembly — world_model / sim_manager injected via closure
# ---------------------------------------------------------------------------


def build_soma_graph(
    world_model: Any,
    sim_manager: Any = None,
) -> CompiledStateGraph:
    """Compiles the SOMA graph; sim_manager needs .primary/.comparison envs."""

    async def orchestrator_node(state: SomaOrchestratorState) -> dict:
        regional = dict(world_model.belief.get_regional_errors())
        global_e = float(world_model.belief.prediction_error_map.mean())

        done = set(state["completed_agents"]) | set(state["failed_agents"])
        still_active = [aid for aid in state["active_agents"] if aid not in done]
        # Send payloads bypass state channels — register spawned agents here
        # so capacity accounting sees them from the cycle after their spawn.
        known = done | set(still_active)
        for aid in _AGENT_REGIONS:
            if aid not in known:
                still_active.append(aid)

        return {
            "regional_errors": regional,
            "global_mean_error": global_e,
            "active_dof": int(world_model.belief.active_dof),
            "active_agents": still_active,
            "cycle_count": state["cycle_count"] + 1,
            "agent_region": {**state["agent_region"], **_AGENT_REGIONS},
        }

    async def unlock_dof_node(state: SomaOrchestratorState) -> dict:
        new_dof = state["active_dof"] + 1
        cap_id = f"cap_{new_dof}"
        _AGENT_REGIONS.setdefault(cap_id, "")  # no region → breaker skips caps
        write_event(
            "agent_spawned",
            agent_id=cap_id,
            parent_id="orchestrator",
            reason="dof_unlock",
        )
        update: dict[str, Any] = {"unlock_requested": True, "target_dof": new_dof}

        try:
            from backend.agents.capability import CapabilityAgent
        except ImportError:
            return {**update, "failed_agents": [cap_id]}

        primary = getattr(sim_manager, "primary", None) if sim_manager else None
        comparison = getattr(sim_manager, "comparison", None) if sim_manager else None
        try:
            if primary is None or comparison is None:
                raise RuntimeError(
                    "sim_manager must expose .primary/.comparison SurRoL envs"
                )
            agent = CapabilityAgent(cap_id, world_model, primary, comparison)
            result = await agent.run()
        except Exception as exc:
            # Boundary catch: unlock intent stays recorded; the next cycle's
            # high-error map re-triggers the unlock (spec self-correction).
            write_event("agent_failed", agent_id=cap_id, error=str(exc))
            return {**update, "failed_agents": [cap_id]}

        if bool(result.get("success", False)):
            return {**update, "completed_agents": [cap_id]}
        return {**update, "failed_agents": [cap_id]}

    builder = StateGraph(SomaOrchestratorState)

    async def spawn_explore_entry(state: SomaOrchestratorState) -> Command:
        # Deviation 6: langgraph 1.x needs Command(go_to=...) for fan-out;
        # the spec's list[Send] contract lives in spawn_exploration_node.
        return Command(goto=await spawn_exploration_node(state))

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("spawn_explore", spawn_explore_entry)
    builder.add_node("unlock_dof", unlock_dof_node)
    builder.add_node("collect_status", collect_status_node)
    builder.add_node("exploration_agent_runner", exploration_agent_runner)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "spawn_explore": "spawn_explore",
            "unlock_dof": "unlock_dof",
            "wait": "collect_status",
        },
    )
    builder.add_edge("unlock_dof", "collect_status")
    builder.add_edge("exploration_agent_runner", "collect_status")
    builder.add_edge("collect_status", "orchestrator")

    # MemorySaver: in-process state, lost on restart — acceptable for demo
    # (weights + event log persist to disk). No END edges: see docstring.
    return builder.compile(checkpointer=MemorySaver())
