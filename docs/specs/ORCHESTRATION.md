# ORCHESTRATION.md
# SOMA — Runtime Agent Specification
# Authoritative for: LangGraph graph definition, agent contracts,
# orchestrator decision logic, ExplorationAgent (Claude API + tool use),
# CapabilityAgent, ReactiveAgent, agent lifecycle invariants,
# production patterns (circuit breaker, timeout, rate limiting,
# fine-tune queue, cost tracking, streaming, backpressure).
#
# Invariants referenced here are DEFINED in ARCHITECTURE.md.
# Read this before building: orchestrator.py, agents/exploration.py,
#                            agents/capability.py, agents/reactive.py.
# Build guidance (task routing, OMO commands): BUILD_PLAN.md + /AGENTS.md.

---

## LANGGRAPH GRAPH

**File:** `backend/orchestrator.py`

```bash
uv add langgraph langgraph-checkpoint langsmith
```

### State

Keep state JSON-serializable — LangGraph's MemorySaver serializes to JSON.
No numpy arrays, no dataclasses. Python primitives only.

List fields written by parallel branches use `Annotated[list, operator.add]`
so LangGraph merges results from concurrent Send nodes instead of
overwriting them. Omitting the annotation causes only the last writer's
result to appear in state.

```python
from typing import Annotated, TypedDict
import operator

class SomaOrchestratorState(TypedDict):
    # Set by orchestrator_node each cycle.
    regional_errors:    dict[str, float]  # exactly 6 keys — see BeliefState
    global_mean_error:  float
    active_dof:         int
    active_agents:      list[str]         # agent_ids currently running
    cycle_count:        int

    # Written by parallel fan-out nodes. Reducer appends across branches.
    completed_agents: Annotated[list[str], operator.add]
    failed_agents:    Annotated[list[str], operator.add]

    # Set by unlock_dof_node.
    unlock_requested: bool
    target_dof:       int

    # Circuit breaker — survives across cycles via MemorySaver checkpoint.
    region_failure_counts: dict[str, int]    # cumulative failures per region
    region_backoff_until:  dict[str, float]  # unix timestamp per region
```

### Graph Construction

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def build_soma_graph(
    world_model: WorldModel,
    primary_sim: SurROLTissueEnv,
    comparison_sim: SurROLTissueEnv,
) -> CompiledGraph:
    """Builds and compiles the SOMA orchestration graph.

    The graph runs autonomously once started. It reads belief state,
    decides whether to spawn exploration agents or unlock capability,
    executes the decision in parallel via Send, and loops every 2 seconds.

    Args:
        world_model: The WorldModel singleton. Passed via closure to nodes.
        primary_sim: Primary SurRoL environment (MPC agent operates here).
        comparison_sim: Comparison SurRoL environment (reactive baseline).

    Returns:
        Compiled LangGraph graph with MemorySaver checkpointer.
    """
    builder = StateGraph(SomaOrchestratorState)

    builder.add_node("orchestrator",   orchestrator_node)
    builder.add_node("spawn_explore",  spawn_exploration_node)
    builder.add_node("unlock_dof",     unlock_dof_node)
    builder.add_node("collect_status", collect_status_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "spawn_explore": "spawn_explore",
            "unlock_dof":    "unlock_dof",
            "wait":          "collect_status",
        },
    )
    builder.add_edge("spawn_explore",  "collect_status")
    builder.add_edge("unlock_dof",     "collect_status")
    builder.add_edge("collect_status", "orchestrator")

    # MemorySaver: in-process state. Lost on process restart.
    # Acceptable for demo — world model weights and event log persist to disk.
    # PostgresSaver would survive restarts but requires a running Postgres instance.
    return builder.compile(checkpointer=MemorySaver())
```

### Routing Function

Priority 1 — DOF unlock (rarer, higher value than exploration).
Priority 2 — exploration (spawn into highest-error available region).
Default — wait (sleep and cycle again).

```python
def route_orchestrator(state: SomaOrchestratorState) -> str:
    """Routes the orchestrator to its next action based on belief state.

    Checks DOF unlock eligibility before exploration to ensure capability
    expansion happens as soon as the world model is ready, rather than
    being delayed by active exploration agents.

    Args:
        state: Current orchestrator state with regional errors and agent counts.

    Returns:
        One of: "spawn_explore", "unlock_dof", "wait".
    """
    # Priority 1: unlock DOF if global error crossed the threshold.
    unlock_thresh = DOF_THRESHOLDS.get(state["active_dof"], 1.0)
    already_unlocking = any(a.startswith("cap_") for a in state["active_agents"])
    if (state["global_mean_error"] < unlock_thresh
            and state["active_dof"] < 6
            and not already_unlocking):
        return "unlock_dof"

    # Priority 2: spawn exploration agent into highest-error open region.
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
```

### Orchestrator Node

```python
async def orchestrator_node(
    state: SomaOrchestratorState,
) -> SomaOrchestratorState:
    """Refreshes belief state and removes completed agents from tracking.

    Direct call to WorldModel.belief — Architecture Law Exception #2.
    The orchestrator reads errors every 2 seconds. Event log mediation
    would introduce up to 2 seconds of lag, causing agents to spawn into
    already-resolved regions.

    Args:
        state: Current orchestrator state.

    Returns:
        Updated state with fresh belief state and cleaned agent tracking.
    """
    regional = world_model.belief.get_regional_errors()
    global_e = float(world_model.belief.prediction_error_map.mean())

    # Remove completed and failed agents from active tracking.
    # Reducer fields (completed_agents, failed_agents) carry over from
    # spawn nodes; drain them here before the next cycle.
    done = set(state["completed_agents"]) | set(state["failed_agents"])
    still_active = [a for a in state["active_agents"] if a not in done]

    return {
        **state,
        "regional_errors":  regional,
        "global_mean_error": global_e,
        "active_dof":       world_model.belief.active_dof,
        "active_agents":    still_active,
        "completed_agents": [],   # reset reducers for next cycle
        "failed_agents":    [],
        "cycle_count":      state["cycle_count"] + 1,
    }
```

### Spawn Exploration Node — Fan-Out via Send

```python
from langgraph.types import Send

async def spawn_exploration_node(
    state: SomaOrchestratorState,
) -> list[Send]:
    """Spawns one or more ExplorationAgent tasks in parallel via Send.

    Send is graph-native parallelism — the checkpointer journals each
    branch. asyncio.create_task() would run outside the graph; a process
    restart would lose those tasks with no record.

    agent_spawned is written here, before Send, so the canvas node appears
    before the agent starts running. Writing it inside the agent would make
    nodes appear mid-execution. See AGENT_SPAWNED_BEFORE_LOOP INVARIANT.

    Args:
        state: Current orchestrator state with regional errors.

    Returns:
        List of Send objects targeting "exploration_agent_runner".
    """
    now = time.time()
    candidates = sorted(
        [
            (err, region)
            for region, err in state["regional_errors"].items()
            if err >= EXPLORE_THRESHOLD
            and not any(region in aid for aid in state["active_agents"])
            and now >= state["region_backoff_until"].get(region, 0.0)
            and state["region_failure_counts"].get(region, 0) < CIRCUIT_BREAKER_THRESHOLD
        ],
        reverse=True,  # highest error first
    )

    slots = MAX_CONCURRENT_AGENTS - len(state["active_agents"])
    sends: list[Send] = []

    for error, region in candidates[:slots]:
        agent_id = f"exp_{uuid4().hex[:6]}"
        # Write agent_spawned BEFORE Send. Canvas requires this ordering.
        write_event(
            "agent_spawned",
            agent_id=agent_id,
            parent_id="orchestrator",
            region=region,
            error_before=error,
            planned_steps=EXPLORE_N_STEPS,
        )
        sends.append(Send("exploration_agent_runner", {
            "agent_id":    agent_id,
            "region":      region,
            "error_before": error,
        }))

    return sends or [Send("collect_status", state)]
```

### Collect Status Node

```python
async def collect_status_node(
    state: SomaOrchestratorState,
) -> SomaOrchestratorState:
    """Sleeps the orchestrator cycle interval, then yields to the next cycle.

    Updating circuit breaker state here (after agents report outcomes)
    keeps failure tracking in graph state, so MemorySaver persists it
    across the process lifetime.

    Args:
        state: State including completed_agents and failed_agents from fan-out.

    Returns:
        Updated state with circuit breaker counters adjusted.
    """
    # Update circuit breaker failure counts from this cycle's results.
    failure_counts = dict(state["region_failure_counts"])
    backoff_until  = dict(state["region_backoff_until"])

    for agent_id in state["failed_agents"]:
        region = _region_for_agent(agent_id)
        if region:
            failure_counts[region] = failure_counts.get(region, 0) + 1
            if failure_counts[region] >= CIRCUIT_BREAKER_THRESHOLD:
                backoff_until[region] = time.time() + CIRCUIT_BREAKER_BACKOFF_S

    # Reset counts for regions that succeeded this cycle.
    for agent_id in state["completed_agents"]:
        region = _region_for_agent(agent_id)
        if region:
            failure_counts.pop(region, None)
            backoff_until.pop(region, None)

    # LOOP: orchestrator_cycle
    # Pre-condition:  circuit breaker state updated for this cycle's outcomes.
    # Invariant:      orchestrator waits exactly ORCHESTRATOR_SLEEP_S between cycles.
    # Termination:    never — cancelled on shutdown via task.cancel().
    # Yield:          asyncio.sleep() releases the event loop.
    await asyncio.sleep(ORCHESTRATOR_SLEEP_S)

    return {
        **state,
        "region_failure_counts": failure_counts,
        "region_backoff_until":  backoff_until,
    }
```

---

## AGENT CONTRACTS

Explicit TypedDicts for every agent's input and output. Untyped `**kwargs`
produce events with missing fields that the frontend silently ignores —
the canvas shows missing nodes with no error.

```python
class ExplorationInput(TypedDict):
    agent_id:     str    # format: "exp_{6 hex chars}"
    region:       str    # one of the 6 named regions from BeliefState
    error_before: float  # regional error at spawn time

class ExplorationOutput(TypedDict):
    agent_id:          str
    region:            str
    error_before:      float
    error_after:       float
    samples_collected: int
    fine_tune_loss:    float
    api_tokens_used:   int   # cumulative Claude API tokens across episode
    success:           bool  # error_after < error_before - 0.05

class CapabilityInput(TypedDict):
    agent_id:   str  # format: "cap_{target_dof}"
    target_dof: int

class CapabilityOutput(TypedDict):
    agent_id:          str
    target_dof:        int
    samples_collected: int
    fine_tune_loss:    float
    success:           bool
```

---

## EXPLORATION AGENT

**File:** `backend/agents/exploration.py`

Collects focused simulation data in a high-error region, fine-tunes the
world model, and reports the error delta. Uses Claude API for action
selection — its reasoning appears in the agent feed live during the demo.

### Claude API Tool Definition

Structured tool use guarantees parseable output. Free-text JSON parsing
fails on Claude's explanatory preambles and markdown code blocks.
`tool_choice={"type": "any"}` forces a tool call on every response —
without it, Claude sometimes responds in prose.

```python
EXPLORE_TOOL = {
    "name": "select_action",
    "description": (
        "Select the next action for data collection in the target region. "
        "Prioritize actions that generate informative tissue dynamics data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["MOVE", "CAUTERIZE", "WAIT"],
            },
            "delta_x":   {"type": "number", "description": "EE x delta [-1, 1]"},
            "delta_y":   {"type": "number", "description": "EE y delta [-1, 1]"},
            "delta_z":   {"type": "number", "description": "EE z delta [-1, 1]"},
            "magnitude": {"type": "number", "description": "Action scale [0, 1]"},
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining this action choice.",
            },
        },
        "required": ["action_type", "reasoning"],
    },
}
```

### Rate Limiter

Module-level semaphore — shared across all concurrent exploration instances.
Peak rate: 4 agents × (10 steps/sec ÷ 5 steps/call) = 8 req/sec. Well
within tier limits, but startup bursts and retries can spike this.

```python
# Module-level. Initialized once. Shared by all ExplorationAgent instances.
_api_semaphore = asyncio.Semaphore(10)  # max 10 concurrent Claude API calls
```

### API Call — Streaming + Retry

```python
async def _call_claude(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    feed_callback: Callable[[str], None],
    max_retries: int = 3,
) -> tuple[dict, int]:
    """Calls Claude API with streaming and exponential-backoff retry.

    Streams reasoning tokens to feed_callback as they arrive, so the
    agent feed in the frontend updates token-by-token during the demo.
    Tool input is only parsed after stream.get_final_message() — never
    from intermediate stream.text_stream chunks (those are incomplete JSON).

    Args:
        client:        Anthropic async client.
        prompt:        Formatted exploration prompt.
        feed_callback: Called with each streamed reasoning token.
        max_retries:   Number of retry attempts on rate limit or connection error.

    Returns:
        Tuple of (parsed tool input dict, total tokens used).

    Raises:
        AgentAPIError: After max_retries exhausted, or if Claude doesn't call tool.
    """
    for attempt in range(max_retries):
        try:
            async with _api_semaphore:
                async with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=300,
                    tools=[EXPLORE_TOOL],
                    tool_choice={"type": "any"},  # force tool call every response
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for token in stream.text_stream:
                        feed_callback(token)
                    final = await stream.get_final_message()

            tool_block = next(
                (b for b in final.content if b.type == "tool_use"),
                None,
            )
            if tool_block is None:
                raise AgentAPIError("Claude responded without calling select_action.")
            tokens = final.usage.input_tokens + final.usage.output_tokens
            return tool_block.input, tokens

        except anthropic.RateLimitError:
            # Exponential backoff: 1 s, 2 s, 4 s.
            await asyncio.sleep(2 ** attempt)
        except anthropic.APIConnectionError:
            if attempt == max_retries - 1:
                raise AgentAPIError("Claude API unreachable after retries.")
            await asyncio.sleep(1.0)

    raise AgentAPIError(f"Claude API failed after {max_retries} retries.")
```

### Episode Loop

```python
async def run_episode(
    self,
    n_steps: int = EXPLORE_N_STEPS,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Runs one exploration episode, collecting (state, action, next_state) tuples.

    Calls Claude API every CLAUDE_CALL_EVERY_N steps; repeats the last
    action in between. This keeps API costs low while ensuring each decision
    covers enough steps to observe tissue dynamics.

    Args:
        n_steps: Total environment steps to collect.

    Returns:
        Tuple of (samples list, total_api_tokens).
    """
    samples: list[tuple] = []
    total_tokens = 0
    last_action = _default_move_action()

    await asyncio.to_thread(self.env.reset)

    # LOOP: exploration_episode
    # Pre-condition:  env.reset() completed; world_model weights loaded.
    # Invariant:      each (state, action, next_state) triple is consistent
    #                 with the environment's actual physics.
    # Termination:    step count reaches n_steps.
    # Yield:          asyncio.to_thread() yields for every env.step() call.
    for step in range(n_steps):
        state_vec = self.env.get_state_vector()

        if step % CLAUDE_CALL_EVERY_N == 0:
            prompt = self._build_prompt(state_vec, step)
            try:
                tool_input, tokens = await _call_claude(
                    self._client,
                    prompt,
                    lambda tok: self._buffer.append(tok),
                )
                last_action = _tool_input_to_action(tool_input)
                total_tokens += tokens
                self._buffer.flush(step)
            except AgentAPIError as exc:
                # Degraded mode: repeat last action and log the fallback.
                self._buffer.append(f"[step {step}] API fallback: {exc}")
                self._buffer.flush(step)

        _, _, done, _ = await asyncio.to_thread(
            self.env.step,
            action_to_surrol(last_action, self.env.config.dof),
        )
        next_state_vec = self.env.get_state_vector()
        samples.append((state_vec, last_action.copy(), next_state_vec))

        if step % 10 == 0:
            write_event("agent_step", agent_id=self.agent_id, step=step)

        if done:
            await asyncio.to_thread(self.env.reset)

    return samples, total_tokens
```

### Prompt Construction

```python
def _build_prompt(self, state_vec: np.ndarray, step: int) -> str:
    """Formats the per-step exploration prompt for Claude.

    Extracts human-readable values from the 806-element state vector.
    Indices from QUICK_REFERENCE.md state vector table.

    Args:
        state_vec: Current environment state vector, shape (806,).
        step:      Current step number within the episode.

    Returns:
        Prompt string ready for the Claude API messages array.
    """
    tissue_mean    = float(state_vec[0:256].mean())
    bleeding_count = int(state_vec[512:768].sum())
    ee_pos         = state_vec[768:771].tolist()
    active_dof     = int(np.argmax(state_vec[799:805])) + 1

    return (
        f"You are an exploration agent in a surgical robot simulation.\n"
        f"Goal: collect informative tissue dynamics data in '{self.region}'.\n\n"
        f"State:\n"
        f"  target_region:    {self.region}\n"
        f"  initial_error:    {self.error_before:.3f}\n"
        f"  step:             {step}/{EXPLORE_N_STEPS}\n"
        f"  tissue_integrity: {tissue_mean:.3f} (mean)\n"
        f"  bleeding_cells:   {bleeding_count}\n"
        f"  ee_position:      {ee_pos}\n"
        f"  active_dof:       {active_dof}\n\n"
        f"Strategy: vary action types within {self.region}. Prefer CAUTERIZE "
        f"when bleeding > 0. Prefer MOVE toward high-vascularity cells.\n\n"
        f"Call select_action with your chosen action and one-sentence reasoning."
    )
```

### Full Agent Run

```python
async def run(self) -> ExplorationOutput:
    """Runs the complete exploration pipeline: collect → fine-tune → measure.

    Uses asyncio.timeout for a hard 10-minute limit. Always closes the
    SurRoL environment in the finally block to release the PyBullet
    physics server. See SIMULATION RELEASED ON FAILURE INVARIANT.

    Returns:
        ExplorationOutput with error_before, error_after, and success flag.
    """
    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            samples, tokens = await self.run_episode()
            fine_tune_loss  = await world_model.fine_tune(samples)
            final_error     = await self._measure_region_error()

        write_event(
            "agent_completed",
            agent_id=self.agent_id,
            region=self.region,
            error_before=self.error_before,  # top-level field — not in payload
            error_after=final_error,          # top-level field — drives animation
            samples_collected=len(samples),
            fine_tune_loss=fine_tune_loss,
            api_tokens_used=tokens,
        )
        return ExplorationOutput(
            agent_id=self.agent_id, region=self.region,
            error_before=self.error_before, error_after=final_error,
            samples_collected=len(samples), fine_tune_loss=fine_tune_loss,
            api_tokens_used=tokens,
            success=(final_error < self.error_before - 0.05),
        )

    except TimeoutError:
        write_event("agent_failed", agent_id=self.agent_id,
                    region=self.region, error="timeout")
        return ExplorationOutput(
            agent_id=self.agent_id, region=self.region,
            error_before=self.error_before, error_after=self.error_before,
            samples_collected=0, fine_tune_loss=0.0,
            api_tokens_used=0, success=False,
        )

    except Exception as exc:
        write_event("agent_failed", agent_id=self.agent_id,
                    region=self.region, error=str(exc))
        raise

    finally:
        # Always release the SurRoL instance. Leaked instances hold a PyBullet
        # physics server (~200 MB each). 6 leaks → ~1.2 GB RAM exhaustion.
        await asyncio.to_thread(self.env.close)
```

### `_measure_region_error`

```python
async def _measure_region_error(self) -> float:
    """Computes the world model's post-fine-tune error on the target region.

    Runs 20 predict-then-observe cycles and takes the mean squared error
    of the tissue head predictions for the target region's cells only.
    This is the authoritative error_after value written to the event log.

    Returns:
        Mean squared prediction error for the target region, in [0.0, 1.0].
    """
    rows, cols = REGION_SLICE_MAP[self.region]
    errors: list[float] = []
    await asyncio.to_thread(self.env.reset)

    for _ in range(20):
        state_vec  = self.env.get_state_vector()
        action_vec = self._random_focused_action()
        predicted  = world_model.network.predict(state_vec, action_vec)
        await asyncio.to_thread(
            self.env.step,
            action_to_surrol(action_vec, self.env.config.dof),
        )
        actual   = self.env.get_state_vector()[0:256].reshape(16, 16)
        mse      = float(((predicted.tissue - actual) ** 2)[rows, cols].mean())
        errors.append(mse)

    return float(np.mean(errors))
```

---

## AGENT FEED BUFFER

Thread-safe streaming buffer for the frontend agent feed. The Claude API
streaming callback runs in the event loop thread; `append()` is called from
there. `flush()` writes to the event log and must not block.

```python
class AgentFeedBuffer:
    """Accumulates streaming tokens and flushes to the event log periodically.

    Attributes:
        agent_id: The agent whose reasoning this buffer tracks.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._tokens:  list[str] = []
        self._loop:    asyncio.AbstractEventLoop = asyncio.get_running_loop()

    def append(self, token: str) -> None:
        """Appends one streamed token. Called from the event loop thread."""
        self._tokens.append(token)

    def flush(self, step: int) -> None:
        """Writes accumulated tokens to the event log as an agent_step event.

        Uses asyncio.create_task() rather than asyncio.get_event_loop()
        (deprecated in Python 3.10+) or call_soon_threadsafe() (needed only
        when calling from a non-event-loop thread). The streaming callback
        always runs in the event loop thread, so create_task() is correct.

        Args:
            step: Current episode step, written to the event payload.
        """
        if not self._tokens:
            return
        reasoning = "".join(self._tokens)
        self._tokens.clear()
        asyncio.create_task(
            _write_agent_step_async(self.agent_id, step, reasoning)
        )
```

---

## CAPABILITY AGENT

**File:** `backend/agents/capability.py`

Does not run an episode loop. Does not call Claude API. Unlocks a DOF,
collects 1,000 focused steps demonstrating the new DOF, fine-tunes.

```python
async def run(self) -> CapabilityOutput:
    """Unlocks a DOF, collects focused training data, and fine-tunes.

    DOF unlock is NOT rolled back on failure. If fine-tuning times out,
    the simulation gains the new DOF but the world model has high error
    on it. The orchestrator detects this and spawns exploration agents
    automatically. The system self-corrects.

    Returns:
        CapabilityOutput with success flag.
    """
    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            # Update DOF atomically on both simulations.
            primary_sim.config.dof    = self.target_dof
            comparison_sim.config.dof = self.target_dof
            world_model.belief.active_dof = self.target_dof

            write_event(
                "capability_unlocked",
                agent_id=self.agent_id,
                new_dof=self.target_dof,
                previous_dof=self.target_dof - 1,
                trigger_error=float(world_model.belief.prediction_error_map.mean()),
                threshold=DOF_THRESHOLDS[self.target_dof - 1],
            )

            # Collect DOF-focused data using a distinct seed for variety.
            # seed = 42 + dof gives 6 deterministic but distinct anatomies.
            dof_env = await create_env(SimConfig(
                seed=42 + self.target_dof,
                dof=self.target_dof,
                n_vessels=primary_sim.config.n_vessels,
            ))
            try:
                samples = await self._collect_dof_samples(dof_env, n=1000)
            finally:
                await asyncio.to_thread(dof_env.close)

            # More epochs than exploration fine-tune: new DOF is completely
            # unseen by the network and requires a larger update.
            loss = await world_model.fine_tune(samples, epochs=10)
            wandb.log({
                "active_dof":   self.target_dof,
                "unlock_error": float(world_model.belief.prediction_error_map.mean()),
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
```

---

## REACTIVE AGENT

**File:** `backend/agents/reactive.py`

Greedy policy. No world model. No Claude API. Structurally isolated — zero
imports from `world_model`, `anthropic`, or `orchestrator`. This is enforced
by a grep test. Any import from those modules would undermine the comparison
baseline, making SOMA's advantage appear smaller than it is.

```python
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
            state_vec    = self._env.get_state_vector()
            action_vec   = self._select_action(state_vec)
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
                task_failed=bool(self._env._task_failed),
                active_dof=self._env.config.dof,
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
        ee    = state_vec[768:770]   # normalized EE x, y
        tgt   = state_vec[776:778]   # normalized target col, row
        bleed = state_vec[512:768].sum()

        vec = np.zeros(12, dtype=np.float32)

        if bleed > 0:
            # Cauterize at current position when bleeding is active.
            vec[9]  = 1.0  # CAUTERIZE mode
            vec[11] = 1.0
        else:
            direction = tgt - ee
            dist      = float(np.linalg.norm(direction)) + 1e-8
            unit      = direction / dist
            speed     = 0.1 if dist > 0.1 else 0.05  # slow near target
            vec[0:2]  = unit * speed
            vec[7]    = 1.0  # MOVE mode
            vec[11]   = 0.7 if dist > 0.1 else 0.4

        return vec
```

---

## PRODUCTION PATTERNS

### Circuit Breaker

Prevents the orchestrator from burning API credits on a region that the
world model consistently cannot improve (e.g., grid boundary physics anomaly).

```python
# In collect_status_node (already shown above).
# Values match QUICK_REFERENCE.md:
CIRCUIT_BREAKER_THRESHOLD = 3    # consecutive failures to open the breaker
CIRCUIT_BREAKER_BACKOFF_S = 300  # seconds before the region is retried (5 min)
```

On 3 consecutive failures for the same region: that region is excluded
from spawn candidates for 5 minutes. On any success: failure count resets.

### Agent Timeout

Hard 10-minute limit on every agent via `asyncio.timeout()`. Without it,
a stalled Claude API call or SurRoL deadlock holds a simulation instance
permanently. With `MAX_CONCURRENT_AGENTS = 4`, one zombie blocks 25%
of agent capacity.

```python
AGENT_TIMEOUT_SECONDS = 600  # 10 minutes
# Applied in ExplorationAgent.run() and CapabilityAgent.run() via asyncio.timeout()
```

### Fine-Tune Backpressure Queue

When two agents complete simultaneously, both call `world_model.fine_tune()`.
The second waits on `_finetune_lock` (up to 5 minutes). Queue samples
before the wait so they are not lost on process restart.

```python
# In world_model.py — module-level queue, drained by a background worker.
_fine_tune_queue: asyncio.Queue[list[tuple]] = asyncio.Queue(maxsize=10)

async def fine_tune_worker() -> None:
    """Drains the fine-tune queue sequentially. Started in main.py lifespan.

    # LOOP: fine_tune_drain
    # Pre-condition:  _fine_tune_queue initialized; world_model singleton exists.
    # Invariant:      fine-tune calls are serialized; no concurrent weight updates.
    # Termination:    cancelled via task.cancel() on shutdown.
    # Yield:          asyncio.Queue.get() suspends until samples are available.
    """
    while True:
        samples = await _fine_tune_queue.get()
        try:
            await world_model.fine_tune(samples)
        except Exception as exc:
            logging.error("fine_tune_worker: fine-tune failed: %s", exc)
            write_event("agent_failed", error=f"fine_tune_error: {exc}")
        finally:
            _fine_tune_queue.task_done()
```

**Trade-off:** `error_after` in `agent_completed` is computed before the
queue worker runs. The heatmap animation fires on `agent_completed` but
the actual error map update happens slightly later. For demo timing, the
visual precedes the metric by one orchestrator cycle. Acceptable — the
animation is what matters. Document in MASTER_STATE.md under KNOWN DEVIATIONS.

### API Cost Tracking

```python
# In write_event("agent_completed", ...):
api_tokens_used=tokens,
estimated_cost_usd=tokens * 3e-6,  # claude-sonnet-4-6 rate

# In W&B:
wandb.log({
    "cumulative_api_tokens":   _api_token_count,
    "cumulative_api_cost_usd": _api_token_count * 3e-6,
})
```

Budget estimate: 4 agents × 100 steps × (1 call / 5 steps) × 150 tokens ≈
12,000 tokens ≈ $0.036 per demo. Negligible.

### Graceful Degradation on API Failure

```python
# In run_episode(), when AgentAPIError is caught:
self._buffer.append(f"[step {step}] API fallback: {exc}")
last_action = self._random_focused_action()  # continue with random actions
```

Episode continues. Training data is collected. `fine_tune()` runs. Error may
decrease less (random actions cover the region less efficiently than
Claude's selections), but the self-improvement loop remains functional.
The agent feed shows `[API fallback]` instead of reasoning — judges may
notice; use the BUILD_PLAN.md Scope Cut 1 framing.

---

## AGENT LIFECYCLE INVARIANTS

Referenced by name in tests and MASTER_STATE.md entries.

### AGENT_SPAWNED_BEFORE_LOOP INVARIANT

`agent_spawned` must be written by `spawn_exploration_node` before `Send`
issues the agent's task — never inside the agent's own `run()` method.

Why: the frontend canvas creates nodes from `agent_spawned` events. If
the event arrives after the agent has been running, the node appears
mid-execution or post-completion. Judges see branches that materialize
from nowhere.

**Test:** `test:agents:spawned_before_loop`
```python
events = get_events_for_agent(agent_id)
spawned    = next(e for e in events if e["event_type"] == "agent_spawned")
first_step = next(e for e in events if e["event_type"] == "agent_step")
assert spawned["timestamp"] < first_step["timestamp"]
```

### AGENT_COMPLETED_AFTER_FINETUNE INVARIANT

`agent_completed` must be written after `world_model.fine_tune()` returns
(or after `_fine_tune_queue.put()` if using the backpressure pattern).

Why: `error_after` in `agent_completed` drives the frontend's error
reduction animation. If written before fine-tuning, `error_after` equals
`error_before` and the animation condition `error_after < error_before`
is never true. The demo's primary visual moment breaks.

**Test:** `test:agents:completed_after_finetune`
```python
events   = get_events_since(0)
completed = next(e for e in events if e["event_type"] == "agent_completed"
                 and e["agent_id"] == agent_id)
wm_updates = [
    e for e in events
    if e["event_type"] == "world_model_updated"
    and e["timestamp"] < completed["timestamp"]
]
assert len(wm_updates) >= 1, "world_model_updated must precede agent_completed"
```

### SIMULATION_RELEASED_ON_FAILURE INVARIANT

Every code path in an agent's `run()` must call `env.close()` via `finally`.

Why: each leaked SurRoL instance holds a PyBullet physics server (~200 MB).
With `MAX_CONCURRENT_AGENTS = 4` and 2 leaks per run, the process exhausts
memory within an hour. The symptom (gradual slowdown, no explicit error)
is difficult to attribute to a leak without measuring RAM.

**Test:** `test:agents:sim_released`
```bash
grep -n "env.close\|finally" backend/agents/exploration.py
# Must show finally: ... env.close() pattern in run()
```

### REACTIVE_AGENT_ISOLATION INVARIANT

`backend/agents/reactive.py` must have zero imports from `world_model`,
`prediction_net`, `belief_state`, `orchestrator`, or `anthropic`.

Why: the comparison baseline is only valid if it has no access to the
world model. Any import creates a dependency that could be accidentally
used. The grep test catches this before it causes a silent correctness bug.

**Test:** `test:agents:reactive_isolation`
```bash
grep -E "from backend\.(world_model|prediction_net|belief_state|orchestrator)" \
    backend/agents/reactive.py
# Must return empty.
```

---

## CONSTANTS

All values also appear in QUICK_REFERENCE.md. Change one → update both.

```python
# Orchestrator thresholds
EXPLORE_THRESHOLD      = 0.15   # regional error that triggers agent spawn
MAX_CONCURRENT_AGENTS  = 4      # exploration agents max (primary+comparison+4=6≤8)
ORCHESTRATOR_SLEEP_S   = 2.0    # seconds between decision cycles
DOF_THRESHOLDS         = {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.04, 5: 0.03}

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD = 3    # consecutive failures to open breaker
CIRCUIT_BREAKER_BACKOFF_S = 300  # seconds before retry (5 min)

# Exploration agents
EXPLORE_N_STEPS        = 100    # steps per exploration episode
CLAUDE_CALL_EVERY_N    = 5      # steps between Claude API calls
AGENT_TIMEOUT_SECONDS  = 600    # 10-minute hard kill

# Claude API
CLAUDE_MODEL           = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS      = 300
CLAUDE_MAX_RETRIES     = 3
API_SEMAPHORE_LIMIT    = 10     # max concurrent calls across all agents
```

---

## PITFALLS — SYMPTOM FIRST

---

**SYMPTOM: Canvas shows no agent nodes even though logs show agents
completing. `get_events_since(0)` returns `agent_completed` events
but no `agent_spawned` events for those agents.**

CAUSE: `agent_spawned` written inside the agent's `run()` method rather
than in `spawn_exploration_node` before the `Send`.
CONFIRM: `SELECT event_type, timestamp FROM events WHERE agent_id='exp_xxx' ORDER BY timestamp` —
if `agent_spawned` is absent or has a later timestamp than `agent_step`: violated.
FIX: Move `write_event("agent_spawned", ...)` to `spawn_exploration_node`,
immediately before `sends.append(Send(...))`. The agent's `run()` writes
nothing for `agent_spawned`.
Test: `test:agents:spawned_before_loop`

---

**SYMPTOM: Heatmap shows no animation on `agent_completed`. Events exist
with `error_before` and `error_after` present, but `error_before == error_after`
in all of them.**

CAUSE: `_measure_region_error()` called before `fine_tune()` returned.
The error map hasn't updated, so pre and post measurements are identical.
CONFIRM: Compare timestamps of `world_model_updated` and `agent_completed`
for the same agent. If `agent_completed` is earlier: fine-tune ran after.
FIX: Call `_measure_region_error()` after `await world_model.fine_tune(samples)`
returns. If using the backpressure queue, measure error inside
`fine_tune_worker()` after the fine-tune completes, then write `agent_completed`.
Test: `test:agents:completed_after_finetune`

---

**SYMPTOM: `StopIteration` in `_call_claude`. Agents log
"Claude responded without calling select_action."**

CAUSE: `tool_choice` is `"auto"` or absent. Claude decides to respond in
prose rather than calling the tool.
FIX: Set `tool_choice={"type": "any"}` in every API call. This forces a
tool call regardless of Claude's preference. With one tool defined, it
always calls `select_action`.

---

**SYMPTOM: Backend slows to a crawl after 30–60 minutes. RAM grows.
PyBullet logs "physics server warning".**

CAUSE: SIMULATION_RELEASED_ON_FAILURE INVARIANT violated. Failed agents
did not close their SurRoL instances.
CONFIRM: `python -c "import pybullet as p; print(p.getNumBodies())"` —
if count grows over time: leaked instances.
FIX: Ensure `finally: await asyncio.to_thread(self.env.close)` in every
agent `run()` method, including `CapabilityAgent`'s temp DOF environment.
Test: `test:agents:sim_released`

---

**SYMPTOM: `GraphRecursionError` after 25+ cycles. Orchestrator logs
"maximum recursion" error.**

CAUSE: LangGraph's default `recursion_limit` is 25 node invocations per
`invoke()` call. With 5 nodes and 4 parallel agents, 25 is reachable
legitimately. This is not an infinite loop.
FIX: Set `recursion_limit=100` in graph config:
```python
graph.invoke(initial_state, config={"recursion_limit": 100})
```
If graph still loops without terminating after 100 cycles: check that
`collect_status_node` properly drains `completed_agents` from `active_agents`.

---

**SYMPTOM: `RuntimeError: no running event loop` in `AgentFeedBuffer.flush()`.**

CAUSE: `asyncio.get_event_loop()` called from context where no loop is
running (deprecated in Python 3.10+, removed in 3.12+).
FIX: Use `asyncio.create_task()` directly. The streaming callback always
runs in the event loop thread, so `create_task` is correct. Use
`loop.call_soon_threadsafe()` only if calling from an `asyncio.to_thread()`
worker.

---

## TESTING PROTOCOL

Run only these after Tasks 1.6–1.7. Full suite at sprint checkpoint only.

```bash
# Task 1.6 — orchestrator
pytest tests/test_orchestrator.py -v
pytest tests/test_orchestrator.py::test_graph_compiles -v
pytest tests/test_orchestrator.py::test_route_explore -v
pytest tests/test_orchestrator.py::test_route_unlock -v
pytest tests/test_orchestrator.py::test_circuit_breaker -v
pytest tests/test_orchestrator.py::test_spawn_writes_event_first -v

# Task 1.7 — agents
pytest tests/test_agents.py -v
pytest tests/test_agents.py::test_spawned_before_loop -v       # MUST PASS
pytest tests/test_agents.py::test_completed_after_finetune -v  # MUST PASS
pytest tests/test_agents.py::test_sim_released -v              # MUST PASS
pytest tests/test_agents.py::test_reactive_isolation -v        # MUST PASS
pytest tests/test_agents.py::test_api_fallback_on_error -v
pytest tests/test_agents.py::test_agent_timeout -v
```

### Minimum Passing Set for Task Completion

**Task 1.6:**
```python
# test:orchestrator:graph_compiles
graph = build_soma_graph(mock_world_model, mock_primary, mock_comparison)
assert graph is not None

# test:orchestrator:route_valid_strings
for state in [low_error_state, high_error_state, mixed_state]:
    result = route_orchestrator(state)
    assert result in ("spawn_explore", "unlock_dof", "wait")
```

**Task 1.7:**
```bash
# test:agents:reactive_isolation — fastest check, runs in 1 second
grep -E "from backend\.(world_model|prediction_net|belief_state|orchestrator)" \
    backend/agents/reactive.py
# Must return empty.
```

---

## VERIFICATION CHECKLIST

```
ORCHESTRATOR (Task 1.6):
  □ graph.compile() succeeds with MemorySaver
  □ Orchestrator cycles every 2 s (check log timestamps)
  □ agent_spawned written before Send() issued
  □ route_orchestrator() returns "unlock_dof" at correct error thresholds
  □ MAX_CONCURRENT_AGENTS = 4 enforced; 5th agent not spawned
  □ Circuit breaker: region excluded after 3 failures; retry after 5 min

EXPLORATION AGENT (Task 1.7):
  □ agent_spawned event exists before first agent_step event
  □ Claude API returns structured tool call (not prose)
  □ Streaming: agent feed updates token-by-token
  □ agent_completed written after world_model_updated
  □ error_after < error_before by ≥ 0.01 on successful run
  □ env.close() called on success path AND failure/timeout path
  □ API fallback: random actions used when Claude API fails

CAPABILITY AGENT (Task 1.7):
  □ capability_unlocked event written before agent_completed
  □ primary_sim.config.dof updated to target_dof
  □ comparison_sim.config.dof updated to same target_dof
  □ DOF env uses seed = 42 + target_dof
  □ fine_tune called with epochs=10 (not default 5)

REACTIVE AGENT (Task 1.7):
  □ Zero imports from world_model, anthropic, orchestrator (grep test)
  □ Writes simulation_step events with sim_id="comparison"
  □ Resets episode on task_complete or task_failed
  □ await asyncio.sleep(0.05) present — loop invariant comment present
```
