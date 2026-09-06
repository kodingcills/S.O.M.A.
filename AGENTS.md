# AGENTS.md
# SOMA — Agent Specification
# Authoritative for: LangGraph graph definition, agent contracts,
# orchestrator decision logic, ExplorationAgent (Claude API + tool use),
# CapabilityAgent, ReactiveAgent, agent lifecycle invariants,
# production engineering (circuit breaker, timeout, rate limiting,
# fine-tune queue, cost tracking, streaming, backpressure).
#
# Invariants referenced here are DEFINED in ARCHITECTURE.md.
# Read this before building: orchestrator.py, agents/exploration.py,
# agents/capability.py, agents/reactive.py.

---

## TWO AGENT SYSTEMS — DO NOT CONFUSE THEM

SOMA involves two completely separate agent systems at different layers.
Mixing them up is the most likely source of architectural confusion.

```
LAYER 1 — BUILD AGENTS (Oh My OpenCode / OpenCode)
  What: OpenCode sessions that WRITE SOMA's source code
  Where: developer's terminal, managed by Oh My OpenCode
  Reads: /AGENTS.md (project root), CLAUDE.md, MASTER_STATE.md
  Coordinates via: MASTER_STATE.md checkpoint protocol
  Agents: Sisyphus, Prometheus, Oracle, Librarian (Oh My OpenCode)
  Scope: build time only — these agents do not exist at runtime

LAYER 2 — RUNTIME AGENTS (LangGraph + Claude API)
  What: Python asyncio tasks that EXPLORE SOMA's simulation
  Where: backend/agents/, runs inside SOMA during demo
  Reads: WorldModel belief state, SurRoL environment
  Coordinates via: SQLite event log (write_event / get_events_since)
  Agents: ExplorationAgent, CapabilityAgent, ReactiveAgent
  Scope: runtime only — these agents do not know about OpenCode
```

**This file (docs/specs/AGENTS.md) specifies Layer 2 exclusively.**

For Layer 1, see: `/AGENTS.md` (project root — Oh My OpenCode instructions).
The two files must never be merged. Oh My OpenCode auto-injects the
root AGENTS.md into every build session; it must contain build instructions,
not simulation physics.

---

## OH MY OPENCODE BUILD NOTES

Oh My OpenCode provides specialist agents for the build. Map tasks to agents:

| Task | Oh My OpenCode Agent | Why |
|---|---|---|
| 1.1 Event log + API | Sisyphus | Straightforward implementation task |
| 1.2 SurRoL environment | Sisyphus + Librarian | Librarian pulls SurRoL live docs via Context7 |
| 1.3 BeliefState | Sisyphus | Clear spec, no novel architecture |
| 1.4 PredictionNetwork | Sisyphus | PyTorch boilerplate |
| 1.5 WorldModel | **Oracle** | Singleton pattern, lock model, async complexity |
| 1.6 LangGraph Orchestrator | **Oracle** | Graph architecture, Command routing |
| 1.7 Exploration + Capability agents | **Oracle** | Claude API tool use, streaming |
| 1.8 MPC agent | Sisyphus | Math-heavy but well-specified |
| 1.9 main.py integration | Sisyphus | Wiring, not architecture |
| Any integration failure | **Oracle** | Architecture/debugging specialist |
| SurRoL/LangGraph API questions | **Librarian** | Context7 fetches live docs |
| AST-level refactor | Explore | Fast grep across codebase |

**Prometheus (planner) use:** run Prometheus before Task 1.6 to plan the
LangGraph graph topology. Prometheus produces a structured plan; Sisyphus
or Oracle executes it. Don't skip planning for graph-topology decisions —
changing the graph structure after agents are implemented requires
rebuilding the entire orchestrator.

**AGENTS.md auto-injection:** Oh My OpenCode injects `/AGENTS.md` into
every session. The root `/AGENTS.md` must contain:
- The Oh My OpenCode agent roster and delegation rules
- Task-to-agent mapping (table above)
- MASTER_STATE.md checkpoint protocol
- NOT: simulation physics or LangGraph implementation details

---

## LANGGRAPH GRAPH DEFINITION

**File:** `backend/orchestrator.py`

### Dependencies

```bash
uv add langgraph langgraph-checkpoint
# LangSmith (optional, observability):
uv add langsmith
```

### Graph State

State is the single source of truth for the entire orchestration graph.
Keep it small — only what nodes need to make routing decisions.
List fields that multiple nodes write to use `Annotated[list, operator.add]`
to avoid write conflicts during parallel fan-out.

```python
from typing import TypedDict, Annotated
import operator

class SomaOrchestratorState(TypedDict):
    # Read-only inputs to each cycle (set by orchestrator_node)
    regional_errors:    dict[str, float]   # 6 regions → error values
    global_mean_error:  float
    active_dof:         int
    active_agents:      list[str]          # agent_ids currently running
    cycle_count:        int

    # Written by fan-out agents (reducer: list append)
    completed_agents: Annotated[list[str], operator.add]
    failed_agents:    Annotated[list[str], operator.add]

    # Written by capability check
    unlock_requested: bool
    target_dof:       int

    # Circuit breaker state (persisted across cycles via checkpointer)
    region_failure_counts:    dict[str, int]    # failures per region
    region_backoff_until:     dict[str, float]  # timestamp per region
    consecutive_fine_tune_failures: int
```

Why `TypedDict` not `Pydantic`: LangGraph's checkpointing serializes state
to JSON. TypedDict serializes cleanly; Pydantic models require custom
serializers for numpy arrays and other non-JSON types. Keep state JSON-clean:
no numpy, no dataclasses — only Python primitives.

### Graph Construction

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def build_soma_graph(world_model, sim_manager) -> CompiledGraph:
    builder = StateGraph(SomaOrchestratorState)

    # Nodes
    builder.add_node("orchestrator",    orchestrator_node)
    builder.add_node("spawn_explore",   spawn_exploration_node)
    builder.add_node("unlock_dof",      unlock_dof_node)
    builder.add_node("collect_status",  collect_status_node)

    # Edges
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "spawn_explore": "spawn_explore",
            "unlock_dof":    "unlock_dof",
            "wait":          "collect_status",
        }
    )
    builder.add_edge("spawn_explore",  "collect_status")
    builder.add_edge("unlock_dof",     "collect_status")
    builder.add_edge("collect_status", "orchestrator")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # Inject dependencies via closure (nodes are functions, not methods)
    # — see node implementations below
    return graph
```

**`MemorySaver` vs `PostgresSaver`:** MemorySaver stores state in-process.
If the backend crashes and restarts, orchestrator state is lost and the
graph restarts from cycle 0. For a hackathon demo, this is acceptable —
the world model weights and event log persist to disk; only the
orchestrator's in-flight cycle state is lost. PostgresSaver would require
running a PostgreSQL instance, which is not worth the operational cost.

### Route Function

```python
def route_orchestrator(state: SomaOrchestratorState) -> str:
    now = time.time()

    # Priority 1: DOF unlock (rarer, higher value)
    thresholds = {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.04, 5: 0.03}
    unlock_thresh = thresholds.get(state["active_dof"], 1.0)
    if (state["global_mean_error"] < unlock_thresh
            and state["active_dof"] < 6
            and "cap" not in "".join(state["active_agents"])):
        return "unlock_dof"

    # Priority 2: exploration (check all regions, pick highest error)
    spawn_targets = []
    for region, error in state["regional_errors"].items():
        if error < EXPLORE_THRESHOLD:
            continue
        if region in "".join(state["active_agents"]):  # already active
            continue
        backoff = state["region_backoff_until"].get(region, 0)
        if now < backoff:
            continue
        failure_count = state["region_failure_counts"].get(region, 0)
        if failure_count >= CIRCUIT_BREAKER_THRESHOLD:  # 3 failures
            continue
        spawn_targets.append((error, region))

    # Can we spawn more agents?
    n_active = len(state["active_agents"])
    if spawn_targets and n_active < MAX_CONCURRENT_AGENTS:
        return "spawn_explore"

    return "wait"
```

### Orchestrator Node

```python
async def orchestrator_node(
    state: SomaOrchestratorState,
) -> SomaOrchestratorState:
    # Direct call to WorldModel — Architecture Law Exception #2
    regional = world_model.belief.get_regional_errors()
    global_e = float(world_model.belief.prediction_error_map.mean())

    # Clean up completed/failed agents from tracking
    still_active = [
        aid for aid in state["active_agents"]
        if aid not in state["completed_agents"]
        and aid not in state["failed_agents"]
    ]

    return {
        **state,
        "regional_errors":   regional,
        "global_mean_error": global_e,
        "active_dof":        world_model.belief.active_dof,
        "active_agents":     still_active,
        "completed_agents":  [],   # reset reducers for next cycle
        "failed_agents":     [],
        "cycle_count":       state["cycle_count"] + 1,
    }
```

### Spawn Exploration Node (Fan-Out via Send)

```python
from langgraph.types import Send

async def spawn_exploration_node(
    state: SomaOrchestratorState,
) -> list[Send]:
    """
    Returns Send objects to spawn ExplorationAgent tasks in parallel.
    LangGraph executes all Sends concurrently — true fan-out.
    Each Send targets 'exploration_agent_runner' sub-node.
    """
    now = time.time()
    sends = []

    # Find top candidate regions (up to remaining agent capacity)
    candidates = sorted(
        [(e, r) for r, e in state["regional_errors"].items()
         if e >= EXPLORE_THRESHOLD
         and r not in "".join(state["active_agents"])
         and now >= state["region_backoff_until"].get(r, 0)
         and state["region_failure_counts"].get(r, 0) < CIRCUIT_BREAKER_THRESHOLD
        ],
        reverse=True
    )

    slots = MAX_CONCURRENT_AGENTS - len(state["active_agents"])
    for error, region in candidates[:slots]:
        agent_id = f"exp_{uuid4().hex[:6]}"
        # Write agent_spawned BEFORE Send — canvas node must appear first
        write_event("agent_spawned",
                    agent_id=agent_id, parent_id="orchestrator",
                    region=region, error_before=error,
                    planned_steps=EXPLORE_N_STEPS)
        sends.append(Send("exploration_agent_runner", {
            "agent_id": agent_id,
            "region":   region,
            "error_before": error,
        }))

    return sends if sends else [Send("collect_status", state)]
```

Why `Send` not `asyncio.create_task`: LangGraph's `Send` is graph-native.
It creates a parallel branch in the graph that the checkpointer tracks.
`asyncio.create_task` creates a task outside the graph — the checkpointer
doesn't know it exists. If the process restarts, `create_task` tasks are
lost with no record. `Send` tasks are journaled.

---

## AGENT CONTRACTS

Every agent has explicit input/output TypedDicts. No untyped `**kwargs`.
Agents that don't follow their contract produce events with wrong fields,
which the frontend silently ignores — the canvas shows missing nodes with
no error.

```python
class ExplorationInput(TypedDict):
    agent_id:     str         # format: 'exp_{6 hex chars}'
    region:       str         # one of the 6 named regions
    error_before: float       # regional error at spawn time

class ExplorationOutput(TypedDict):
    agent_id:         str
    region:           str
    error_before:     float
    error_after:      float
    samples_collected: int
    fine_tune_loss:   float
    api_tokens_used:  int     # total Claude API tokens across episode
    success:          bool    # error_after < error_before - 0.05

class CapabilityInput(TypedDict):
    agent_id:    str         # format: 'cap_{new_dof}'
    target_dof:  int

class CapabilityOutput(TypedDict):
    agent_id:    str
    target_dof:  int
    samples_collected: int
    fine_tune_loss:    float
    success:     bool
```

---

## EXPLORATION AGENT

**File:** `backend/agents/exploration.py`

The system's primary self-improvement mechanism. Collects focused data
in a high-error region, fine-tunes the world model, and reports the
error delta. Uses Claude API for action selection — this is what populates
the agent feed visible to judges.

### Event Sequence (mandatory — violating order breaks canvas)

```python
# ① agent_spawned — written by orchestrator BEFORE spawning this task
#    (already written in spawn_exploration_node above)

# ② During episode, every 10 steps:
write_event("agent_step", agent_id=agent_id, step=step_count)

# ③ After fine_tune() completes — NOT before:
write_event("agent_completed",
            agent_id=agent_id,
            region=region,
            error_before=error_before,
            error_after=final_error,
            samples_collected=len(samples),
            fine_tune_loss=fine_tune_loss,
            api_tokens_used=total_tokens)
```

Writing `agent_completed` before fine-tuning: the error_after field
will be the pre-fine-tune error, making the reduction appear zero.
The heatmap animation won't fire. Writing it after ensures `error_after`
reflects the actual improvement.

### Dedicated Simulation

Each exploration agent constructs its own SurRoL instance.
Uses seed=42 — same anatomy as primary. See SEED REPRODUCIBILITY
INVARIANT in ARCHITECTURE.md and the detailed reasoning in
SIMULATION.md Instance table.

```python
env = await create_env(SimConfig(
    seed=42,
    dof=world_model.belief.active_dof,  # match current DOF
    n_vessels=primary_sim.config.n_vessels,
    difficulty=primary_sim.config.difficulty,
))
```

### Claude API Tool Use

Structured tool use guarantees parseable output. Free-text JSON parsing
fails on Claude's explanatory preambles and markdown formatting.

```python
TOOLS = [
    {
        "name": "select_action",
        "description": (
            "Select the next action for data collection in the target region. "
            "Prioritize actions that generate informative tissue dynamics "
            "data in the specified region."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["MOVE", "CAUTERIZE", "WAIT"],
                    "description": "Action to execute this step"
                },
                "delta_x": {"type": "number", "description": "EE x delta [-1,1]"},
                "delta_y": {"type": "number", "description": "EE y delta [-1,1]"},
                "delta_z": {"type": "number", "description": "EE z delta [-1,1]"},
                "magnitude": {"type": "number", "description": "Action scale [0,1]"},
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining this action choice"
                }
            },
            "required": ["action_type", "reasoning"]
        }
    }
]
```

### Rate Limiter

Claude API tier limits (as of 2026): ~50 req/sec for claude-sonnet-4-6.
With 4 concurrent exploration agents each calling every 5 steps at
~10 steps/sec, peak rate is 4 × 10/5 = 8 req/sec. Well within limits.
However: fine-tuning pauses steps, startup burst on agent creation, and
retries can spike this. Use a semaphore as a soft rate limit.

```python
# Module-level — shared across all exploration agent instances
_claude_api_semaphore = asyncio.Semaphore(10)  # max 10 concurrent API calls
_api_call_count = 0
_api_token_count = 0
```

### API Call with Retry and Streaming

```python
async def _call_claude_with_retry(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    agent_feed_callback: Callable[[str], None],
    max_retries: int = 3,
) -> tuple[dict, int]:
    """
    Returns: (parsed_tool_input, tokens_used)
    Raises: AgentAPIError after max_retries exhausted
    """
    for attempt in range(max_retries):
        try:
            async with _claude_api_semaphore:
                # Stream response for real-time agent feed
                async with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=300,
                    tools=TOOLS,
                    tool_choice={"type": "any"},  # force tool use
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    # Stream reasoning tokens to agent feed in real time
                    async for text in stream.text_stream:
                        agent_feed_callback(text)

                    final = await stream.get_final_message()

            # Extract tool use block
            tool_block = next(
                b for b in final.content
                if b.type == "tool_use" and b.name == "select_action"
            )
            tokens = final.usage.input_tokens + final.usage.output_tokens
            return tool_block.input, tokens

        except anthropic.RateLimitError:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            await asyncio.sleep(wait)
        except anthropic.APIConnectionError:
            if attempt == max_retries - 1:
                raise AgentAPIError("Claude API unreachable after retries")
            await asyncio.sleep(1)
        except StopIteration:
            raise AgentAPIError("Claude did not call the select_action tool")

    raise AgentAPIError(f"API call failed after {max_retries} retries")
```

Why `tool_choice={"type": "any"}`: forces Claude to use a tool rather
than responding with text. Without this, Claude occasionally says "I would
choose action X" in prose instead of calling the tool, producing a
`StopIteration` on the tool block extraction. `"any"` means "use any
available tool" — with one tool defined, it always calls `select_action`.

Why streaming: the agent feed in the frontend shows reasoning token-by-token
as the agent thinks. Without streaming, the feed shows nothing until the
complete response arrives (~500ms delay). With streaming, judges see
the reasoning appear live during the pitch.

### Episode Loop

```python
async def run_episode(
    self,
    n_steps: int = EXPLORE_N_STEPS,  # 100
) -> tuple[list[tuple], int]:
    """Returns: (samples, total_api_tokens)"""
    samples = []
    total_tokens = 0
    last_action_vec = np.zeros(12, dtype=np.float32)
    last_action_vec[7] = 1.0  # default: MOVE

    obs = await asyncio.to_thread(self.env.reset)

    for step in range(n_steps):
        state_vec = self.env.get_state_vector()

        # Call Claude API every 5 steps; repeat last action in between
        if step % 5 == 0:
            prompt = self._build_prompt(state_vec, step)
            try:
                tool_input, tokens = await _call_claude_with_retry(
                    self._client, prompt,
                    lambda text: self._feed_callback(step, text)
                )
                last_action_vec = self._tool_input_to_action(tool_input)
                total_tokens += tokens
            except AgentAPIError as e:
                # Degraded mode: use last action or random fallback
                self._feed_callback(step, f"[API fallback at step {step}: {e}]")
                last_action_vec = self._random_focused_action()

        action_vec = last_action_vec
        surrol_action = action_to_surrol(action_vec, self.env.config.dof)

        next_obs, reward, done, _ = await asyncio.to_thread(
            self.env.step, surrol_action
        )
        next_state_vec = self.env.get_state_vector()
        samples.append((state_vec, action_vec, next_state_vec))

        if step % 10 == 0:
            write_event("agent_step", agent_id=self.agent_id, step=step)

        if done:
            await asyncio.to_thread(self.env.reset)

    return samples, total_tokens
```

### Prompt Construction

```python
def _build_prompt(self, state_vec: np.ndarray, step: int) -> str:
    # Extract readable values from state vector
    tissue_mean = float(state_vec[0:256].mean())
    bleeding_count = int(state_vec[512:768].sum())
    ee_pos = state_vec[768:771]
    dof = int(np.argmax(state_vec[799:805])) + 1

    return f"""You are an exploration agent running experiments in a surgical
simulation. Your goal is to collect diverse, informative tissue dynamics
data in the '{self.region}' region to improve prediction accuracy there.

Current state:
- Target region: {self.region}
- Initial prediction error: {self.error_before:.3f}
- Step: {step}/{EXPLORE_N_STEPS}
- Tissue mean integrity: {tissue_mean:.3f}
- Active bleeding cells: {bleeding_count}
- EE position (normalized): {ee_pos.tolist()}
- Active DOF: {dof}

Strategy: focus your actions in {self.region}. Vary action types to
explore different tissue dynamics. Prioritize CUT actions near vessel
areas (high vascularity) and CAUTERIZE when bleeding is active.

Call select_action with your chosen action and a brief reasoning."""
```

### Full Agent Run

```python
async def run(self) -> ExplorationOutput:
    timeout = AGENT_TIMEOUT_SECONDS  # 600 — 10 minutes hard limit

    try:
        async with asyncio.timeout(timeout):
            # Phase 1: collect data
            samples, tokens = await self.run_episode()

            # Phase 2: fine-tune (may be queued if WorldModel busy)
            fine_tune_loss = await world_model.fine_tune(samples)

            # Phase 3: measure actual error improvement
            final_error = await self._measure_region_error()

        write_event("agent_completed",
                    agent_id=self.agent_id, region=self.region,
                    error_before=self.error_before, error_after=final_error,
                    samples_collected=len(samples),
                    fine_tune_loss=fine_tune_loss,
                    api_tokens_used=tokens)

        return ExplorationOutput(
            agent_id=self.agent_id, region=self.region,
            error_before=self.error_before, error_after=final_error,
            samples_collected=len(samples), fine_tune_loss=fine_tune_loss,
            api_tokens_used=tokens,
            success=(final_error < self.error_before - 0.05)
        )

    except TimeoutError:
        write_event("agent_failed", agent_id=self.agent_id,
                    region=self.region, error="timeout")
        return ExplorationOutput(..., success=False)

    except Exception as e:
        write_event("agent_failed", agent_id=self.agent_id,
                    region=self.region, error=str(e))
        raise
    finally:
        await asyncio.to_thread(self.env.close)  # always release sim
```

### `_measure_region_error()`

Run 20 predict-then-observe cycles in the dedicated sim and compute
mean squared error for the target region's tissue integrity cells.
This is the authoritative `error_after` value written to the event log.

```python
async def _measure_region_error(self) -> float:
    region_rows, region_cols = REGION_SLICE_MAP[self.region]
    errors = []
    obs = await asyncio.to_thread(self.env.reset)

    for _ in range(20):
        state_vec = self.env.get_state_vector()
        action_vec = self._random_focused_action()
        predicted = world_model.network.predict(state_vec, action_vec)
        await asyncio.to_thread(
            self.env.step, action_to_surrol(action_vec, self.env.config.dof)
        )
        actual_integrity = self.env.get_state_vector()[0:256].reshape(16, 16)
        pred_integrity = predicted.tissue
        mse = float(((pred_integrity - actual_integrity) ** 2)
                    [region_rows, region_cols].mean())
        errors.append(mse)

    return float(np.mean(errors))
```

---

## CAPABILITY AGENT

**File:** `backend/agents/capability.py`

Does not run an episode loop. Does not call Claude API. Its job is to
unlock a DOF, collect focused training data demonstrating the new DOF,
and fine-tune.

```python
async def run(self) -> CapabilityOutput:
    # Write agent_spawned already done by orchestrator
    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            # 1. Update DOF on both primary and comparison sims atomically
            primary_sim.config.dof = self.target_dof
            comparison_sim.config.dof = self.target_dof
            world_model.belief.active_dof = self.target_dof

            write_event("capability_unlocked",
                        agent_id=self.agent_id,
                        new_dof=self.target_dof,
                        previous_dof=self.target_dof - 1,
                        trigger_error=world_model.belief.prediction_error_map.mean(),
                        threshold=DOF_THRESHOLDS[self.target_dof - 1])

            # 2. Collect DOF-focused data
            dof_env = await create_env(SimConfig(
                seed=42 + self.target_dof,  # see SIMULATION.md instances table
                dof=self.target_dof,
                n_vessels=primary_sim.config.n_vessels,
            ))
            samples = await self._collect_dof_samples(dof_env, n=1000)
            await asyncio.to_thread(dof_env.close)

            # 3. Fine-tune with more epochs (new DOF = larger adaptation)
            loss = await world_model.fine_tune(samples, epochs=10)

            wandb.log({"active_dof": self.target_dof,
                       "unlock_error": world_model.belief.prediction_error_map.mean()})

        write_event("agent_completed", agent_id=self.agent_id,
                    samples_collected=len(samples), fine_tune_loss=loss)

        return CapabilityOutput(agent_id=self.agent_id,
                                target_dof=self.target_dof,
                                samples_collected=len(samples),
                                fine_tune_loss=loss, success=True)

    except TimeoutError:
        write_event("agent_failed", agent_id=self.agent_id, error="timeout")
        # ROLLBACK: if fine-tuning didn't complete, DOF was unlocked but
        # world model doesn't understand it. This is acceptable — the
        # orchestrator will spawn another capability agent on the next cycle
        # once the error map shows high error for the new DOF actions.
        return CapabilityOutput(..., success=False)
```

**DOF unlock is NOT rolled back on failure.** The simulation gains the
new DOF regardless of whether fine-tuning completed. The world model's
high error on the new DOF actions will trigger exploration agents
automatically. This is intentional — the system self-corrects.

---

## REACTIVE AGENT

**File:** `backend/agents/reactive.py`

Runs on `comparison_sim`. Zero imports from `world_model`, `anthropic`,
or `orchestrator`. This is enforced — the reactive agent must have no
access to the world model to be a valid comparison baseline.

```python
class ReactiveAgent:
    """Greedy policy with no lookahead. No world model. No Claude API.
    Demonstrates what surgical robots do without predictive planning."""

    def __init__(self, env: SurROLTissueEnv):
        self._env = env     # comparison_sim only
        # NO world_model parameter — by design

    async def run(self) -> None:
        obs = await asyncio.to_thread(self._env.reset)
        step = 0

        while True:
            state_vec = self._env.get_state_vector()
            action_vec = self._select_action(state_vec)
            surrol_action = action_to_surrol(
                action_vec, self._env.config.dof
            )
            obs, reward, done, _ = await asyncio.to_thread(
                self._env.step, surrol_action
            )
            write_event("simulation_step", sim_id="comparison",
                        step=step, reward=float(reward),
                        tissue_mean=float(state_vec[0:256].mean()),
                        vessel_damaged=bool(state_vec[779+3::4].max() > 0.5),
                        target_reached=bool(state_vec[778] > 0.5),
                        task_failed=bool(self._env._task_failed),
                        active_dof=self._env.config.dof,
                        ee_pos=state_vec[768:771].tolist())
            step += 1
            if done:
                obs = await asyncio.to_thread(self._env.reset)
                step = 0
            await asyncio.sleep(0.05)  # yield event loop

    def _select_action(self, state_vec: np.ndarray) -> np.ndarray:
        """Greedy: move toward target, cauterize if bleeding, cut near target."""
        ee    = state_vec[768:770]    # normalized EE xy
        tgt   = state_vec[776:778]    # normalized target xy
        bleed = state_vec[512:768].sum()

        vec = np.zeros(12, dtype=np.float32)
        if bleed > 0:
            vec[9] = 1.0   # CAUTERIZE mode
            vec[11] = 1.0
        elif np.linalg.norm(tgt - ee) > 0.1:
            d = (tgt - ee) / (np.linalg.norm(tgt - ee) + 1e-8)
            vec[0:2] = d * 0.1  # MOVE
            vec[7] = 1.0
            vec[11] = 0.7
        else:
            d = (tgt - ee) / (np.linalg.norm(tgt - ee) + 1e-8)
            vec[0:2] = d * 0.05  # slow approach near target
            vec[7] = 1.0
            vec[11] = 0.4
        return vec
```

The reactive agent WILL damage vessels. It has no model of where they
are relative to its approach path. This is the demo's closing argument:
MPC (with world model) avoids vessels; reactive (without) doesn't.

---

## PRODUCTION ENGINEERING

These patterns are not optional nice-to-haves. Each addresses a failure
mode that will occur in a 36-hour build under time pressure.

### Circuit Breaker

If exploration agents consistently fail in the same region (3+ times),
the orchestrator is likely spawning into an environment condition that
isn't fixable by the current world model (e.g., region always fails due
to a simulation edge case near the grid boundary).

```python
CIRCUIT_BREAKER_THRESHOLD = 3    # failures before opening
CIRCUIT_BREAKER_BACKOFF   = 300  # seconds (5 min)

# In collect_status_node, after processing failed_agents:
for agent_id in state["failed_agents"]:
    region = get_region_for_agent(agent_id)
    if region:
        new_count = state["region_failure_counts"].get(region, 0) + 1
        updates["region_failure_counts"][region] = new_count
        if new_count >= CIRCUIT_BREAKER_THRESHOLD:
            updates["region_backoff_until"][region] = time.time() + CIRCUIT_BREAKER_BACKOFF
            write_event("prediction_error_high",
                        region=region, error="circuit_breaker_open",
                        failure_count=new_count)
```

Circuit breakers reset on successful completion (`region_failure_counts[r] = 0`).
They prevent the orchestrator from burning API credits and sim instances
on a region that can't be improved with current architecture.

### Agent Timeout

10-minute hard timeout on every agent. Without it, a stalled Claude API
call or SurRoL deadlock leaves a zombie agent holding a simulation instance
permanently. With MAX_CONCURRENT=4, one zombie blocks 25% of agent capacity.

```python
AGENT_TIMEOUT_SECONDS = 600  # 10 minutes

# Already shown in agent run() implementations above using asyncio.timeout()
```

### Fine-Tune Backpressure Queue

If two exploration agents complete simultaneously, both try to call
`world_model.fine_tune()`. The second call blocks on `_finetune_lock`
for 2-5 minutes. Its samples are in memory. If the process restarts
during this wait, those samples are lost.

Solution: queue the samples before waiting on the lock, not during the wait.

```python
# In world_model.py:
_fine_tune_queue: asyncio.Queue[list[tuple]] = asyncio.Queue(maxsize=10)

async def fine_tune_worker():
    """Background task in main.py that drains the fine-tune queue."""
    while True:
        samples = await _fine_tune_queue.get()
        try:
            await world_model.fine_tune(samples)
        except Exception as e:
            write_event("agent_failed", error=f"fine_tune_error: {e}")
        finally:
            _fine_tune_queue.task_done()

# ExplorationAgent.run() — instead of awaiting fine_tune directly:
await world_model._fine_tune_queue.put(samples)  # non-blocking, queued
# agent_completed event written immediately after queuing, not after fine_tune
```

Trade-off: `error_after` in `agent_completed` will be computed before
fine-tuning completes. The heatmap animation fires immediately on
`agent_completed`, but the actual error map update happens when the
queue worker runs. For demo purposes this is acceptable — the visual
timing matters more than the metric precision. Annotate this decision
in MASTER_STATE.md under KNOWN DEVIATIONS.

### API Cost Tracking

Each exploration agent logs its API token usage. The total across the
demo session is visible in W&B and in the agent_completed events.

```python
# In agent_completed event payload:
"api_tokens_used": total_tokens,
"estimated_cost_usd": total_tokens * 3e-6  # claude-sonnet-4-6 rate
```

Log aggregate in W&B:
```python
wandb.log({"cumulative_api_tokens": sum_all_agents,
           "cumulative_api_cost_usd": sum_all_agents * 3e-6})
```

Budget estimate for full demo: 4 agents × 100 steps × (1 Claude call/5 steps)
= 80 calls × ~150 tokens avg = 12,000 tokens ≈ $0.036. Negligible.

### Idempotency via world_model_version

If fine_tune() is called twice with the same samples (retry scenario),
check the version:

```python
async def fine_tune(self, samples, expected_version: int | None = None):
    if expected_version and self.belief.world_model_version != expected_version:
        # Already fine-tuned past this version — skip (idempotent)
        return 0.0
    # ... normal fine_tune logic
```

The orchestrator passes `expected_version=state["world_model_version"]`
when requesting fine_tune. If the version has already incremented, the
call is a no-op.

### Graceful Degradation on Claude API Unavailability

If all Claude API retries fail, exploration agents fall back to
`_random_focused_action()` — random actions targeting the region.
Data collection continues, though less efficiently.

The agent_completed event notes `api_fallback: True` in the payload.
The agent feed shows "[API fallback]" instead of reasoning.
Judges may notice an agent with empty reasoning — use the contingency
script in BUILD_PLAN.md (Scope Cut 2) which describes how to address this.

---

## AGENT FEED CALLBACK

The agent feed in the frontend shows Claude's streaming reasoning
token-by-token. The callback is a function passed into `_call_claude_with_retry`.

```python
class AgentFeedBuffer:
    """Thread-safe buffer for streaming agent reasoning to the frontend.
    Collected tokens are flushed to the event log every 5 seconds
    and on agent completion."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._buffer: list[str] = []
        self._lock = threading.Lock()  # threading.Lock because callback
                                       # is called from stream thread

    def append(self, token: str) -> None:
        with self._lock:
            self._buffer.append(token)

    def flush(self, step: int) -> None:
        with self._lock:
            if not self._buffer:
                return
            text = "".join(self._buffer)
            self._buffer.clear()
        # Write to event log (asyncio-safe via create_task)
        asyncio.get_event_loop().call_soon_threadsafe(
            asyncio.create_task,
            asyncio.coroutine(lambda: write_event(
                "agent_step", agent_id=self.agent_id,
                step=step, reasoning=text
            ))()
        )
```

The frontend `DetailView/AgentFeed.tsx` subscribes to `agent_step` events
for the selected agent and renders `payload.reasoning` incrementally.

---

## OBSERVABILITY (LANGSMITH)

```python
# In main.py, before creating the LangGraph graph:
import os
from langsmith import Client

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "soma-surgical"
    # LangGraph automatically traces all node executions to LangSmith
```

LangSmith provides:
- Per-node latency breakdown across the orchestrator cycle
- Full input/output for each node (for debugging routing decisions)
- Error traces when a node raises an exception
- Token usage and cost across all Claude API calls

For a hackathon demo, LangSmith is optional — wrap in try/except and
degrade to console logs if not configured. But having the trace URL
available during demos adds significant engineering credibility if
a judge asks "how do you debug the agent decisions?"

---

## CONSTANTS

All thresholds. Change one → must update QUICK_REFERENCE.md.

```python
# Orchestrator
EXPLORE_THRESHOLD      = 0.15  # regional error to trigger spawn
MAX_CONCURRENT_AGENTS  = 4     # total exploration agents max
ORCHESTRATOR_SLEEP_S   = 2.0   # seconds between cycles
SPAWN_COOLDOWN_S       = 30.0  # per-region respawn delay

# DOF unlock thresholds (indexed by current DOF)
DOF_THRESHOLDS = {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.04, 5: 0.03}

# Exploration
EXPLORE_N_STEPS       = 100    # steps per exploration episode
CLAUDE_CALL_EVERY_N   = 5      # steps between Claude API calls
AGENT_TIMEOUT_SECONDS = 600    # 10-minute hard kill

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD = 3    # failures before opening
CIRCUIT_BREAKER_BACKOFF_S = 300  # 5-minute cooldown

# Claude API
CLAUDE_MODEL          = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS     = 300
CLAUDE_MAX_RETRIES    = 3
API_SEMAPHORE_LIMIT   = 10       # max concurrent Claude calls

# Fine-tune queue
FINE_TUNE_QUEUE_MAXSIZE = 10     # if full: block agent until space
```

---

## AGENT LIFECYCLE INVARIANTS

These complement the invariants in ARCHITECTURE.md.

---

**AGENT_SPAWNED_BEFORE_LOOP INVARIANT**

`agent_spawned` event must be written before the agent's episode loop
starts, not after. This is handled by `spawn_exploration_node` before
issuing the `Send` command. The exploration agent itself never writes
`agent_spawned` — the orchestrator does.

Why: the canvas creates nodes from `agent_spawned` events. If the event
arrives after the agent has been running for seconds, the node appears
on the canvas already completed or mid-progress. The canvas tree looks
like branches materialized from nowhere.

Test: `test:agents:spawned_before_loop`
```python
# Confirm agent_spawned event timestamp is earlier than first agent_step
events = get_events_for_agent(agent_id)
spawned = next(e for e in events if e["event_type"] == "agent_spawned")
first_step = next(e for e in events if e["event_type"] == "agent_step")
assert spawned["timestamp"] < first_step["timestamp"]
```

---

**AGENT_COMPLETED_AFTER_FINETUNE INVARIANT**

`agent_completed` must be written AFTER `world_model.fine_tune()` returns
(or after `_fine_tune_queue.put()` if using the backpressure pattern).
It must NOT be written before fine-tuning.

Why: `error_after` in `agent_completed` is what drives the frontend's
error reduction animation. If written before fine-tuning, `error_after`
equals `error_before` (fine-tuning hasn't run yet) and the animation
condition `error_after < error_before` is never true. The heatmap doesn't
animate. The most important demo moment is broken.

Test: `test:agents:completed_after_finetune`
```python
# The world_model_version must have incremented before agent_completed
events = get_events_for_agent(agent_id)
completed = next(e for e in events if e["event_type"] == "agent_completed")
wm_updates = [e for e in get_events_since(0)
              if e["event_type"] == "world_model_updated"
              and e["timestamp"] < completed["timestamp"]]
assert len(wm_updates) >= 1, "world_model_updated must precede agent_completed"
```

---

**SIMULATION RELEASED ON FAILURE INVARIANT**

Every code path in `ExplorationAgent.run()` must call `env.close()`.
Use `finally:` to guarantee release regardless of exception or timeout.

Why: leaked SurRoL instances keep PyBullet physics servers alive.
Each server occupies ~200MB RAM. With 4 agents and 2 leaks per run,
the process exhausts memory within an hour. The symptom is a gradual
slowdown that's hard to diagnose as a memory leak.

Test: `test:agents:sim_released`
```python
# After agent completes or fails, PyBullet server count must not increase
import pybullet as p
before_count = p.getNumBodies(physicsClientId=...)  # approximate
agent = ExplorationAgent(...); await agent.run()
after_count = p.getNumBodies(physicsClientId=...)
assert after_count <= before_count  # no leaked bodies
```

---

**REACTIVE_AGENT_ISOLATION INVARIANT**

`backend/agents/reactive.py` must have zero imports from:
`world_model`, `prediction_net`, `belief_state`, `anthropic`, `orchestrator`.

Why: the comparison demo's story is "same anatomy, no world model."
Any import from world model modules creates a dependency that the agent
might accidentally use (especially if an IDE auto-completes a `world_model.`
call). The grep test catches this before it causes a silent correctness bug.

Test: `test:agents:reactive_isolation`
```bash
grep -E "from backend\.(world_model|prediction_net|belief_state|orchestrator)" \
    backend/agents/reactive.py
# Must return empty. Any hit is a violation.
```

---

## PITFALLS — SYMPTOM FIRST

---

**SYMPTOM: Canvas shows no agent nodes even though logs show agents
completing. `get_events_since(0)` returns `agent_completed` events
but no `agent_spawned` events for those agents.**

CAUSE: `agent_spawned` written after the episode completed (or not at
all). The canvas creates nodes from `agent_spawned`. Without it,
`agent_completed` arrives for an unknown agent — the canvas ignores it.
CONFIRM: `SELECT event_type, timestamp FROM events WHERE agent_id = 'exp_xxx'
ORDER BY timestamp` — if `agent_spawned` is absent or has later timestamp
than `agent_step`: violated.
FIX: Move `agent_spawned` write to `spawn_exploration_node` in
orchestrator.py, before the `Send`. Never write it inside the agent.
Test: `test:agents:spawned_before_loop`

---

**SYMPTOM: Heatmap shows no animation on agent completion.
`agent_completed` events exist with non-null `error_before` and
`error_after`, but `error_before == error_after` in all of them.**

CAUSE: `_measure_region_error()` called before fine-tuning completed.
The error map hasn't been updated by fine-tuning yet, so pre and post
measurements are identical.
CONFIRM: Check `world_model_version` in `agent_completed.payload` vs
the `world_model_updated` event timestamp. If `agent_completed` is
earlier than `world_model_updated`: fine-tune ran after the measurement.
FIX: call `_measure_region_error()` after `fine_tune()` returns. If
using the backpressure queue, call `_measure_region_error()` inside
the fine_tune_worker after the fine-tune completes, then write
`agent_completed` from the worker.
Test: `test:agents:completed_after_finetune`

---

**SYMPTOM: `StopIteration` inside `_call_claude_with_retry`.
Agents log "Claude did not call the select_action tool" frequently.**

CAUSE: Claude is responding with text instead of a tool call. This
happens when `tool_choice` is not set or set to `"auto"` instead of
`"any"`. With `"auto"`, Claude decides whether to use a tool. It
often chooses to describe the action in prose instead.
CONFIRM: Set `LANGCHAIN_TRACING_V2=true` and inspect the LangSmith
trace for the failing call. The response content will have type "text"
instead of "tool_use".
FIX: Set `tool_choice={"type": "any"}` in the API call. This forces
a tool call on every request regardless of Claude's preference.

---

**SYMPTOM: `RuntimeError: no running event loop` inside `AgentFeedBuffer.flush()`.**

CAUSE: `asyncio.get_event_loop()` called from a thread that doesn't
have a running event loop. `flush()` is called from the main asyncio
context (not a thread), but `call_soon_threadsafe` is designed for
cross-thread calls and behaves differently when called from within
the event loop.
FIX: From within the event loop, use `asyncio.create_task()` directly:
```python
# In flush(), if already in asyncio context:
asyncio.create_task(write_event_async(...))
```
The callback passed to the streaming API runs in the event loop's
thread, so `create_task` is safe there. Use `call_soon_threadsafe`
only if calling from an actual `asyncio.to_thread()` worker.

---

**SYMPTOM: System slows to a crawl after 30-60 minutes. Memory usage
grows steadily. Logs show "physics server warning" from PyBullet.**

CAUSE: SIM RELEASED ON FAILURE INVARIANT violated. Failed agents didn't
close their SurRoL instances. Each leaked instance keeps a PyBullet
physics server running.
CONFIRM: Count running physics servers:
`python -c "import pybullet as p; print(p.getNumBodies())"` — if count
grows over time: leaked instances.
FIX: Ensure `finally: await asyncio.to_thread(self.env.close)` in every
agent `run()` method. Check CapabilityAgent too — it creates a temp env.
Test: `test:agents:sim_released`

---

**SYMPTOM: LangGraph graph raises `GraphRecursionError` during demo.
The graph cycles indefinitely between orchestrator and collect_status.**

CAUSE: `collect_status_node` is not properly updating `completed_agents`
and `failed_agents` in the state, so the orchestrator never sees agents
as done and keeps re-entering the same cycle.
CONFIRM: Log the state at each orchestrator cycle entry. If `active_agents`
never shrinks despite agents completing: the status collection is broken.
FIX: Set `recursion_limit` in graph config to prevent runaway:
```python
graph.invoke(initial_state, config={"recursion_limit": 100})
```
Then fix `collect_status_node` to properly drain `completed_agents`
and `failed_agents` back into `active_agents` removal.

---

## ROOT AGENTS.MD (OH MY OPENCODE)

The root `/AGENTS.md` must exist for Oh My OpenCode. It is auto-injected
into every OpenCode session. Keep it focused on BUILD instructions only.

Minimal root `/AGENTS.md` template:

```markdown
# SOMA — Agent Instructions for Oh My OpenCode

## Project Context
SOMA is a surgical robot simulation that teaches itself.
Backend: Python + FastAPI + LangGraph + Claude API + SurRoL + PyBullet.
Frontend: React + TypeScript + @xyflow/react v12 + Plotly.

## Build Checkpoint Protocol
1. Read MASTER_STATE.md before starting any task.
2. Implement exactly the task in your prompt — nothing beyond it.
3. Run the named test target for your component (see ARCHITECTURE.md).
4. Update MASTER_STATE.md: completed tasks, deviations, interfaces.
5. Commit: `git commit -m "task: [id] [name]"`

## Agent Delegation Hints
- Architecture or debugging questions → Oracle
- SurRoL, LangGraph, or PyTorch API questions → Librarian (Context7)
- Standard implementation tasks → Sisyphus
- Plan a complex component before building → Prometheus

## Critical Invariants
See docs/specs/ARCHITECTURE.md. The nine named invariants are not
guidelines — violations produce silent demo failures that are hard to
diagnose under time pressure.

## Never Do
- Import WorldModel inside an agent or API handler
- Use numpy.random.seed() — use RandomState(seed)
- Use BatchNorm — use LayerNorm
- Modify a component that's already marked VERIFIED in MASTER_STATE.md
- Write agent_spawned after the episode starts
```

---

## TESTING PROTOCOL

Run only these after Tasks 1.6–1.7. Sprint checkpoint runs full suite.

```bash
# Orchestrator (Task 1.6)
pytest tests/test_orchestrator.py -v
pytest tests/test_orchestrator.py::test_graph_compiles -v
pytest tests/test_orchestrator.py::test_route_decision_explore -v
pytest tests/test_orchestrator.py::test_route_decision_unlock -v
pytest tests/test_orchestrator.py::test_circuit_breaker -v
pytest tests/test_orchestrator.py::test_spawn_writes_event_first -v

# Agents (Task 1.7)
pytest tests/test_agents.py -v
pytest tests/test_agents.py::test_spawned_before_loop -v            # MUST PASS
pytest tests/test_agents.py::test_completed_after_finetune -v       # MUST PASS
pytest tests/test_agents.py::test_sim_released -v                   # MUST PASS
pytest tests/test_agents.py::test_reactive_isolation -v             # MUST PASS
pytest tests/test_agents.py::test_api_fallback_on_error -v
pytest tests/test_agents.py::test_agent_timeout -v
```

### Minimal Set for Task Completion

**Task 1.6 (orchestrator.py):**
```python
# test:orchestrator:graph_compiles
graph = build_soma_graph(mock_world_model, mock_sim_manager)
assert graph is not None  # compile() succeeded

# test:orchestrator:route_returns_valid_strings
# route_orchestrator must return one of: "spawn_explore", "unlock_dof", "wait"
for state in [low_error_state, high_error_state, mixed_state]:
    result = route_orchestrator(state)
    assert result in ("spawn_explore", "unlock_dof", "wait")
```

**Task 1.7 (agents):**
```python
# test:agents:reactive_isolation (grep test — fastest)
result = subprocess.run(
    ["grep", "-E",
     r"from backend\.(world_model|prediction_net|belief_state|orchestrator)",
     "backend/agents/reactive.py"],
    capture_output=True
)
assert result.stdout == b"", f"ReactiveAgent imports world model: {result.stdout}"

# test:agents:tool_choice_any (Claude API always returns tool call)
# Mock the Anthropic client to return a text response (no tool)
# Verify: AgentAPIError is raised (not StopIteration or KeyError)
# This catches the tool_choice="auto" bug before it reaches production
```

---

## VERIFICATION CHECKLIST

```
ORCHESTRATOR (Task 1.6):
  □ graph.compile() succeeds with MemorySaver
  □ Orchestrator cycles every 2 seconds (check log timestamps)
  □ agent_spawned written before Send() issued
  □ should_unlock() returns True at correct thresholds
  □ MAX_CONCURRENT_AGENTS enforced (4th agent not spawned when 4 active)
  □ Circuit breaker prevents respawn after 3 failures

EXPLORATION AGENT (Task 1.7):
  □ agent_spawned event exists before first agent_step
  □ Claude API tool use returns structured action (not text)
  □ Streaming: agent feed shows reasoning incrementally
  □ agent_completed written after world_model_updated
  □ error_after < error_before by at least 0.01 on successful run
  □ env.close() called on success AND failure paths
  □ API fallback works: random actions collected when API fails

CAPABILITY AGENT (Task 1.7):
  □ capability_unlocked event written before agent_completed
  □ primary_sim.config.dof updated
  □ comparison_sim.config.dof updated (same step)
  □ DOF env created with seed = 42 + target_dof
  □ fine_tune called with epochs=10 (not default 5)

REACTIVE AGENT (Task 1.7):
  □ grep: zero imports from world_model, anthropic modules
  □ Writes simulation_step events with sim_id='comparison'
  □ Resets episode on task_complete or task_failed
  □ await asyncio.sleep(0.05) present in loop
```
