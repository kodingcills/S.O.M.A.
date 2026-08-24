"""ExplorationAgent — SOMA Task 1.7a.

Collects focused simulation data in a high-error region, fine-tunes the
world model, and reports the error delta. Claude API selects actions with
forced tool use; its reasoning streams token-by-token into the agent feed.

Spec: docs/specs/ORCHESTRATION.md "EXPLORATION AGENT", "AGENT FEED BUFFER",
"PRODUCTION PATTERNS". Invariants honored:
  - SEED REPRODUCIBILITY (env always seed=42 — same anatomy as primary)
  - SIMULATION RELEASED ON FAILURE (finally: env.close())
  - C5 error_before/error_after as top-level event columns, written AFTER fine_tune
"""

from __future__ import annotations

import asyncio
import os
from typing import TypedDict

import numpy as np

from backend.belief_state import BeliefState, region_to_slice
from backend.event_log import write_event
from backend.simulation import (
    SimConfig,
    action_to_surrol,
    create_env,
)

try:
    import anthropic
except ImportError:  # pragma: no cover - anthropic optional in test envs
    anthropic = None

# Constants (QUICK_REFERENCE.md / ORCHESTRATION.md). No shared constants
# module yet — defined locally per build convention.
EXPLORE_N_STEPS = 100
CLAUDE_CALL_EVERY_N = 5
AGENT_TIMEOUT_SECONDS = 600
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 300
CLAUDE_MAX_RETRIES = 3
API_SEMAPHORE_LIMIT = 10
MEASURE_CYCLES = 20

_api_semaphore = asyncio.Semaphore(API_SEMAPHORE_LIMIT)

_MODE_INDEX = {"MOVE": 7, "CAUTERIZE": 9, "WAIT": 10}

Sample = tuple[np.ndarray, np.ndarray, np.ndarray]

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
            "delta_x": {"type": "number", "description": "EE x delta [-1, 1]"},
            "delta_y": {"type": "number", "description": "EE y delta [-1, 1]"},
            "delta_z": {"type": "number", "description": "EE z delta [-1, 1]"},
            "magnitude": {"type": "number", "description": "Action scale [0, 1]"},
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining this action choice.",
            },
        },
        "required": ["action_type", "reasoning"],
    },
}


class ExplorationInput(TypedDict):
    agent_id: str     # format: 'exp_{6 hex chars}'
    region: str       # one of the 6 named regions from BeliefState
    error_before: float


class ExplorationOutput(TypedDict):
    agent_id: str
    region: str
    error_before: float
    error_after: float
    samples_collected: int
    fine_tune_loss: float
    api_tokens_used: int
    success: bool


class AgentAPIError(Exception):
    """Claude API failed after retries, or returned no tool call."""


async def _call_claude(
    client,
    prompt: str,
    feed_callback,
    max_retries: int = CLAUDE_MAX_RETRIES,
) -> tuple[dict, int]:
    """Streams reasoning tokens to feed_callback; returns (tool_input, tokens)."""
    last_connection_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with _api_semaphore:
                async with client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    tools=[EXPLORE_TOOL],
                    tool_choice={"type": "any"},  # force tool call every response
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for token in stream.text_stream:
                        feed_callback(token)
                    final = await stream.get_final_message()

            tool_block = next(
                (b for b in final.content if getattr(b, "type", None) == "tool_use"),
                None,
            )
            if tool_block is None:
                raise AgentAPIError("Claude responded without calling select_action.")
            tokens = final.usage.input_tokens + final.usage.output_tokens
            return tool_block.input, tokens

        except AgentAPIError:
            raise  # no-tool-call is deterministic — retrying won't help
        except anthropic.RateLimitError:
            await asyncio.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
        except anthropic.APIConnectionError as exc:
            last_connection_error = exc
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(1.0)

    raise AgentAPIError(f"Claude API failed after {max_retries} retries.") from last_connection_error


async def _write_agent_step_async(agent_id: str, step: int, reasoning: str) -> None:
    try:
        write_event("agent_step", agent_id=agent_id, step=step, reasoning=reasoning)
    except Exception:  # noqa: BLE001 — feed must never crash an episode
        pass


class AgentFeedBuffer:
    """Accumulates streamed reasoning tokens; flushes to the event log."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._tokens: list[str] = []

    def append(self, token: str) -> None:
        self._tokens.append(token)

    def flush(self, step: int) -> None:
        if not self._tokens:
            return
        reasoning = "".join(self._tokens)
        self._tokens.clear()
        asyncio.create_task(_write_agent_step_async(self.agent_id, step, reasoning))


def _tool_input_to_action(tool_input: dict) -> np.ndarray:
    vec = np.zeros(12, dtype=np.float32)
    mode = _MODE_INDEX.get(str(tool_input.get("action_type", "MOVE")).upper(), 7)
    vec[mode] = 1.0
    vec[0] = float(np.clip(float(tool_input.get("delta_x", 0.0)), -1.0, 1.0))
    vec[1] = float(np.clip(float(tool_input.get("delta_y", 0.0)), -1.0, 1.0))
    vec[2] = float(np.clip(float(tool_input.get("delta_z", 0.0)), -1.0, 1.0))
    vec[11] = float(np.clip(float(tool_input.get("magnitude", 0.6)), 0.0, 1.0))
    return vec


def _random_focused_action(rng) -> np.ndarray:
    vec = np.zeros(12, dtype=np.float32)
    vec[7] = 1.0  # MOVE
    vec[0:3] = rng.uniform(-0.4, 0.4, size=3).astype(np.float32)
    vec[11] = rng.uniform(0.2, 0.7)
    return vec


class ExplorationAgent:
    """Collects focused data in one high-error region, fine-tunes, reports.

    The SurRoL environment is created lazily inside run() (never __init__)
    so construction stays cheap and every path releases the PyBullet server
    via finally-close.
    """

    def __init__(
        self,
        agent_id: str,
        world_model,
        region: str,
        error_before: float,
        primary_sim_config: SimConfig | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._world_model = world_model
        self.region = region
        self.error_before = float(error_before)
        self.primary_sim_config = primary_sim_config
        self._rng = np.random.default_rng()
        self._buffer = AgentFeedBuffer(agent_id)
        self.env = None
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self._client = anthropic.AsyncAnthropic() if (anthropic and has_key) else None

    def _build_prompt(self, state_vec: np.ndarray, step: int) -> str:
        tissue_mean = float(state_vec[0:256].mean())
        bleeding_count = int(state_vec[512:768].sum())
        ee_pos = state_vec[768:771].tolist()
        active_dof = int(np.argmax(state_vec[799:805])) + 1
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

    async def run_episode(self, n_steps: int | None = None) -> tuple[list[Sample], int]:
        """Runs one exploration episode, collecting (state, action, next_state).

        Calls Claude every CLAUDE_CALL_EVERY_N steps; repeats the last action
        in between. On API failure (or no API key) falls back to random
        focused actions — data collection continues.

        SPEC DEVIATION: n_steps defaults to None and reads EXPLORE_N_STEPS at
        call time — a literal default freezes the value at import, making
        monkeypatched test overrides silent no-ops. Same call-time-read
        pattern as AGENT_TIMEOUT_SECONDS in agents/capability.py.
        """
        steps = EXPLORE_N_STEPS if n_steps is None else n_steps
        samples: list[Sample] = []
        total_tokens = 0
        last_action = np.zeros(12, dtype=np.float32)
        last_action[7] = 1.0  # default: MOVE

        await asyncio.to_thread(self.env.reset)

        # LOOP: exploration_episode
        # Pre-condition:  env.reset() completed; world model weights loaded.
        # Invariant:      each (state, action, next_state) triple is consistent
        #                 with the environment's actual physics.
        # Termination:    step count reaches n_steps.
        # Yield:          asyncio.to_thread() yields for every env.step() call.
        for step in range(steps):
            state_vec = self.env.get_state_vector()

            if step % CLAUDE_CALL_EVERY_N == 0:
                if self._client is not None:
                    try:
                        tool_input, tokens = await _call_claude(
                            self._client,
                            self._build_prompt(state_vec, step),
                            self._buffer.append,
                        )
                        last_action = _tool_input_to_action(tool_input)
                        total_tokens += tokens
                    except AgentAPIError as exc:
                        self._buffer.append(f"[step {step}] API fallback: {exc}")
                        last_action = _random_focused_action(self._rng)
                    finally:
                        self._buffer.flush(step)
                else:
                    self._buffer.append("[no API key — random focused actions]")
                    last_action = _random_focused_action(self._rng)
                    self._buffer.flush(step)

            _, _, done, _info = await asyncio.to_thread(
                self.env.step, action_to_surrol(last_action, self.env.config.dof)
            )
            next_state_vec = self.env.get_state_vector()
            samples.append((state_vec.copy(), last_action.copy(), next_state_vec.copy()))

            if step % 10 == 0:
                try:
                    write_event("agent_step", agent_id=self.agent_id, step=step)
                except Exception:  # noqa: BLE001 — telemetry never blocks collection
                    pass

            if done:
                await asyncio.to_thread(self.env.reset)

        return samples, total_tokens

    async def _measure_region_error(self) -> float:
        rows, cols = region_to_slice(self.region, self._world_model.belief)
        errors: list[float] = []
        await asyncio.to_thread(self.env.reset)

        for _ in range(MEASURE_CYCLES):
            state_vec = self.env.get_state_vector()
            action_vec = _random_focused_action(self._rng)
            predicted = self._world_model.network.predict(state_vec, action_vec)
            await asyncio.to_thread(
                self.env.step, action_to_surrol(action_vec, self.env.config.dof)
            )
            actual = self.env.get_state_vector()[0:256].reshape(16, 16)
            mse = float(((predicted.tissue - actual) ** 2)[rows, cols].mean())
            errors.append(mse)

        return float(np.mean(errors))

    async def run(self) -> ExplorationOutput:
        try:
            dof = int(self._world_model.belief.active_dof)
            # SEED INVARIANT: seed 42 == primary sim anatomy. Data collected on
            # any other seed would teach the model tissue that does not exist.
            config_source = self.primary_sim_config or SimConfig()
            self.env = await create_env(SimConfig(
                seed=42, dof=dof, n_vessels=config_source.n_vessels,
            ))

            try:
                async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
                    samples, tokens = await self.run_episode()
                    fine_tune_loss = await self._world_model.fine_tune(samples)
                    # C5: measured AFTER fine_tune returns so error_after
                    # reflects the update — this drives the demo's animation.
                    final_error = await self._measure_region_error()

                write_event(
                    "agent_completed",
                    agent_id=self.agent_id,
                    region=self.region,
                    error_before=float(self.error_before),
                    error_after=float(final_error),
                    samples_collected=len(samples),
                    fine_tune_loss=fine_tune_loss,
                    api_tokens_used=tokens,
                )
                return ExplorationOutput(
                    agent_id=self.agent_id,
                    region=self.region,
                    error_before=self.error_before,
                    error_after=float(final_error),
                    samples_collected=len(samples),
                    fine_tune_loss=float(fine_tune_loss),
                    api_tokens_used=tokens,
                    success=bool(final_error < self.error_before - 0.05),
                )

            except TimeoutError:
                write_event(
                    "agent_failed", agent_id=self.agent_id,
                    region=self.region, error="timeout",
                )
                return ExplorationOutput(
                    agent_id=self.agent_id,
                    region=self.region,
                    error_before=self.error_before,
                    error_after=self.error_before,
                    samples_collected=0,
                    fine_tune_loss=0.0,
                    api_tokens_used=0,
                    success=False,
                )

            except Exception as exc:
                write_event(
                    "agent_failed", agent_id=self.agent_id,
                    region=self.region, error=str(exc),
                )
                raise

        finally:
            # SIMULATION RELEASED ON FAILURE INVARIANT — every path closes env.
            if self.env is not None:
                try:
                    await asyncio.to_thread(self.env.close)
                except Exception:  # noqa: BLE001 — already-disconnected is fine
                    pass
