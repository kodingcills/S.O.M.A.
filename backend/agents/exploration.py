from __future__ import annotations

import asyncio
import os
from typing import Final, TypedDict

import anthropic
import numpy as np

from backend.event_log import write_event

EXPLORE_N_STEPS: Final = 100
CLAUDE_CALL_EVERY_N: Final = 5
AGENT_TIMEOUT_SECONDS = 600
CLAUDE_MODEL: Final = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS: Final = 300
CLAUDE_MAX_RETRIES: Final = 3
TOP_K: Final = 5

Sample = tuple[np.ndarray, int, np.ndarray]

EXPLORE_TOOL = {
    "name": "select_action",
    "description": "Select one of the five JSD-ranked real Craftax actions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidate_rank": {
                "type": "integer",
                "minimum": 1,
                "maximum": TOP_K,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["candidate_rank", "reasoning"],
    },
}


class ExplorationInput(TypedDict):
    agent_id: str
    region: str
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


class ToolInput(TypedDict):
    candidate_rank: int
    reasoning: str


class AgentAPIError(Exception):
    pass


_api_semaphore = asyncio.Semaphore(10)


async def _call_claude(
    client,
    prompt: str,
    feed_callback,
    max_retries: int = CLAUDE_MAX_RETRIES,
) -> tuple[ToolInput, int]:
    connection_error: anthropic.APIConnectionError | None = None
    for attempt in range(max_retries):
        try:
            async with _api_semaphore:
                async with client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    tools=[EXPLORE_TOOL],
                    tool_choice={"type": "any"},
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for token in stream.text_stream:
                        feed_callback(token)
                    final = await stream.get_final_message()
            tool_block = next(
                (
                    block
                    for block in final.content
                    if getattr(block, "type", None) == "tool_use"
                    and getattr(block, "name", None) == "select_action"
                ),
                None,
            )
            if tool_block is None:
                raise AgentAPIError("Claude did not call select_action")
            usage = final.usage.input_tokens + final.usage.output_tokens
            return tool_block.input, int(usage)
        except anthropic.RateLimitError:
            await asyncio.sleep(2**attempt)
        except anthropic.APIConnectionError as exc:
            connection_error = exc
            if attempt + 1 < max_retries:
                await asyncio.sleep(1.0)
    raise AgentAPIError("Claude API unavailable after retries") from connection_error


async def _write_feed(agent_id: str, step: int, reasoning: str) -> None:
    write_event("agent_step", agent_id=agent_id, step=step, reasoning=reasoning)


class AgentFeedBuffer:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._tokens: list[str] = []

    def append(self, token: str) -> None:
        self._tokens.append(token)

    def flush(self, step: int) -> None:
        if self._tokens:
            reasoning = "".join(self._tokens)
            self._tokens.clear()
            asyncio.create_task(_write_feed(self.agent_id, step, reasoning))


class ExplorationAgent:
    def __init__(
        self,
        agent_id: str,
        world_model,
        driver,
        region: str,
        error_before: float,
    ) -> None:
        self.agent_id = agent_id
        self._world_model = world_model
        self._driver = driver
        self.region = region
        self.error_before = float(error_before)
        self._buffer = AgentFeedBuffer(agent_id)
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self._client = anthropic.AsyncAnthropic() if has_key else None

    def _ranked_candidates(self):
        candidates = self._driver.evaluate_stamped_candidates()
        return sorted(
            candidates,
            key=lambda candidate: float(candidate.output.J_ua),
            reverse=True,
        )[:TOP_K]

    def _build_prompt(self, ranked, step: int) -> str:
        rows = [
            f"rank={rank} action={candidate.output.action_idx} "
            f"jsd={float(candidate.output.J_ua):.6f} "
            f"reward={float(candidate.output.reward_expectation):.6f}"
            for rank, candidate in enumerate(ranked, start=1)
        ]
        return (
            f"Explore Craftax uncertainty region {self.region} at step {step}.\n"
            + "\n".join(rows)
        )

    async def _advisory_step(self, step: int) -> tuple[int, int, bool]:
        ranked = await asyncio.to_thread(self._ranked_candidates)
        if self._client is None:
            driver_step, _probabilities = await asyncio.to_thread(
                self._driver.step_controller
            )
            self._buffer.append("[no API key — released controller fallback]")
            self._buffer.flush(step)
            return int(driver_step.action), 0, bool(driver_step.done)
        try:
            tool_input, tokens = await _call_claude(
                self._client,
                self._build_prompt(ranked, step),
                self._buffer.append,
            )
            selected = ranked[int(tool_input["candidate_rank"]) - 1]
            driver_step = await asyncio.to_thread(
                self._driver.step_with_action, int(selected.output.action_idx)
            )
            return int(driver_step.action), tokens, bool(driver_step.done)
        except (AgentAPIError, IndexError, KeyError, TypeError, ValueError) as exc:
            self._buffer.append(f"[step {step}] API fallback: {exc}")
            driver_step, _probabilities = await asyncio.to_thread(
                self._driver.step_controller
            )
            return int(driver_step.action), 0, bool(driver_step.done)
        finally:
            self._buffer.flush(step)

    async def run_episode(self, n_steps: int | None = None) -> tuple[list[Sample], int]:
        samples: list[Sample] = []
        tokens = 0
        for step in range(EXPLORE_N_STEPS if n_steps is None else n_steps):
            raw_obs = self._driver.env.raw_observation()
            if step % CLAUDE_CALL_EVERY_N == 0:
                action, used, done = await self._advisory_step(step)
                tokens += used
            else:
                driver_step, _probabilities = await asyncio.to_thread(
                    self._driver.step_controller
                )
                action = int(driver_step.action)
                done = bool(driver_step.done)
            next_raw_obs = self._driver.env.raw_observation()
            samples.append((raw_obs.copy(), action, next_raw_obs.copy()))
            if step % 10 == 0:
                write_event("agent_step", agent_id=self.agent_id, step=step)
            if done:
                await asyncio.to_thread(self._driver.reset, 42)
        return samples, tokens

    def _region_error(self, fallback: float) -> float:
        override = self._world_model.belief.regional_override
        return fallback if override is None else float(override[self.region])

    async def _ingest_cycle(self) -> None:
        prediction = await self._world_model.predict()
        transition = await self._world_model.update(prediction)
        await self._world_model.ingest(transition)

    async def run(self, n_steps: int | None = None) -> ExplorationOutput:
        start_error = self._region_error(self.error_before)
        try:
            async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
                samples, tokens = await self.run_episode(n_steps)
                await self._ingest_cycle()
                final_error = self._region_error(start_error)
            write_event(
                "agent_completed",
                agent_id=self.agent_id,
                region=self.region,
                error_before=start_error,
                error_after=final_error,
                samples_collected=len(samples),
                fine_tune_loss=0.0,
                api_tokens_used=tokens,
            )
            return ExplorationOutput(
                agent_id=self.agent_id,
                region=self.region,
                error_before=start_error,
                error_after=final_error,
                samples_collected=len(samples),
                fine_tune_loss=0.0,
                api_tokens_used=tokens,
                success=final_error < start_error - 0.05,
            )
        except TimeoutError:
            write_event(
                "agent_failed", agent_id=self.agent_id, region=self.region, error="timeout"
            )
            return ExplorationOutput(
                agent_id=self.agent_id,
                region=self.region,
                error_before=start_error,
                error_after=start_error,
                samples_collected=0,
                fine_tune_loss=0.0,
                api_tokens_used=0,
                success=False,
            )
        finally:
            close = getattr(self._driver, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
            else:
                env_close = getattr(self._driver.env, "close", None)
                if callable(env_close):
                    await asyncio.to_thread(env_close)
