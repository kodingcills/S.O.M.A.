"""SOMA Task 1.7a — ExplorationAgent named invariant tests.

Zero network: Claude is mocked via fake stream/client objects; degraded mode
(_client=None) exercises the random-focused fallback path.

NOTE: event_log caches its DB path on first init, so events accumulate
across the test session despite per-test SOMA_DB_PATH — every event lookup
here is scoped by agent_id (unique per test) for that reason.

Named targets (ARCHITECTURE.md testing protocol):
  test_spawned_before_loop                    — C4: spawn precedes first step
  test_completed_after_finetune               — C5: completed follows fine-tune
  test_sim_released                           — env.close() on success AND failure
  test_reactive_isolation                     — grep-enforced zero WM imports
  test_api_fallback_on_error                  — AgentAPIError → random actions
  test_tool_choice_forces_structured_output   — A-1: no prose responses
  test_agent_timeout                          — hard kill → agent_failed(timeout)
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest

from backend.agents import exploration
from backend.agents.exploration import AgentAPIError, ExplorationAgent
from backend.belief_state import BeliefState
from backend.event_log import get_events_since, write_event
from backend.prediction_net import PredictedState
from backend.simulation import SimConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _stub_world_model(fine_tune=None):
    """WorldModel contract stub backed by a REAL BeliefState."""
    belief = BeliefState()

    async def default_fine_tune(samples, epochs=5):
        return 0.01

    def predict(state_vec, action_vec):
        return PredictedState(
            tissue=np.zeros((16, 16), dtype=np.float32),
            damage_prob=0.0,
            vessel_risk=np.zeros(5, dtype=np.float32),
            pred_ee_norm=np.zeros(4, dtype=np.float32),
        )

    return SimpleNamespace(
        belief=belief,
        fine_tune=fine_tune or default_fine_tune,
        network=SimpleNamespace(predict=predict),
    )


class _FakeStream:
    """Async context manager mimicking anthropic's streaming response."""

    def __init__(self, final_message, text_tokens: tuple[str, ...] = ()):
        self._final = final_message
        self._tokens = text_tokens

    @property
    def text_stream(self):
        async def _gen():
            for token in self._tokens:
                yield token

        return _gen()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._final


class _FakeMessages:
    """messages namespace: records stream() kwargs, replays canned behavior."""

    def __init__(self, final_message=None, error: Exception | None = None):
        self._final = final_message
        self._error = error
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeStream(self._final)


def _tool_use_message(tool_input: dict):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input=tool_input)],
        usage=SimpleNamespace(input_tokens=50, output_tokens=20),
    )


class _FakeConnectionError(anthropic.APIConnectionError):
    """Real anthropic exception type (isinstance checks in _call_claude)
    without needing a live httpx request object."""

    def __init__(self) -> None:
        super().__init__(message="connection down", request=SimpleNamespace())


@pytest.fixture
def fast_run(monkeypatch):
    """Shrink episode + measurement cost so the suite stays well under 90 s.

    Works because run_episode reads EXPLORE_N_STEPS at call time (SPEC
    DEVIATION noted in its docstring) and _measure_region_error reads
    MEASURE_CYCLES at call time.
    """
    monkeypatch.setattr(exploration, "EXPLORE_N_STEPS", 10)
    monkeypatch.setattr(exploration, "MEASURE_CYCLES", 2)


@pytest.fixture
def env_spy(monkeypatch):
    """Wraps exploration.create_env: captures SimConfigs, counts env.close().

    Fresh state per test — no cross-test contamination of close counts.
    """
    spy: dict = {"configs": [], "close_calls": 0}
    real_create_env = exploration.create_env

    async def spying_create_env(config: SimConfig):
        spy["configs"].append(config)
        env = await real_create_env(config)
        original_close = env.close

        def counting_close():
            spy["close_calls"] += 1
            original_close()

        env.close = counting_close
        return env

    monkeypatch.setattr(exploration, "create_env", spying_create_env)
    return spy


async def _drain_feed_tasks():
    """Yield ticks so AgentFeedBuffer's create_task'd writes land in the log."""
    for _ in range(3):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# test:agents:spawned_before_loop (C4)
# ---------------------------------------------------------------------------

async def test_spawned_before_loop(fast_run, env_spy):
    """agent_spawned timestamp must precede every agent_step timestamp;
    the dedicated sim config carries the SEED INVARIANT seed=42."""
    # Orchestrator writes spawn BEFORE the agent task starts (C4 producer side).
    write_event(
        "agent_spawned",
        agent_id="exp_test01",
        parent_id="orchestrator",
        region="upper_left",
        error_before=0.8,
    )

    stub = _stub_world_model()
    agent = ExplorationAgent("exp_test01", stub, "upper_left", 0.8)
    agent._client = None  # degraded mode — no network
    agent.env = await exploration.create_env(SimConfig(seed=42, dof=1))

    try:
        samples, tokens = await agent.run_episode(n_steps=12)
    finally:
        await asyncio.to_thread(agent.env.close)

    events = get_events_since(0)
    spawned_ts = next(
        e["timestamp"] for e in events
        if e["event_type"] == "agent_spawned" and e["agent_id"] == "exp_test01"
    )
    step_events = [
        e for e in events
        if e["event_type"] == "agent_step" and e["agent_id"] == "exp_test01"
    ]
    assert step_events, "episode must emit agent_step telemetry"
    assert all(spawned_ts < e["timestamp"] for e in step_events), (
        "C4 violated: agent_spawned must precede every agent_step"
    )

    assert len(samples) == 12
    assert tokens == 0, "degraded mode spends zero API tokens"

    assert env_spy["configs"][0].seed == 42, "SEED INVARIANT: always seed 42"


# ---------------------------------------------------------------------------
# test:agents:completed_after_finetune (C5)
# ---------------------------------------------------------------------------

async def test_completed_after_finetune(fast_run, env_spy):
    """agent_completed must be written AFTER world_model_updated, with
    error_before/error_after as non-null TOP-LEVEL columns."""

    async def recording_fine_tune(samples, epochs=5):
        write_event("world_model_updated", payload_loss=0.02)
        return 0.02

    stub = _stub_world_model(fine_tune=recording_fine_tune)
    agent = ExplorationAgent("exp_c5test", stub, "upper_left", 0.8)
    agent._client = None

    t0 = time.time()
    output = await agent.run()
    await _drain_feed_tasks()

    events = get_events_since(0)
    completed = next(
        e for e in events
        if e["event_type"] == "agent_completed" and e["agent_id"] == "exp_c5test"
    )
    wm_updates = [
        e for e in events
        if e["event_type"] == "world_model_updated"
        and t0 < e["timestamp"] < completed["timestamp"]
    ]
    assert len(wm_updates) >= 1, "C5 violated: no fine-tune before completion"

    assert completed["error_before"] == pytest.approx(0.8)
    assert isinstance(completed["error_after"], float)
    assert completed["payload"]["samples_collected"] == 10
    assert completed["payload"]["fine_tune_loss"] == pytest.approx(0.02)

    assert output["agent_id"] == "exp_c5test"
    assert output["samples_collected"] == 10
    assert output["fine_tune_loss"] == pytest.approx(0.02)
    assert set(output.keys()) == {
        "agent_id", "region", "error_before", "error_after",
        "samples_collected", "fine_tune_loss", "api_tokens_used", "success",
    }
    assert env_spy["close_calls"] == 1


# ---------------------------------------------------------------------------
# test:agents:sim_released
# ---------------------------------------------------------------------------

async def test_sim_released(fast_run, env_spy):
    """env.close() fires exactly once per run() — success AND failure paths."""
    agent = ExplorationAgent(
        "exp_close01", _stub_world_model(), "lower_right", 0.6
    )
    agent._client = None
    await agent.run()
    assert env_spy["close_calls"] == 1, "success path must release the sim"

    async def exploding_fine_tune(samples, epochs=5):
        raise RuntimeError("boom")

    failing = ExplorationAgent(
        "exp_close02", _stub_world_model(fine_tune=exploding_fine_tune),
        "lower_right", 0.6,
    )
    failing._client = None
    with pytest.raises(RuntimeError):
        await failing.run()
    assert env_spy["close_calls"] == 2, "failure path must also release the sim"


# ---------------------------------------------------------------------------
# test:agents:reactive_isolation
# ---------------------------------------------------------------------------

def test_reactive_isolation():
    """REACTIVE_AGENT_ISOLATION INVARIANT: zero imports from world_model,
    prediction_net, belief_state, orchestrator."""
    result = subprocess.run(
        [
            "grep",
            "-E",
            r"from backend\.(world_model|prediction_net|belief_state|orchestrator)",
            str(REPO_ROOT / "backend" / "agents" / "reactive.py"),
        ],
        capture_output=True,
    )
    assert result.stdout == b"", (
        f"ReactiveAgent imports world-model modules: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# test:agents:api_fallback_on_error
# ---------------------------------------------------------------------------

async def test_api_fallback_on_error(fast_run, env_spy, monkeypatch):
    """Connection errors exhaust retries → AgentAPIError; the episode then
    continues on random focused actions and logs '[API fallback]' to the feed."""
    broken = _FakeMessages(error=_FakeConnectionError())
    client = SimpleNamespace(messages=broken)

    # Retry backoff sleeps are pure latency here — zero them out. Scoped to
    # this test via monkeypatch; nothing else runs concurrently in-loop.
    async def instant_sleep(_delay):
        return None

    monkeypatch.setattr(exploration.asyncio, "sleep", instant_sleep)

    # Direct contract: retries exhausted → typed error, one attempt per retry.
    with pytest.raises(AgentAPIError, match="retries"):
        await exploration._call_claude(client, "prompt", lambda tok: None, max_retries=2)
    assert len(broken.calls) == 2

    # Episode-level behavior: fallback keeps collecting data.
    agent = ExplorationAgent("exp_fbtest", _stub_world_model(), "upper_left", 0.7)
    agent._client = client
    agent.env = await exploration.create_env(SimConfig(seed=42, dof=1))

    try:
        samples, tokens = await agent.run_episode(n_steps=6)
    finally:
        await asyncio.to_thread(agent.env.close)
    await _drain_feed_tasks()

    assert len(samples) == 6, "fallback path must still collect data"
    assert tokens == 0, "no successful API calls — zero token spend"
    assert len(broken.calls) == 2 + 2 * exploration.CLAUDE_MAX_RETRIES

    feed_events = [
        e for e in get_events_since(0)
        if e["event_type"] == "agent_step"
        and e["agent_id"] == "exp_fbtest"
        and "reasoning" in e["payload"]
    ]
    assert any("API fallback" in e["payload"]["reasoning"] for e in feed_events)


# ---------------------------------------------------------------------------
# test:agents:tool_choice_forces_structured_output (A-1)
# ---------------------------------------------------------------------------

async def test_tool_choice_forces_structured_output(monkeypatch):
    """A text-only response raises AgentAPIError naming select_action —
    never StopIteration/KeyError — and every call forces tool_choice='any'."""

    async def instant_sleep(_delay):
        return None

    monkeypatch.setattr(exploration.asyncio, "sleep", instant_sleep)

    text_only = _FakeMessages(
        final_message=SimpleNamespace(
            content=[SimpleNamespace(type="text")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
    )
    client = SimpleNamespace(messages=text_only)

    with pytest.raises(AgentAPIError, match="select_action"):
        await exploration._call_claude(client, "prompt", lambda tok: None, max_retries=1)

    call_kwargs = text_only.calls[0]
    assert call_kwargs["tool_choice"] == {"type": "any"}, (
        "A-1 regression guard: tool_choice must force structured tool calls"
    )
    assert call_kwargs["model"] == exploration.CLAUDE_MODEL
    assert call_kwargs["tools"] == [exploration.EXPLORE_TOOL]


# ---------------------------------------------------------------------------
# test:agents:agent_timeout
# ---------------------------------------------------------------------------

async def test_agent_timeout(fast_run, env_spy, monkeypatch):
    """Hard AGENT_TIMEOUT_SECONDS kill → success=False output and an
    agent_failed event carrying error='timeout'; env still released."""

    async def slow_fine_tune(samples, epochs=5):
        await asyncio.sleep(3.0)
        return 0.0

    stub = _stub_world_model(fine_tune=slow_fine_tune)
    monkeypatch.setattr(exploration, "AGENT_TIMEOUT_SECONDS", 0.5)

    agent = ExplorationAgent("exp_timeout", stub, "lower_left", 0.5)
    agent._client = None

    output = await agent.run()

    assert output["success"] is False
    assert output["error_after"] == output["error_before"]
    assert output["samples_collected"] == 0

    failed = next(
        e for e in get_events_since(0)
        if e["event_type"] == "agent_failed" and e["agent_id"] == "exp_timeout"
    )
    assert failed["payload"]["error"] == "timeout"
    assert env_spy["close_calls"] == 1, "timeout path must release the sim"
