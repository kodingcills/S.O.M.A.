"""SOMA Task 1.7a — ExplorationAgent named invariant tests.

Zero network: Claude is mocked via fake stream/client objects; degraded mode
(_client=None) exercises the random-focused fallback path.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.agents import exploration
from backend.agents.exploration import (
    AgentAPIError,
    AgentFeedBuffer,
    ExplorationAgent,
)
from backend.belief_state import BeliefState
from backend.event_log import get_events_since, init_db, write_event
from backend.prediction_net import PredictedState
from backend.simulation import SimConfig, create_env

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_state_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = np.zeros(806, dtype=np.float32)
    v[0:256] = rng.uniform(0.3, 1.0, 256)
    v[256:512] = rng.uniform(0.0, 1.0, 256)
    v[768:771] = [0.5, 0.5, 0.2]
    v[799] = 1.0  # DOF 1 active
    return v


def _make_stub_world_model(events_recorder: list | None = None):
    belief = BeliefState()
    belief.update_from_observation(_make_state_vec())

    async def fine_tune(samples, epochs=5):
        if events_recorder is not None:
            events_recorder.append("world_model_updated")
        try:
            write_event("world_model_updated", payload_loss=0.01)
        except Exception:
            pass
        return 0.01

    def predict(state_vec, action_vec):
        return PredictedState(
            tissue=np.full((16, 16), 0.9, dtype=np.float32),
            damage_prob=0.1,
            vessel_risk=np.zeros(5, dtype=np.float32),
            pred_ee_norm=np.array([0.5, 0.5, 0.2, 0.0], dtype=np.float32),
        )

    return SimpleNamespace(
        belief=belief,
        fine_tune=fine_tune,
        network=SimpleNamespace(predict=predict),
    )


class _FakeStream:
    def __init__(self, final_message):
        self._final = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        class _Empty:
            def __aiter__(self_inner):
                return self_inner

            async def __aniter__(self_inner):
                return self_inner

            def __aiter__(self_inner):  # noqa: F811 — single protocol
                return self_inner

        async def _gen():
            if False:
                yield ""

        return _gen()

    async def get_final_message(self):
        return self._final


class _FakeMessages:
    def __init__(self, final_message=None, error: Exception | None = None):
        self._final = final_message
        self._error = error
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _FakeStream(self._final)


def _tool_use_final(tool_input: dict):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input=tool_input)],
        usage=SimpleNamespace(input_tokens=50, output_tokens=20),
    )


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_DB_PATH", str(tmp_path / "events.db"))
    init_db()
    yield tmp_path


@pytest.fixture
def fast_agent(monkeypatch):
    """Shrink episode + measurement cost so the suite stays <90s."""
    monkeypatch.setattr(exploration, "EXPLORE_N_STEPS", 10)
    monkeypatch.setattr(exploration, "MEASURE_CYCLES", 2)


async def _capturing_create_env(captured: list[SimConfig]):
    real_create_env = exploration.create_env

    async def wrapper(config):
        captured.append(config)
        env = await real_create_env(config)
        close_calls.append(0)

        original_close = env.close

        def counting_close():
            close_calls[-1] += 1
            return original_close()

        env.close = counting_close
        return env

    return wrapper


close_calls: list[int] = []


# ---------------------------------------------------------------------------
# test:agents:spawned_before_loop (C4)
# ---------------------------------------------------------------------------

async def test_spawned_before_loop(isolated_db, fast_agent, monkeypatch):
    captured_configs: list[SimConfig] = []
    wrapper = await _capturing_create_env(captured_configs)
    monkeypatch.setattr(exploration, "create_env", wrapper)

    write_event(
        "agent_spawned",
        agent_id="exp_test01",
        parent_id="orchestrator",
        region="upper_left",
        error_before=0.8,
    )

    stub = _make_stub_world_model()
    agent = ExplorationAgent("exp_test01", stub, "upper_left", 0.8)
    agent._client = None  # degraded mode — no network
    agent.env = await exploration.create_env(SimConfig(seed=42, dof=1, n_vessels=2))

    try:
        await agent.run_episode(n_steps=12)
    finally:
        await asyncio.to_thread(agent.env.close)

    events = get_events_since(0)
    spawned_ts = next(
        e["timestamp"] for e in events if e["event_type"] == "agent_spawned"
    )
    step_events = [e for e in events if e["event_type"] == "agent_step"]
    assert step_events, "episode must emit agent_step telemetry"
    first_step_ts = step_events[0]["timestamp"]
    assert spawned_ts < first_step_ts, "C4 violated: spawn must precede steps"

    assert captured_configs[0].seed == 42, "SEED INVARIANT: exploration uses seed 42"
    assert captured_configs[0].dof == int(stub.belief.active_dof)


# ---------------------------------------------------------------------------
# test:agents:completed_after_finetune (C5)
# ---------------------------------------------------------------------------

async def test_completed_after_finetune(isolated_db, fast_agent, monkeypatch):
    recorded: list[str] = []
    stub = _make_stub_world_model(events_recorder=recorded)

    captured_configs: list[SimConfig] = []
    wrapper = await _capturing_create_env(captured_configs)
    monkeypatch.setattr(exploration, "create_env", wrapper)

    agent = ExplorationAgent("exp_c5test", stub, "upper_left", 0.8)
    agent._client = None

    output = await agent.run()

    assert "world_model_updated" in recorded, "fine_tune must run"
    events = get_events_since(0)
    wm_ts = next(
        e["timestamp"] for e in events if e["event_type"] == "world_model_updated"
    )
    completed = next(e for e in events if e["event_type"] == "agent_completed")
    assert wm_ts < completed["timestamp"], "C5 violated: completed before fine-tune"

    assert completed["error_before"] == pytest.approx(0.8)
    assert isinstance(completed["error_after"], float)
    # samples_collected / fine_tune_loss live in payload (event_log column law)
    assert completed["payload"]["samples_collected"] > 0
    assert set(output.keys()) >= {
        "agent_id", "region", "error_before", "error_after",
        "samples_collected", "fine_tune_loss", "api_tokens_used", "success",
    }
    assert close_calls[-1] >= 1, "SIMULATION RELEASED INVARIANT"


# ---------------------------------------------------------------------------
# test:agents:sim_released
# ---------------------------------------------------------------------------

async def test_sim_released(isolated_db, fast_agent, monkeypatch):
    captured_configs: list[SimConfig] = []
    wrapper = await _capturing_create_env(captured_configs)
    monkeypatch.setattr(exploration, "create_env", wrapper)

    agent = ExplorationAgent("exp_close01", _make_stub_world_model(), "lower_right", 0.6)
    agent._client = None
    await agent.run()
    assert close_calls[-1] >= 1, "env.close() must be called on success path"


def test_reactive_isolation():
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

async def test_api_fallback_on_error(isolated_db, fast_agent, monkeypatch, tmp_path):
    import anthropic as anthropic_mod

    request_stub = SimpleNamespace()
    broken = _FakeMessages(error=anthropic_mod.APIConnectionError(request=request_stub))

    captured_configs: list[SimConfig] = []
    wrapper = await _capturing_create_env(captured_configs)
    monkeypatch.setattr(exploration, "create_env", wrapper)
    # Keep retries from stalling the suite.
    monkeypatch.setattr(exploration.asyncio, "sleep", async_noop_sleep)

    agent = ExplorationAgent("exp_fbtest", _make_stub_world_model(), "upper_left", 0.7)
    agent._client = SimpleNamespace(messages=broken)
    agent.env = await exploration.create_env(SimConfig(seed=42, dof=1, n_vessels=2))

    try:
        samples, tokens = await agent.run_episode(n_steps=6)
    finally:
        await asyncio.to_thread(agent.env.close)

    assert len(samples) == 6, "fallback path must still collect data"
    assert tokens == 0, "no successful API calls — zero token spend"

    feed_events = [
        e for e in get_events_since(0)
        if e["event_type"] == "agent_step" and "reasoning" in (e.get("payload") or {})
    ]
    assert any("API fallback" in str(e["payload"]["reasoning"]) for e in feed_events)


async def async_noop_sleep(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# test:agents:tool_choice_any
# ---------------------------------------------------------------------------

async def test_tool_choice_forces_structured_output(isolated_db, monkeypatch):
    text_only = _FakeMessages(
        final_message=SimpleNamespace(
            content=[SimpleNamespace(type="text")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=10),
        )
    )
    monkeypatch.setattr(exploration.asyncio, "sleep", async_noop_sleep)

    client = SimpleNamespace(messages=text_only)
    with pytest.raises(AgentAPIError, match="select_action"):
        await exploration._call_claude(client, "prompt", lambda tok: None)

    call_kwargs = text_only.calls[0]
    assert call_kwargs["tool_choice"] == {"type": "any"}, (
        "A-1 regression guard: tool_choice must force structured tool calls"
    )


# ---------------------------------------------------------------------------
# test:agents:agent_timeout
# ---------------------------------------------------------------------------

async def test_agent_timeout(isolated_db, fast_agent, monkeypatch):
    async def slow_fine_tune(samples, epochs=5):
        await asyncio.sleep(3.0)
        return 0.0

    stub = _make_stub_world_model()
    stub.fine_tune = slow_fine_tune

    captured_configs: list[SimConfig] = []
    wrapper = await _capturing_create_env(captured_configs)
    monkeypatch.setattr(exploration, "create_env", wrapper)
    monkeypatch.setattr(exploration, "AGENT_TIMEOUT_SECONDS", 0.5)

    agent = ExplorationAgent("exp_timeout", stub, "lower_left", 0.5)
    agent._client = None

    output = await agent.run()

    assert output["success"] is False
    failed = next(
        e for e in get_events_since(0) if e["event_type"] == "agent_failed"
    )
    assert failed["payload"]["error"] == "timeout"
