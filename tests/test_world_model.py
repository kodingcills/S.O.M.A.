from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Iterator

import anyio
import numpy as np
import pytest
import torch

from backend.belief_state import BeliefState
from backend.craftax_driver import CraftaxDriver
from backend.simulus import CHECKPOINT_SHA256, MODEL_REVISION
from backend.simulus.soma_bridge import (
    achievements_to_dof,
    jsd_to_error_map,
    jsd_to_regional_errors,
)
from backend.world_model import WorldModel
from backend.world_model_types import (
    DuplicateIngestError,
    OutOfOrderIngestError,
    ReadOnlyWorldModelError,
    StalePredictionError,
)


@pytest.fixture(scope="module")
def released_agent():
    from backend.simulus.runtime import load_simulus_agent

    agent, _config, _device = load_simulus_agent("cpu")
    return agent


@pytest.fixture
def driver(released_agent) -> Iterator[CraftaxDriver]:
    value = CraftaxDriver(released_agent, seed=42)
    yield value


def _state_digest(value) -> bytes:
    digest = hashlib.sha256()

    def feed(item) -> None:
        if item is None:
            digest.update(b"none")
        elif isinstance(item, torch.Tensor):
            digest.update(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, tuple):
            for child in item:
                feed(child)
        else:
            digest.update(str(getattr(item, "n", "")).encode())
            feed(getattr(item, "state", None))

    feed(value)
    return digest.digest()


def _parameter_digest(agent) -> bytes:
    digest = hashlib.sha256()
    for parameter in agent.parameters():
        if not parameter.is_meta:
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.digest()


async def _cycle(model: WorldModel):
    prediction = await model.predict()
    transition = await model.update(prediction)
    ingestion = await model.ingest(transition)
    return prediction, transition, ingestion


def test_constructor_uses_injected_driver_without_loading_or_training(driver):
    belief = BeliefState()
    model = WorldModel(driver, belief)

    assert model.belief is belief
    assert model.identity.model_revision == MODEL_REVISION
    assert model.identity.checkpoint_sha256 == CHECKPOINT_SHA256
    assert model.identity.adaptation_generation == belief.world_model_version == 0


@pytest.mark.anyio
async def test_predict_returns_43_real_deeply_immutable_actions(driver):
    result = await WorldModel(driver).predict()

    assert tuple(action.action_idx for action in result.actions) == tuple(range(43))
    assert len(result.controller_probabilities) == 43
    assert sum(result.controller_probabilities) == pytest.approx(1.0)
    assert all(np.isfinite(action.jsd) for action in result.actions)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        result.actions[0].jsd = 0.0
    with pytest.raises(TypeError):
        result.error_map[0][0] = 0.0


@pytest.mark.anyio
async def test_predict_uses_one_batched_evaluation(driver, monkeypatch):
    calls = 0
    original = driver.evaluate_stamped_candidates

    def counted():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(driver, "evaluate_stamped_candidates", counted)
    await WorldModel(driver).predict()
    assert calls == 1


@pytest.mark.anyio
async def test_predict_preserves_recurrent_and_actor_critic_state(driver):
    recurrent = _state_digest(driver._rs)
    actor = _state_digest(driver._agent.actor_critic.actor_state)
    critic = _state_digest(driver._agent.actor_critic.critic_state)

    await WorldModel(driver).predict()

    assert _state_digest(driver._rs) == recurrent
    assert _state_digest(driver._agent.actor_critic.actor_state) == actor
    assert _state_digest(driver._agent.actor_critic.critic_state) == critic


@pytest.mark.anyio
async def test_detection_cycle_leaves_all_parameters_unchanged(driver):
    before = _parameter_digest(driver._agent)
    await _cycle(WorldModel(driver))
    assert _parameter_digest(driver._agent) == before


@pytest.mark.anyio
async def test_update_matches_released_controller(driver):
    torch.manual_seed(9)
    expected_step, expected_probs = driver.step_controller()
    driver.reset(seed=42)
    torch.manual_seed(9)

    model = WorldModel(driver)
    transition = await model.update(await model.predict())

    assert transition.action == expected_step.action
    assert transition.reward == expected_step.reward
    assert transition.done == expected_step.done
    assert transition.controller_probabilities == pytest.approx(
        tuple(np.asarray(expected_probs).reshape(-1))
    )
    assert transition.selected_prediction is transition.prediction.actions[transition.action]


@pytest.mark.anyio
async def test_update_rejects_stale_step_and_reset_before_mutation(driver, monkeypatch):
    model = WorldModel(driver)
    stale_step = await model.predict()
    driver.step_with_action(0)
    step_before = driver.step_count
    with pytest.raises(StalePredictionError):
        await model.update(stale_step)
    assert driver.step_count == step_before

    driver.reset(seed=42)
    stale_episode = await model.predict()
    driver.reset(seed=42)
    with pytest.raises(StalePredictionError):
        await model.update(stale_episode)
    assert driver.step_count == 0

    current = driver.current_obs_tokens
    changed = current()
    changed["vector"] = changed["vector"].copy()
    changed["vector"][0] += 1.0
    monkeypatch.setattr(driver, "current_obs_tokens", lambda: changed)
    with pytest.raises(StalePredictionError):
        await model.update(stale_episode)
    assert driver.step_count == 0


@pytest.mark.anyio
async def test_ingest_matches_bridge_and_ema_exactly(driver, monkeypatch):
    monkeypatch.setattr("backend.world_model.write_event", lambda *_a, **_k: 17)
    raw = driver.evaluate_stamped_candidates()
    expected_map = jsd_to_error_map([item.output for item in raw])
    expected_regions = jsd_to_regional_errors([item.output for item in raw])
    model = WorldModel(driver)
    prediction, transition, result = await _cycle(model)
    expected = 0.3 * expected_map

    np.testing.assert_allclose(np.asarray(prediction.error_map), expected_map)
    assert dict(prediction.regional_errors) == expected_regions
    np.testing.assert_allclose(model.belief.prediction_error_map, expected)
    assert result.regional_errors == prediction.regional_errors
    assert model.belief.get_regional_errors() == dict(prediction.regional_errors)
    assert result.event_id == 17
    assert transition.reward_step == prediction.decision_point.prediction_step + 1


@pytest.mark.anyio
async def test_ingest_uses_real_achievements_for_active_dof(driver, monkeypatch):
    monkeypatch.setattr("backend.world_model.write_event", lambda *_a, **_k: 1)
    model = WorldModel(driver)
    _prediction, transition, result = await _cycle(model)
    assert result.active_dof == achievements_to_dof(np.asarray(transition.achievements))


@pytest.mark.anyio
async def test_duplicate_ingest_has_no_second_ema_or_event(driver, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        "backend.world_model.write_event", lambda event, **_kwargs: events.append(event) or 1
    )
    model = WorldModel(driver)
    transition = await model.update(await model.predict())
    await model.ingest(transition)
    first_map = model.belief.prediction_error_map.copy()

    with pytest.raises(DuplicateIngestError):
        await model.ingest(transition)
    np.testing.assert_array_equal(model.belief.prediction_error_map, first_map)
    assert events == ["belief_snapshot"]

    later = await model.update(await model.predict())
    await model.ingest(later)
    second_map = model.belief.prediction_error_map.copy()
    with pytest.raises(OutOfOrderIngestError):
        await model.ingest(transition)
    np.testing.assert_array_equal(model.belief.prediction_error_map, second_map)
    assert events == ["belief_snapshot", "belief_snapshot"]


@pytest.mark.anyio
async def test_version_remains_zero_across_detection_cycles(driver, monkeypatch):
    monkeypatch.setattr("backend.world_model.write_event", lambda *_a, **_k: 1)
    model = WorldModel(driver)
    await _cycle(model)
    await _cycle(model)
    assert model.identity.adaptation_generation == model.belief.world_model_version == 0


@pytest.mark.anyio
async def test_fine_tune_rejects_without_side_effects(driver, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        "backend.world_model.write_event", lambda event, **_kwargs: events.append(event) or 1
    )
    model = WorldModel(driver)
    before = _parameter_digest(driver._agent)

    with pytest.raises(ReadOnlyWorldModelError):
        await model.fine_tune(())
    assert _parameter_digest(driver._agent) == before
    assert model.belief.world_model_version == 0
    assert events == []


@pytest.mark.anyio
async def test_predict_and_update_do_not_block_event_loop(driver, monkeypatch):
    original_predict = driver.evaluate_stamped_candidates
    original_update = driver.step_controller
    monkeypatch.setattr(driver, "evaluate_stamped_candidates", lambda: (time.sleep(0.1), original_predict())[1])
    monkeypatch.setattr(driver, "step_controller", lambda: (time.sleep(0.1), original_update())[1])
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(12):
            await anyio.sleep(0.02)
            ticks += 1

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(ticker)
        await WorldModel(driver).update(await WorldModel(driver).predict())
    assert ticks >= 8


@pytest.mark.anyio
async def test_ingest_emits_belief_snapshot_only(driver, monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "backend.world_model.write_event", lambda event, **payload: events.append((event, payload)) or 3
    )
    await _cycle(WorldModel(driver))
    assert [event for event, _payload in events] == ["belief_snapshot"]
    assert events[0][1]["world_model_version"] == 0


def test_world_model_has_no_legacy_predictor_optimizer_or_replay(driver):
    model = WorldModel(driver)
    forbidden = {"network", "model_path", "predict_locked", "_optimizer", "_replay_buffer", "_finetune_lock"}
    assert forbidden.isdisjoint(set(dir(model)))
