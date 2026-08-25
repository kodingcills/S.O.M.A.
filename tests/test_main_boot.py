from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from backend import api, main


MAIN_PATH = Path(main.__file__)


class FakeBelief:
    def __init__(self) -> None:
        self.world_model_version = 0
        self.active_dof = 1
        self.prediction_error_map = np.zeros((16, 16), dtype=np.float32)

    def get_regional_errors(self) -> dict[str, float]:
        return {}


class FakeWorldModel:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver
        self.belief = FakeBelief()


class FakeDriver:
    instances: list[FakeDriver] = []

    def __init__(self, agent: SimpleNamespace, seed: int) -> None:
        self.agent = agent
        self.seed = seed
        self.step_count = 0
        self.episode_generation = 1
        self._observation = {
            "map": np.zeros((99, 4), dtype=np.int32),
            "stats": np.zeros(47, dtype=np.float32),
            "direction": np.zeros(1, dtype=np.int32),
        }
        self.instances.append(self)

    def current_obs_tokens(self) -> dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self._observation.items()}

    def reset(self, seed: int | None = None) -> None:
        self.episode_generation += 1
        if seed is not None:
            self.seed = seed


def test_main_has_one_world_model_construction_and_no_surrol_bootstrap() -> None:
    # Given: the production startup module's syntax tree.
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))

    # When: imports and constructor calls are collected structurally.
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    world_model_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "WorldModel"
    ]

    # Then: startup owns one singleton and no retired substrate dependency.
    assert len(world_model_calls) == 1
    assert not imports.intersection(
        {
            "backend.simulation",
            "backend.data_collector",
            "backend.prediction_net",
            "backend.train",
            "simulation",
            "data_collector",
            "prediction_net",
            "train",
        }
    )


def test_lifespan_registers_seeded_independent_craftax_drivers(monkeypatch) -> None:
    # Given: an instant released-agent load and observable startup boundaries.
    FakeDriver.instances.clear()
    agent = SimpleNamespace(name="released-simulus")
    load_calls: list[None] = []
    events: list[str] = []
    graph_inputs: list[tuple[FakeWorldModel, SimpleNamespace]] = []

    def load_once() -> tuple[SimpleNamespace, None, None]:
        load_calls.append(None)
        return agent, None, None

    def capture_graph(
        world_model: FakeWorldModel, drivers: SimpleNamespace
    ) -> SimpleNamespace:
        graph_inputs.append((world_model, drivers))
        return SimpleNamespace()

    monkeypatch.setattr(main, "load_simulus_agent", load_once)
    monkeypatch.setattr(main, "CraftaxDriver", FakeDriver)
    monkeypatch.setattr(main, "WorldModel", FakeWorldModel)
    monkeypatch.setattr(main, "build_soma_graph", capture_graph)
    monkeypatch.setattr(main, "init_rerun", lambda: None)
    monkeypatch.setattr(main, "write_event", lambda event_type, **_: events.append(event_type))

    # When: the real FastAPI lifespan boots and serves health.
    with TestClient(main.app) as client:
        response = client.get("/health")
        primary = api._env_registry["primary"]
        comparison = api._env_registry["comparison"]
        registered_world_model = api._world_model_ref[0]

        # Then: one load owns two independent, identically initialized drivers.
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert load_calls == [None]
        assert [driver.seed for driver in FakeDriver.instances] == [42, 42]
        assert primary is not comparison
        for name, values in primary.current_obs_tokens().items():
            np.testing.assert_array_equal(values, comparison.current_obs_tokens()[name])
        primary.reset()
        assert primary.episode_generation == 2
        assert comparison.episode_generation == 1
        assert registered_world_model.driver is primary
        assert graph_inputs == [
            (registered_world_model, SimpleNamespace(primary=primary, comparison=comparison))
        ]
        assert "system_ready" in events
