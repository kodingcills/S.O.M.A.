from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pytest

import backend.rerun_logger as rerun_logger


@dataclass(slots=True)  # noqa: MUTABLE_OK
class LoggedEntity:
    kind: str
    data: np.ndarray
    kwargs: dict[str, float | list[float] | list[list[int]] | list[str]]


@dataclass(slots=True)  # noqa: MUTABLE_OK
class FakeRerun:
    init_calls: list[str] = field(default_factory=list)
    grpc_calls: list[int] = field(default_factory=list)
    viewer_calls: list[tuple[int, bool, str]] = field(default_factory=list)
    times: list[tuple[str, int]] = field(default_factory=list)
    logs: list[tuple[str, LoggedEntity]] = field(default_factory=list)

    def init(self, application_id: str) -> None:
        self.init_calls.append(application_id)

    def serve_grpc(self, *, grpc_port: int) -> str:
        self.grpc_calls.append(grpc_port)
        return "rerun+http://127.0.0.1:9876/proxy"

    def serve_web_viewer(
        self, *, web_port: int, open_browser: bool, connect_to: str
    ) -> None:
        self.viewer_calls.append((web_port, open_browser, connect_to))

    def set_time(self, timeline: str, *, sequence: int) -> None:
        self.times.append((timeline, sequence))

    def DepthImage(self, data: np.ndarray, **kwargs) -> LoggedEntity:
        return LoggedEntity("DepthImage", np.asarray(data), kwargs)

    def Points3D(self, data: np.ndarray, **kwargs) -> LoggedEntity:
        return LoggedEntity("Points3D", np.asarray(data), kwargs)

    def log(self, path: str, entity: LoggedEntity) -> None:
        self.logs.append((path, entity))


@pytest.fixture
def fake_rerun(monkeypatch: pytest.MonkeyPatch) -> FakeRerun:
    fake = FakeRerun()
    monkeypatch.setattr(rerun_logger, "rr", fake)
    monkeypatch.setattr(rerun_logger, "_ready", False)
    return fake


def _state_vector() -> np.ndarray:
    state = np.zeros(806, dtype=np.float32)
    state[:256] = np.arange(256, dtype=np.float32) / 255.0
    state[512:514] = 1.0
    state[768:771] = (0.2, 0.3, 0.4)
    state[776:778] = (0.8, 0.6)
    return state


def test_init_serves_current_grpc_and_web_viewer_once(fake_rerun: FakeRerun) -> None:
    # Given: an uninitialized logger backed by a no-network fake.
    # When: initialization is requested twice.
    rerun_logger.init()
    rerun_logger.init()

    # Then: only the current Rerun server APIs are called once.
    assert fake_rerun.init_calls == ["soma"]
    assert fake_rerun.grpc_calls == [9876]
    assert fake_rerun.viewer_calls == [
        (9090, False, "rerun+http://127.0.0.1:9876/proxy")
    ]


def test_log_step_is_noop_before_init(fake_rerun: FakeRerun) -> None:
    # Given: initialization has not run.
    # When: a simulation step is offered.
    rerun_logger.log_step("primary", _state_vector(), 3, [])

    # Then: neither timeline nor entities are emitted.
    assert fake_rerun.times == []
    assert fake_rerun.logs == []


def test_log_step_emits_exact_primary_entities(fake_rerun: FakeRerun) -> None:
    # Given: initialized logging, two vessels, and a primary error map.
    rerun_logger.init()
    vessels: list[rerun_logger.VesselLog] = [
        {"col": 3.0, "row": 6.0, "radius": 1.5, "damaged": False},
        {"col": 12.0, "row": 9.0, "radius": 2.0, "damaged": True},
    ]
    error_map = np.full((16, 16), 0.25, dtype=np.float32)

    # When: the post-step state is logged.
    rerun_logger.log_step("primary", _state_vector(), 7, vessels, error_map)

    # Then: the current sequence timeline and exact entity paths are used.
    assert fake_rerun.times == [("step", 7)]
    by_path = dict(fake_rerun.logs)
    assert set(by_path) == {
        "soma/primary/tissue/integrity",
        "soma/primary/tissue/bleeding",
        "soma/primary/robot/ee",
        "soma/primary/target",
        "soma/primary/vessels",
        "soma/primary/world_model/error",
    }
    integrity = by_path["soma/primary/tissue/integrity"]
    assert integrity.kind == "DepthImage"
    assert integrity.data.shape == (16, 16)
    assert integrity.kwargs["meter"] == 1.0
    np.testing.assert_array_equal(
        by_path["soma/primary/tissue/bleeding"].data,
        _state_vector()[512:768].reshape(16, 16),
    )
    ee = by_path["soma/primary/robot/ee"]
    np.testing.assert_allclose(ee.data, [[0.2, 0.3, 0.4]])
    assert ee.kwargs["colors"] == [[0, 180, 255]]
    assert ee.kwargs["radii"] == [0.03]
    assert ee.kwargs["labels"] == ["EE (primary)"]
    target = by_path["soma/primary/target"]
    np.testing.assert_allclose(target.data, [[0.8, 0.6, 0.0]])
    assert target.kwargs["colors"] == [[50, 220, 80]]
    assert target.kwargs["radii"] == [0.05]
    vessel_entity = by_path["soma/primary/vessels"]
    np.testing.assert_allclose(vessel_entity.data, [[0.2, 0.4, 0.0], [0.8, 0.6, 0.0]])
    assert vessel_entity.kwargs["colors"] == [[80, 140, 255], [220, 50, 50]]
    assert vessel_entity.kwargs["radii"] == [0.04]
    np.testing.assert_array_equal(
        by_path["soma/primary/world_model/error"].data, error_map
    )


def test_comparison_omits_world_model_error(fake_rerun: FakeRerun) -> None:
    # Given: an initialized logger and comparison state.
    rerun_logger.init()

    # When: comparison logs without an error map.
    rerun_logger.log_step("comparison", _state_vector(), 1, [])

    # Then: neither WorldModel nor an empty vessels entity is emitted.
    paths = [path for path, _entity in fake_rerun.logs]
    assert "soma/comparison/world_model/error" not in paths
    assert "soma/comparison/vessels" not in paths


def test_logger_and_reactive_import_guards() -> None:
    # Given: source files that must stay isolated from WorldModel internals.
    forbidden = ("world_model", "prediction_net", "belief_state", "orchestrator")
    logger_source = Path(rerun_logger.__file__).read_text()
    reactive_source = (
        Path(__file__).resolve().parents[1] / "backend" / "agents" / "reactive.py"
    ).read_text()

    # When/Then: neither logger nor comparison policy imports forbidden modules.
    for module_name in forbidden:
        assert f"import backend.{module_name}" not in logger_source
        assert f"from backend.{module_name}" not in logger_source
        assert f"import backend.{module_name}" not in reactive_source
        assert f"from backend.{module_name}" not in reactive_source
