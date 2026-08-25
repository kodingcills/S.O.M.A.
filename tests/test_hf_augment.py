from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from backend.data.hf_augment import (
    CANDIDATES,
    ingest_candidates,
    map_action,
    map_state,
    transform_row,
)


class UpstreamDatasetError(RuntimeError):
    pass


def _row(
    state: Sequence[float] = (0.0,), action: Sequence[float] = (0.0,)
) -> Mapping[str, Sequence[float]]:
    return {"observation.state": state, "action": action}


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def test_maps_state_and_action_when_source_vectors_exceed_ranges() -> None:
    # Given
    source_state = [np.pi, -2 * np.pi, np.pi / 2, *range(37)]
    source_action = [2.0, -2.0, 0.25, *range(10)]

    # When
    state = map_state(source_state)
    action = map_action(source_action)

    # Then
    assert state.shape == (806,)
    assert state.dtype == np.float32
    assert np.count_nonzero(state[:768]) == 0
    np.testing.assert_allclose(state[768:771], [1.0, -1.0, 0.5])
    assert state[805] == pytest.approx(np.clip(34 / np.pi, -1, 1))
    assert action.shape == (12,)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action[:3], [1.0, -1.0, 0.25])
    assert action[-1] == 1.0


def test_copies_next_state_when_transforming_a_valid_row() -> None:
    # Given
    row = _row(state=(np.pi,), action=(0.5,))

    # When
    sample = transform_row(row)
    sample.state[768] = 0.0

    # Then
    assert sample.next_state[768] == 1.0
    assert not np.shares_memory(sample.state, sample.next_state)


def test_scans_exact_candidates_and_skips_sampled_action_dimension_over_12(
    tmp_path: Path,
) -> None:
    # Given
    calls: list[tuple[str, str, bool, str]] = []

    def loader(
        name: str, *, split: str, streaming: bool, token: str
    ) -> Iterable[Mapping[str, Sequence[float]]]:
        calls.append((name, split, streaming, token))
        action_dim = 14 if name.startswith("lerobot/aloha") else 2
        return [_row(action=[0.0] * action_dim)]

    # When
    with closing(_connect(tmp_path / "training.db")) as connection:
        receipts = ingest_candidates(connection, loader, token="secret-token")

    # Then
    assert tuple(call[0] for call in calls) == CANDIDATES
    assert all(call[1:] == ("train", True, "secret-token") for call in calls)
    assert [receipt.compatible for receipt in receipts] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert [receipt.ingested for receipt in receipts] == [0, 0, 0, 1, 1, 1]


def test_rejects_malformed_and_non_finite_rows_without_hiding_programmer_errors(
    tmp_path: Path,
) -> None:
    # Given
    malformed_rows: list[Mapping[str, Sequence[float]]] = [
        {"observation.state": [0.0]},
        _row(state=[[1.0]]),
        _row(action=[np.nan]),
        _row(state=[np.inf]),
        _row(state=[0.5], action=[0.25]),
    ]

    def loader(
        name: str, *, split: str, streaming: bool, token: str
    ) -> Iterable[Mapping[str, Sequence[float]]]:
        return malformed_rows if name == CANDIDATES[0] else [_row(action=[0.0] * 14)]

    # When
    with closing(_connect(tmp_path / "training.db")) as connection:
        receipts = ingest_candidates(connection, loader, token="token")
        count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    # Then
    assert receipts[0].rejected == 4
    assert receipts[0].ingested == 1
    assert count == 1


def test_continues_after_a_candidate_loader_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given
    def loader(
        name: str, *, split: str, streaming: bool, token: str
    ) -> Iterable[Mapping[str, Sequence[float]]]:
        if name == CANDIDATES[0]:
            raise UpstreamDatasetError("upstream rejected secret-token")
        return [_row(state=(np.pi,), action=(0.5,))]

    # When
    with closing(_connect(tmp_path / "training.db")) as connection:
        receipts = ingest_candidates(connection, loader, token="secret-token")
        count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    output = capsys.readouterr().out

    # Then
    failed = receipts[0]
    assert not failed.compatible
    assert failed.ingested == 0
    assert failed.rejected == 0
    assert "secret-token" not in failed.reason
    assert CANDIDATES[0] in output
    assert "secret-token" not in output
    assert receipts[1].compatible
    assert receipts[1].ingested == 1
    assert count == len(CANDIDATES) - 1


def test_returns_failed_receipt_when_stream_fails_before_first_valid_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given
    def loader(
        name: str, *, split: str, streaming: bool, token: str
    ) -> Iterable[Mapping[str, Sequence[float]]]:
        if name == CANDIDATES[0]:

            def rows() -> Iterable[Mapping[str, Sequence[float]]]:
                raise UpstreamDatasetError("stream died holding secret-token")
                yield _row()

            return rows()
        return [_row(state=(np.pi,), action=(0.5,))]

    # When
    with closing(_connect(tmp_path / "training.db")) as connection:
        receipts = ingest_candidates(connection, loader, token="secret-token")
        count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    output = capsys.readouterr().out

    # Then
    failed = receipts[0]
    assert not failed.compatible
    assert failed.ingested == 0
    assert failed.rejected == 0
    assert failed.action_dim is None
    assert failed.reason == "dataset stream failed (UpstreamDatasetError)"
    assert "secret-token" not in failed.reason
    assert CANDIDATES[0] in output
    assert "UpstreamDatasetError" in output
    assert "secret-token" not in output
    assert all(r.compatible and r.ingested == 1 for r in receipts[1:])
    assert count == len(CANDIDATES) - 1


def test_discards_partial_batch_when_stream_fails_mid_collection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given
    def loader(
        name: str, *, split: str, streaming: bool, token: str
    ) -> Iterable[Mapping[str, Sequence[float]]]:
        if name == CANDIDATES[1]:

            def rows() -> Iterable[Mapping[str, Sequence[float]]]:
                yield _row(state=(1.0,))
                yield _row(state=(2.0,))
                raise UpstreamDatasetError("truncated mid-stream secret-token")

            return rows()
        return [_row(state=(np.pi,), action=(0.5,))]

    # When
    with closing(_connect(tmp_path / "training.db")) as connection:
        receipts = ingest_candidates(connection, loader, token="secret-token")
        count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    output = capsys.readouterr().out

    # Then
    failed = receipts[1]
    assert not failed.compatible
    assert failed.ingested == 0
    assert failed.rejected == 0
    assert failed.action_dim == 1
    assert failed.reason == "dataset stream failed (UpstreamDatasetError)"
    assert "secret-token" not in failed.reason
    assert receipts[0].compatible
    assert receipts[0].ingested == 1
    assert all(r.compatible and r.ingested == 1 for r in receipts[2:])
    assert count == len(CANDIDATES) - 1
    assert CANDIDATES[1] in output
    assert "UpstreamDatasetError" in output
    assert "secret-token" not in output


def test_caps_each_compatible_dataset_at_5000_valid_rows(tmp_path: Path) -> None:
    # Given
    def loader(
        name: str, *, split: str, streaming: bool, token: str
    ) -> Iterable[Mapping[str, Sequence[float]]]:
        if name == CANDIDATES[0]:
            return (_row(state=(float(index),)) for index in range(5_001))
        return [_row(action=[0.0] * 14)]

    # When
    with closing(_connect(tmp_path / "training.db")) as connection:
        receipts = ingest_candidates(connection, loader, token="token")
        count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    # Then
    assert receipts[0].ingested == 5_000
    assert count == 5_000


def test_writes_exact_float32_blob_sizes_and_schema_defaults(tmp_path: Path) -> None:
    # Given
    def loader(
        name: str, *, split: str, streaming: bool, token: str
    ) -> Iterable[Mapping[str, Sequence[float]]]:
        return [_row(state=(np.pi,), action=(0.75,))]

    # When
    with closing(_connect(tmp_path / "training.db")) as connection:
        ingest_candidates(connection, loader, token="token")
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        rows = connection.execute(
            "SELECT length(state), length(action), length(next_state), "
            "reward, done, episode_step FROM samples"
        ).fetchall()

    # Then
    assert journal_mode == "wal"
    assert rows == [(3224, 48, 3224, 0.0, 0, 0)] * len(CANDIDATES)
