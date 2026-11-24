"""撮影対象条件によるセッションDBの範囲制限を検証する。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import capture_workflow
from sam3_pipeline.domain.capture import CaptureError
from sam3_pipeline.infrastructure.annotation_ledger import SqliteAnnotationLedger
from sam3_pipeline.infrastructure.annotation_store import FileAnnotationStore
from sam3_pipeline.infrastructure.capture_source import SqliteCaptureSource
from sam3_pipeline.infrastructure.object_review_store import SqliteObjectReviewStore
from sam3_pipeline.infrastructure.prediction_store import SqlitePredictionStore

EXPECTED_DATABASE_COUNT = 3
BUNDLE_SCHEMA = """
CREATE TABLE declarations(declaration_id TEXT, items_json TEXT);
CREATE TABLE images(session_id TEXT, image_id TEXT, declaration_id TEXT, arrangement_no INTEGER,
path TEXT, sha256 TEXT, width INTEGER, height INTEGER, tag_revision INTEGER, tag_status TEXT, depth_path TEXT);
CREATE TABLE hints(image_id TEXT, instrument_uid TEXT, instance_no INTEGER, visible INTEGER,
hint_source TEXT, center_json TEXT, end_a_json TEXT, end_b_json TEXT, occluded INTEGER);
"""


def _create_bundle(root: Path, capture_date: str, session_id: str) -> Path:
    """Create one tagged image and its session database for scope tests."""
    bundle = root / capture_date / session_id
    image_path = bundle / "raw" / "image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(image_path)
    relative_path = image_path.relative_to(root).as_posix()
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    with closing(sqlite3.connect(bundle / "metadata.sqlite3")) as connection, connection:
        connection.executescript(BUNDLE_SCHEMA)
        connection.execute("INSERT INTO declarations VALUES (?, ?)", ("decl", json.dumps([])))
        connection.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, "image", "decl", 1, relative_path, digest, 8, 8, 1, "tagged", None),
        )
    return image_path


def test_target_conditions_use_or_and_deduplicate_matching_databases(tmp_path: Path) -> None:
    """Resolve all OR conditions while returning a database matched more than once once."""
    first = _create_bundle(tmp_path, "2026-10-23", "session_a")
    second = _create_bundle(tmp_path, "2026-10-23", "session_b")
    third = _create_bundle(tmp_path, "2026-10-24", "session_c")
    source = SqliteCaptureSource(
        tmp_path,
        timeout=1.0,
        targets=[
            {"date": "2026-10-23"},
            {"date": "2026-10-23", "session_id": "session_a"},
            {"date": "2026-10-24", "session_id": "session_c"},
        ],
    )

    samples = list(source.iter_samples())

    assert [sample.image_path for sample in samples] == [first.resolve(), second.resolve(), third.resolve()]
    assert len(source.catalog()) == EXPECTED_DATABASE_COUNT
    assert len(source.databases()) == EXPECTED_DATABASE_COUNT


@pytest.mark.parametrize(
    "targets",
    [
        [],
        [{"session_id": "session_a"}],
        [{"date": "2026-02-30"}],
        [{"date": "2026-10-23", "session_id": "../outside"}],
        [{"date": "2026-10-23", "session_id": None}],
        [{"date": "2026-10-23", "extra": "value"}],
    ],
)
def test_invalid_target_conditions_are_rejected(tmp_path: Path, targets: Any) -> None:
    """Reject malformed target entries rather than widening the capture scope."""
    with pytest.raises(CaptureError, match="targets"):
        SqliteCaptureSource(tmp_path, timeout=1.0, targets=targets)


def test_each_target_condition_must_match_an_existing_database(tmp_path: Path) -> None:
    """Fail if one requested session is missing, even when another condition matches."""
    _create_bundle(tmp_path, "2026-10-23", "session_a")
    source = SqliteCaptureSource(
        tmp_path,
        timeout=1.0,
        targets=[{"date": "2026-10-23"}, {"date": "2026-10-23", "session_id": "session_missing"}],
    )

    with pytest.raises(CaptureError, match="session_missing"):
        source.catalog()


def test_capture_workflow_applies_configured_target_to_review_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass capture.json targets through store setup and review source enumeration."""
    _create_bundle(tmp_path, "2026-10-23", "session_selected")
    outside_image = _create_bundle(tmp_path, "2026-10-24", "session_outside")
    outside_database = outside_image.parents[1] / "metadata.sqlite3"
    outside_contents = outside_database.read_bytes()
    config_path = tmp_path / "capture.json"
    config_path.write_text(
        json.dumps(
            {
                "capture_root": str(tmp_path),
                "output_root": str(tmp_path.parent / f"{tmp_path.name}_output"),
                "targets": [{"date": "2026-10-23", "session_id": "session_selected"}],
                "matching": {"max_distance": 0.03, "ambiguity_margin": 0.05},
                "review_confidence": {"medium_score": 0.65, "high_score": 0.85},
            }
        ),
        encoding="utf-8",
    )
    reviewed: dict[str, Any] = {}
    monkeypatch.setattr(
        capture_workflow.BatchReviewServer,
        "serve",
        lambda server: reviewed.update({"samples": server.samples}),
    )

    capture_workflow.main(["--config", str(config_path), "review"])

    assert [sample.session_id for sample in reviewed["samples"].values()] == ["session_selected"]
    assert outside_database.read_bytes() == outside_contents


def test_currentness_checks_do_not_open_an_out_of_scope_database(tmp_path: Path) -> None:
    """Keep SQL and image currentness checks inside the target scope, including store calls."""
    _create_bundle(tmp_path, "2026-10-23", "session_selected")
    outside_image = _create_bundle(tmp_path, "2026-10-24", "session_outside")
    unrestricted_source = SqliteCaptureSource(tmp_path, timeout=1.0)
    outside_sample = next(sample for sample in unrestricted_source.iter_samples() if sample.session_id == "session_outside")
    outside_database = outside_image.parents[1] / "metadata.sqlite3"
    outside_database.write_bytes(b"not a SQLite database")
    source = SqliteCaptureSource(
        tmp_path,
        timeout=1.0,
        targets=[{"date": "2026-10-23", "session_id": "session_selected"}],
    )
    selected_databases = source.databases()
    predictions = SqlitePredictionStore(tmp_path, tmp_path, tmp_path / "masks", 1.0)
    predictions.initialize(selected_databases)
    reviews = SqliteObjectReviewStore(tmp_path, 1.0, predictions)
    reviews.initialize(selected_databases)
    store = FileAnnotationStore(tmp_path / "output", source, review_store=reviews)
    source_record = store._sample_record(outside_sample)

    assert [sample.session_id for sample in source.iter_samples()] == ["session_selected"]
    assert [item["session_id"] for item in source.catalog()] == ["session_selected"]
    assert source.current_sample("session_outside", "image", outside_sample.source_path) is None
    assert not source.is_current(outside_sample)
    assert reviews.progress() == {}
    with pytest.raises(CaptureError, match="一意"), predictions._connection(outside_sample):
        pytest.fail("out-of-scope prediction connection was opened")
    with pytest.raises(CaptureError), reviews._connection(sample=outside_sample):
        pytest.fail("out-of-scope review connection was opened")
    assert not store._source_is_current(source_record)
    assert not store.is_current_snapshot(source_record, outside_sample)
    with pytest.raises(CaptureError, match="no longer present"):
        store.current_sample(source_record)


def test_same_session_id_uses_source_date_for_review_and_ledger_database(tmp_path: Path) -> None:
    """Resolve same-named sessions by source path without touching the other date."""
    selected_image = _create_bundle(tmp_path, "2026-10-23", "session_reused")
    other_image = _create_bundle(tmp_path, "2026-10-24", "session_reused")
    source = SqliteCaptureSource(
        tmp_path,
        timeout=1.0,
        targets=[{"date": "2026-10-23", "session_id": "session_reused"}],
    )
    selected_database = selected_image.parents[1] / "metadata.sqlite3"
    other_database = other_image.parents[1] / "metadata.sqlite3"
    predictions = SqlitePredictionStore(tmp_path, tmp_path, tmp_path / "masks", 1.0)
    predictions.initialize(source.databases())
    reviews = SqliteObjectReviewStore(tmp_path, 1.0, predictions)
    reviews.initialize(source.databases())
    ledger = SqliteAnnotationLedger(tmp_path, 1.0, source.databases())

    assert reviews.catalog_state(
        "session_reused",
        "image",
        "missing-revision",
        "2026-10-23/session_reused/raw/image.png",
    ) is None
    ledger.upsert_annotation_revision(
        session_id="session_reused",
        image_id="image",
        tag_revision=1,
        revision="test-revision",
        status="reviewed",
        annotation_path="reviews/test/annotation.json",
        annotation_sha256="a" * 64,
        source_path="2026-10-23/session_reused/raw/image.png",
    )

    with closing(sqlite3.connect(selected_database)) as connection:
        selected_count = connection.execute(
            "SELECT COUNT(*) FROM annotation_revisions WHERE revision='test-revision'"
        ).fetchone()[0]
    with closing(sqlite3.connect(other_database)) as connection:
        other_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='annotation_revisions'"
        ).fetchone()

    assert selected_count == 1
    assert other_table is None


def test_current_sample_resolves_reused_image_id_by_source_path(tmp_path: Path) -> None:
    """Resolve the selected image from its capture date when IDs are reused."""
    first_image = _create_bundle(tmp_path, "2026-10-23", "session_reused")
    second_image = _create_bundle(tmp_path, "2026-10-24", "session_reused")
    source = SqliteCaptureSource(tmp_path, timeout=1.0)

    first_sample = source.current_sample("session_reused", "image", "2026-10-23/session_reused/raw/image.png")
    second_sample = source.current_sample("session_reused", "image", "2026-10-24/session_reused/raw/image.png")

    assert first_sample is not None
    assert second_sample is not None
    assert first_sample.image_path == first_image.resolve()
    assert second_sample.image_path == second_image.resolve()
