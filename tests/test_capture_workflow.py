"""一時データで撮影入力・推論・レビュー・公開の契約を検証する。

責務:
    撮影DBと画像を一時領域に作り、実データや実モデルを変更せずに動作を確認する。
処理の流れ:
    1. セッションDB・画像・モデル代替を準備する。
    2. 入力アダプターとワークフローを実行する。
    3. 入出力・現行性・公開条件を照合する。
関連ファイル:
    capture_workflow.py、sam3_pipeline/infrastructure/、sam3_pipeline/usecase/capture.py。
設計意図:
    一時SQLiteとダミー推論で、実機・GPUなしに撮影から公開までの整合性を検証する。
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from capture_training.domain import SplitSettings, TrainingSettings
from capture_training.infrastructure.batches import ApprovedBatchSource
from capture_training.infrastructure.datasets import FrozenDatasetWriter
from capture_training.infrastructure.ledger import TrainingMetadataLedger
from capture_training.infrastructure.training import train_frozen_dataset
from capture_training.usecase import build_dataset
from PIL import Image

from capture_workflow import main
from sam3_pipeline.domain.annotation import Assignment, Candidate, MatchResult
from sam3_pipeline.domain.capture import CaptureError, CaptureSample, InstanceTag, TagPoint
from sam3_pipeline.domain.confidence import ConfidenceThresholds
from sam3_pipeline.infrastructure import capture_predictor
from sam3_pipeline.infrastructure.annotation_ledger import SqliteAnnotationLedger
from sam3_pipeline.infrastructure.annotation_store import ANNOTATION_FILE, REVIEWS_DIR, FileAnnotationStore, atomic_json, content_hash, digest_file, read_json
from sam3_pipeline.infrastructure.capture_source import SqliteCaptureSource
from sam3_pipeline.infrastructure.matching import ArrayMask, MatchSettings, ScipyInstanceMatcher
from sam3_pipeline.infrastructure.object_review_store import REVIEW_SCHEMA_VERSION, SqliteObjectReviewStore
from sam3_pipeline.infrastructure.prediction_store import SqlitePredictionStore
from sam3_pipeline.infrastructure.review_server import BatchReviewServer
from sam3_pipeline.usecase.capture import annotate

MATCH_DISTANCE = 0.03
CORRECTED_LEFT = 12

BUNDLE_SCHEMA = """
CREATE TABLE declarations(declaration_id TEXT, items_json TEXT);
CREATE TABLE images(session_id TEXT, image_id TEXT, declaration_id TEXT, arrangement_no INTEGER,
path TEXT, sha256 TEXT, width INTEGER, height INTEGER, tag_revision INTEGER, tag_status TEXT, depth_path TEXT);
CREATE TABLE hints(image_id TEXT, instrument_uid TEXT, instance_no INTEGER, visible INTEGER,
hint_source TEXT, center_json TEXT, end_a_json TEXT, end_b_json TEXT, occluded INTEGER);
"""


def _bundle(root: Path, *, session_id: str = "session_a", capture_date: str = "2026-10-23") -> Path:
    """入力アダプターが使用する列だけでセッション別の撮影データを作る。

    Args:
        root: テスト用の共有データルート。
        session_id: 他セッションとの読み分けを確認するID。
        capture_date: 日付をまたぐ探索を確認する日付。

    Returns:
        作成したRGB画像のパス。
    """
    bundle = root / capture_date / session_id
    bundle.mkdir(parents=True)
    image = bundle / "raw" / "image_a.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "red").save(image)
    # 同じ画像IDを別セッションでも使い、DB境界で取り違えないことを検証する。
    with closing(sqlite3.connect(bundle / "metadata.sqlite3")) as connection, connection:
        connection.executescript(BUNDLE_SCHEMA)
        connection.execute("INSERT INTO declarations VALUES (?, ?)", ("decl_a", json.dumps([{"instrument_uid": "tool_a", "instrument_name": "Forceps"}])))
        connection.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (session_id, "image_a", "decl_a", 1, image.relative_to(root).as_posix(), hashlib.sha256(image.read_bytes()).hexdigest(), 100, 100, 2, "tagged", None))
        connection.execute("INSERT INTO hints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", ("image_a", "tool_a", 1, 1, "manual", json.dumps({"x": 0.5, "y": 0.5, "status": "visible"}), None, None, 0))
    return image


def test_capture_source_relative_path_resolves_from_data_root(tmp_path: Path) -> None:
    """Resolve raw images from the service root rather than the bundle folder."""
    image = _bundle(tmp_path)

    samples = list(SqliteCaptureSource(tmp_path, timeout=1.0).iter_samples())

    assert samples[0].image_path == image.resolve()


def test_capture_source_tag_revision_preserves_snapshot(tmp_path: Path) -> None:
    """Keep tag identity and revision available for downstream invalidation."""
    _bundle(tmp_path)

    sample = next(SqliteCaptureSource(tmp_path, timeout=1.0).iter_samples())

    assert (sample.tag_revision, sample.tags[0].instrument_uid) == (2, "tool_a")


def test_capture_source_changed_image_rejects_hash_mismatch(tmp_path: Path) -> None:
    """Do not annotate an image different from the captured original."""
    image = _bundle(tmp_path)
    image.write_bytes(b"changed")

    with pytest.raises(CaptureError, match="hash mismatch"):
        list(SqliteCaptureSource(tmp_path, timeout=1.0).iter_samples())


@pytest.mark.parametrize("root_database", ["absent", "snapshot", "corrupt"])
def test_capture_source_session_databases_ignore_root_database(tmp_path: Path, root_database: str) -> None:
    """旧DBの有無や内容にかかわらず、日付・セッション順に各DBだけを読む。"""
    second = _bundle(tmp_path, session_id="session_b", capture_date="2026-10-24")
    first = _bundle(tmp_path)
    third = _bundle(tmp_path, session_id="session_c")
    # 旧DBが存在しても、探索・個別取得・現行性確認のどこからも参照させない。
    if root_database == "snapshot":
        with closing(sqlite3.connect(first.parents[1] / "metadata.sqlite3")) as source, closing(sqlite3.connect(tmp_path / "metadata.sqlite3")) as destination:
            source.backup(destination)
    elif root_database == "corrupt":
        (tmp_path / "metadata.sqlite3").write_bytes(b"not a SQLite database")
    before = {path: path.read_bytes() for path in tmp_path.rglob("metadata.sqlite3")}
    source = SqliteCaptureSource(tmp_path, timeout=1.0)

    samples = list(source.iter_samples())

    assert [sample.image_path for sample in samples] == [first.resolve(), third.resolve(), second.resolve()]
    # 同一画像IDでも所属セッションの情報を取得し、別DBの行と混同しない。
    for sample in samples:
        assert source.current_sample(sample.session_id, sample.image_id, sample.source_path) == sample
        assert source.is_current(sample)
    assert {path: path.read_bytes() for path in tmp_path.rglob("metadata.sqlite3")} == before


@pytest.mark.parametrize("root_database_exists", [False, True])
def test_capture_source_without_session_databases_rejects_input(tmp_path: Path, root_database_exists: bool) -> None:
    """セッションDBがなければ旧DBがあっても入力として受け付けない。"""
    if root_database_exists:
        # 旧配置だけの入力を受け付ける互換処理が残らないことを保証する。
        (tmp_path / "metadata.sqlite3").touch()
    source = SqliteCaptureSource(tmp_path, timeout=1.0)

    with pytest.raises(CaptureError, match="No session bundles"):
        list(source.iter_samples())


def test_capture_source_untagged_session_is_excluded(tmp_path: Path) -> None:
    """未タグ画像だけを含むセッションは推論対象にしない。"""
    first = _bundle(tmp_path)
    second = _bundle(tmp_path, session_id="session_b")
    with closing(sqlite3.connect(second.parents[1] / "metadata.sqlite3")) as connection, connection:
        connection.execute("UPDATE images SET tag_status = 'untagged'")

    samples = list(SqliteCaptureSource(tmp_path, timeout=1.0).iter_samples())

    assert [sample.image_path for sample in samples] == [first.resolve()]


@pytest.mark.parametrize("change", ["tag_revision", "untagged", "row_deleted", "database_deleted", "image_deleted"])
def test_capture_source_session_changes_override_old_snapshot(tmp_path: Path, change: str) -> None:
    """旧DBの行が残っていても、更新・削除は所属セッションDBで判定する。"""
    image = _bundle(tmp_path)
    database = image.parents[1] / "metadata.sqlite3"
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    sample = next(source.iter_samples())
    with closing(sqlite3.connect(database)) as session, closing(sqlite3.connect(tmp_path / "metadata.sqlite3")) as old_database:
        session.backup(old_database)
    # 旧DBを変更せず、撮影側で発生する更新・削除だけを再現する。
    if change == "database_deleted":
        database.unlink()
    elif change == "image_deleted":
        image.unlink()
    else:
        statements = {
            "tag_revision": "UPDATE images SET tag_revision = tag_revision + 1",
            "untagged": "UPDATE images SET tag_status = 'untagged'",
            "row_deleted": "DELETE FROM images",
        }
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute(statements[change])

    current = source.current_sample(sample.session_id, sample.image_id, sample.source_path)

    assert not source.is_current(sample)
    if change == "tag_revision":
        assert current is not None
        assert current.tag_revision == sample.tag_revision + 1
    else:
        assert current is None
    assert database.is_file() == (change != "database_deleted")


@pytest.mark.parametrize("operation", ["iter_samples", "current_sample", "is_current"])
@pytest.mark.parametrize("damage", ["corrupt", "missing_table", "missing_column"])
def test_capture_source_invalid_session_reports_database(tmp_path: Path, operation: str, damage: str) -> None:
    """DB破損やスキーマ不整合を対象パス付きのドメイン例外として通知する。"""
    image = _bundle(tmp_path)
    database = image.parents[1] / "metadata.sqlite3"
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    sample = next(source.iter_samples())
    # 行アクセス時に判明する列不足もSQLite実行時エラーと同じ契約で扱う。
    if damage == "corrupt":
        database.write_bytes(b"not a SQLite database")
    else:
        statement = "DROP TABLE images" if damage == "missing_table" else "ALTER TABLE images RENAME COLUMN tag_revision TO obsolete_revision"
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute(statement)

    with pytest.raises(CaptureError) as error:
        if operation == "iter_samples":
            list(source.iter_samples())
        elif operation == "current_sample":
            source.current_sample(sample.session_id, sample.image_id, sample.source_path)
        else:
            source.is_current(sample)

    assert str(database) in str(error.value)


def test_capture_predictor_without_checkpoint_uses_huggingface_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Allow the SAM3 builder to resolve its default Hugging Face checkpoint."""
    package = tmp_path / "sam3"
    package.mkdir()
    distribution = SimpleNamespace(version="test", read_text=lambda _: "{}")
    monkeypatch.setattr(capture_predictor.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(package / "__init__.py")))
    monkeypatch.setattr(capture_predictor.importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(capture_predictor, "digest_file", str)

    predictor = capture_predictor.Sam3CapturePredictor(None, "surgical instrument", 0.5)

    assert predictor.checkpoint is None
    assert predictor.signature


@pytest.mark.parametrize("coordinate", [-0.1, 1.1, float("nan")])
def test_tag_point_invalid_coordinate_rejects_input(coordinate: float) -> None:
    """Reject invalid coordinates before converting to image pixels."""
    with pytest.raises(CaptureError, match="normalized"):
        TagPoint(coordinate, 0.5)


def _sample(tags: tuple[InstanceTag, ...], path: Path = Path("image.png"), revision: int = 1) -> CaptureSample:
    """Build a pure capture contract for matching and workflow tests."""
    return CaptureSample("session_a", "image_a", "decl_a", 1, path, "digest", 100, 100, revision, tags, (("a", "a"),), path.name)


def _tag(uid: str, x: float, y: float) -> InstanceTag:
    """Create a visible center-only instance hint."""
    return InstanceTag(uid, uid, 1, True, "manual", TagPoint(x, y), None, None, False)


def _candidate(index: int, left: int, right: int) -> Candidate:
    """Create a rectangular foreground with pixel geometry available to matching."""
    pixels = np.zeros((100, 100), dtype=bool)
    pixels[10:90, left:right] = True
    return Candidate(index, (float(left), 10.0, float(right), 90.0), 0.9, ArrayMask(pixels))


def test_matcher_two_separate_objects_assigns_distinct_uids() -> None:
    """Use tags, not the shared SAM prompt, as class identity."""
    sample = _sample((_tag("a", 0.2, 0.5), _tag("b", 0.8, 0.5)))
    matcher = ScipyInstanceMatcher(MatchSettings(0.03, 0.05))

    result = matcher.match(sample, (_candidate(0, 10, 30), _candidate(1, 70, 90)))

    assert [(item.candidate_index, item.instrument_uid) for item in result.assignments] == [(0, "a"), (1, "b")]


def test_matcher_merged_mask_retains_ambiguity() -> None:
    """Do not arbitrarily label one merged detection as either tagged instrument."""
    sample = _sample((_tag("a", 0.2, 0.5), _tag("b", 0.8, 0.5)))
    matcher = ScipyInstanceMatcher(MatchSettings(0.03, 0.05))

    result = matcher.match(sample, (_candidate(0, 10, 90),))

    assert not result.assignments


def test_matcher_no_detections_records_missing_tag() -> None:
    """Keep missed instruments visible to reviewers rather than dropping the image."""
    sample = _sample((_tag("a", 0.2, 0.5),))
    matcher = ScipyInstanceMatcher(MatchSettings(0.03, 0.05))

    result = matcher.match(sample, ())

    assert result.issues == ("unmatched_tag:a:1",)


def test_array_mask_center_in_hole_measures_nearby_foreground() -> None:
    """Allow thin rings and small mask holes without requiring exact containment."""
    candidate = _candidate(0, 10, 90)
    assert isinstance(candidate.mask, ArrayMask)
    candidate.mask.pixels[49:52, 49:52] = False

    distance = candidate.mask.distance(0.5, 0.5)

    assert 0 < distance < MATCH_DISTANCE


class FakePredictor:
    """Implement the predictor port without importing Torch or SAM3."""

    signature = "test-model-fixed-prompt"

    def __init__(self) -> None:
        """Track calls to discriminate cache reuse from repeated inference."""
        self.calls = 0

    def infer(self, sample: CaptureSample) -> tuple[Candidate, ...]:
        """Return one deterministic candidate for cache integration tests."""
        self.calls += 1
        return (_candidate(0, 10, 30),)


class MemorySource:
    """Implement the capture source port using immutable test snapshots."""

    def __init__(self, sample: CaptureSample) -> None:
        """Retain a single immutable input snapshot."""
        self.samples = {(sample.session_id, sample.image_id): sample}

    def iter_samples(self) -> Iterator[CaptureSample]:
        """Yield one test sample."""
        yield from self.samples.values()

    def current_sample(self, session_id: str, image_id: str, source_path: str) -> CaptureSample | None:
        """Return a matching in-memory snapshot for isolated workflow tests."""
        sample = self.samples.get((session_id, image_id))
        return sample if sample is not None and sample.source_path == source_path else None

    def is_current(self, sample: CaptureSample) -> bool:
        """Compare a test snapshot without accessing a production database."""
        return self.samples.get((sample.session_id, sample.image_id)) == sample

    def add(self, sample: CaptureSample) -> None:
        """Register an additional test sample as a current SQL row."""
        self.samples[(sample.session_id, sample.image_id)] = sample

    def remove(self, sample: CaptureSample) -> None:
        """Simulate deletion of a capture row from the authoritative SQL."""
        self.samples.pop((sample.session_id, sample.image_id), None)


def _register_sample(root: Path, sample: CaptureSample) -> None:
    """Create the SQL row required by the real downstream batch validator."""
    database = root.joinpath(*Path(sample.source_path).parts[:2], "metadata.sqlite3")
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(BUNDLE_SCHEMA)
        connection.execute(
            "INSERT INTO declarations VALUES (?, ?)",
            (sample.declaration_id, json.dumps([{"instrument_uid": uid, "instrument_name": name} for uid, name in sample.instruments])),
        )
        connection.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sample.session_id, sample.image_id, sample.declaration_id, sample.arrangement_no, sample.source_path, sample.sha256, sample.width, sample.height, sample.tag_revision, "tagged", None),
        )
        for tag in sample.tags:
            connection.execute(
                "INSERT INTO hints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sample.image_id, tag.instrument_uid, tag.instance_no, int(tag.visible), tag.hint_source, json.dumps({"x": tag.center.x, "y": tag.center.y, "status": tag.center.status}) if tag.center else None, None, None, int(tag.occluded)),
            )


def _draft(tmp_path: Path, ledger: object | None = None) -> tuple[FileAnnotationStore, str, CaptureSample]:
    """Generate an actual image and draft through the public use case."""
    source_path = "2026-10-23/session_a/raw/source.png"
    image_path = tmp_path / Path(source_path)
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(image_path)
    sample = replace(_sample((_tag("a", 0.2, 0.5),), image_path), sha256=digest_file(image_path), source_path=source_path)
    _register_sample(tmp_path, sample)
    source = MemorySource(sample)
    store = FileAnnotationStore(tmp_path / "artifacts", source, ledger)
    revisions = annotate(source, FakePredictor(), ScipyInstanceMatcher(MatchSettings(0.03, 0.05)), store)
    return store, revisions[0], sample


def _prediction_store(root: Path) -> SqlitePredictionStore:
    """既存workflowテストの画像をSQL・PNG保存にも使用する。"""
    database = root / "2026-10-23" / "session_a" / "metadata.sqlite3"
    store = SqlitePredictionStore(database, root, root / "annotations" / "sam3", 1.0, model_metadata={"prompt": "metal tool", "min_score": 0.5})
    store.initialize()
    return store


def test_annotate_sql_cache_tag_change_reuses_prediction(tmp_path: Path) -> None:
    """SQL経路では新規NPZを作らず、タグ変更時も再推論しない。"""
    store, _, sample = _draft(tmp_path)
    predictions = _prediction_store(tmp_path)
    source = MemorySource(sample)
    predictor = FakePredictor()
    predictor.signature = "new-native-model"
    old_npz = set((store.root / "cache").glob("*.npz"))
    matcher = ScipyInstanceMatcher(MatchSettings(0.03, 0.05))

    first = annotate(source, predictor, matcher, store, prediction_store=predictions)
    source.add(replace(sample, tag_revision=sample.tag_revision + 1))
    second = annotate(source, predictor, matcher, store, prediction_store=predictions)

    assert predictor.calls == 1
    assert first != second
    assert set((store.root / "cache").glob("*.npz")) == old_npz


def test_annotate_legacy_npz_imports_without_inference(tmp_path: Path) -> None:
    """既存キーのNPZを遅延移行し、モデルを再実行しない。"""
    store, _, sample = _draft(tmp_path)
    predictions = _prediction_store(tmp_path)
    predictor = FakePredictor()

    annotate(MemorySource(sample), predictor, ScipyInstanceMatcher(MatchSettings(0.03, 0.05)), store, prediction_store=predictions)

    assert predictor.calls == 0
    with sqlite3.connect(predictions.database_path) as connection:
        assert connection.execute("SELECT model_signature, provenance FROM sam3_predictions").fetchall() == [(None, "legacy_npz")]


def test_migrate_predictions_repeated_preserves_old_artifacts(tmp_path: Path) -> None:
    """NPZ移行は元の承認・注釈・キャッシュを変更せず再実行できる。"""
    store, revision, _ = _draft(tmp_path)
    store.approve(revision, "reviewer", confirm_complete=True)
    predictions = _prediction_store(tmp_path)
    before = {path: digest_file(path) for path in store.root.rglob("*") if path.is_file()}

    first = store.migrate_predictions(predictions)
    second = store.migrate_predictions(predictions)

    assert (first, second) == (1, 1)
    assert {path: digest_file(path) for path in before} == before
    with sqlite3.connect(predictions.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_prediction_objects").fetchone() == (1,)


def test_migrate_predictions_missing_npz_rejects_partial_result(tmp_path: Path) -> None:
    """草案のみ残った不完全キャッシュを推論完了とみなさない。"""
    store, _, _ = _draft(tmp_path)
    predictions = _prediction_store(tmp_path)
    next((store.root / "cache").glob("*.npz")).unlink()

    with pytest.raises(CaptureError, match="missing"):
        store.migrate_predictions(predictions)

    with sqlite3.connect(predictions.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_predictions").fetchone() == (0,)


def test_migrate_predictions_cli_imports_without_model(tmp_path: Path) -> None:
    """移行コマンドはモデルやGPUを初期化せず実行できる。"""
    config = tmp_path / "config.json"
    source_root = tmp_path / "capture"
    image = _bundle(source_root)
    sample = next(SqliteCaptureSource(source_root, timeout=1.0).iter_samples())
    source = MemorySource(sample)
    artifacts = FileAnnotationStore(tmp_path / "annotations", source)
    annotate(source, FakePredictor(), ScipyInstanceMatcher(MatchSettings(0.03, 0.05)), artifacts)
    atomic_json(config, {"capture_root": str(source_root), "output_root": str(artifacts.root), "sqlite_timeout": 1.0})

    result = main(["--config", str(config), "migrate-predictions"])

    assert result == 0
    assert image.is_file()
    assert not (source_root / "metadata.sqlite3").exists()
    with sqlite3.connect(image.parents[1] / "metadata.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_predictions").fetchone() == (1,)


def _object_review(tmp_path: Path) -> tuple[FileAnnotationStore, SqliteObjectReviewStore, CaptureSample, str]:
    """既存草案と撮影SQLから、未確認の物体レビューを構築する。"""
    store, revision, sample = _draft(tmp_path)
    predictions = _prediction_store(tmp_path)
    store.migrate_predictions(predictions)
    reviews = SqliteObjectReviewStore(predictions.database_path, 1.0, predictions)
    reviews.initialize()
    metadata = read_json(store.root / "drafts" / revision / "metadata.json")
    annotation = read_json(store.root / "reviews" / revision / "annotation.json")
    reviews.sync_revision(sample, revision, metadata, annotation)
    store.review_store = reviews
    return store, reviews, sample, revision


def _three_object_review(tmp_path: Path) -> tuple[FileAnnotationStore, SqliteObjectReviewStore, CaptureSample, str]:
    """同一画像の3物体を個別に確認できるSQL・PNG・草案を作る。"""
    previous, reviews, sample, _ = _object_review(tmp_path)
    tags = tuple(replace(_tag("a", position, 0.5), instance_no=index + 1) for index, position in enumerate((0.2, 0.45, 0.75)))
    sample = replace(sample, tags=tags, tag_revision=sample.tag_revision + 1)
    store = FileAnnotationStore(previous.root, MemorySource(sample), review_store=reviews)
    with sqlite3.connect(reviews.database_path) as connection:
        connection.execute("UPDATE images SET tag_revision=? WHERE image_id=?", (sample.tag_revision, sample.image_id))
    signature = "three-object-model"
    key = hashlib.sha256(f"{sample.sha256}:{signature}".encode()).hexdigest()
    candidates = (_candidate(0, 10, 30), _candidate(1, 35, 55), _candidate(2, 65, 85))
    reviews.predictions.save(sample, key, candidates, signature)
    result = MatchResult(tuple(Assignment(index, "a", index + 1) for index in range(len(candidates))), ())
    revision = store.save_draft(sample, candidates, result, key, "three-object-matcher")
    return store, reviews, sample, revision


def test_object_review_initial_state_is_pending(tmp_path: Path) -> None:
    """旧bbox承認とは別に、物体確認は未確認から開始する。"""
    _, reviews, sample, revision = _object_review(tmp_path)

    objects = reviews.objects(sample, revision)

    assert [item["dino_status"] for item in objects] == ["pending"]


def test_object_review_accepted_requires_image_approval(tmp_path: Path) -> None:
    """採用だけでは公開せず、承認後に推論原本を公開する。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_dino_objects").fetchone() == (0,)

    store.approve(revision, "operator", confirm_complete=True)

    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT object_index FROM sam3_dino_objects").fetchall() == [(0,)]


def test_object_review_revoked_clears_decision_and_retains_history(tmp_path: Path) -> None:
    """承認取消後は未確認へ戻し、以前の採用を履歴として保持する。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    annotation = read_json(store.root / "reviews" / revision / "annotation.json")
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    reviews.record_annotation(revision, annotation, "approved", {"approval_id": "approval"})

    reviews.record_annotation(revision, annotation, "revoked")

    assert reviews.objects(sample, revision)[0]["dino_status"] == "pending"
    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_dino_objects").fetchone() == (0,)
        assert connection.execute("SELECT dino_status FROM sam3_object_review_events ORDER BY event_id").fetchall() == [("accepted",), ("pending",)]


def test_object_review_new_revision_invalidates_previous_decision(tmp_path: Path) -> None:
    """対応付け設定の変更は旧採用を引き継がない。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    annotation = read_json(store.root / "reviews" / revision / "annotation.json")
    metadata = read_json(store.root / "drafts" / revision / "metadata.json")
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")

    metadata["matching_signature"] = "changed-matching"
    new_revision = content_hash(metadata)
    reviews.sync_revision(sample, new_revision, metadata, annotation)

    assert reviews.objects(sample, new_revision)[0]["dino_status"] == "pending"
    with pytest.raises(CaptureError, match="current"):
        reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")


def test_object_review_two_of_three_exports_original_indices(tmp_path: Path) -> None:
    """3物体中2物体だけを公開し、不採用番号を詰めない。"""
    store, reviews, sample, revision = _three_object_review(tmp_path)
    for index, status in enumerate(("accepted", "rejected", "accepted")):
        reviews.decide(sample, revision, index, status=status, reviewer="operator")

    store.approve(revision, "approver", confirm_complete=True)

    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT object_index FROM sam3_dino_objects ORDER BY object_index").fetchall() == [(0,), (2,)]


def test_object_review_yolo_edit_never_replaces_prediction_bbox(tmp_path: Path) -> None:
    """YOLO修正後もDINOのbbox・maskは推論原本だけを参照する。"""
    store, reviews, sample, revision = _three_object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    reviews.decide(sample, revision, 2, status="accepted", reviewer="operator")
    store.approve(revision, "approver", confirm_complete=True)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))
    annotation = read_json(store.root / "reviews" / revision / "annotation.json")
    annotation["shapes"][0]["points"][0][0] = CORRECTED_LEFT

    server._save_annotation({"revision": revision, "annotation": annotation})
    store.approve(revision, "approver", confirm_complete=True)

    with sqlite3.connect(reviews.database_path) as connection:
        boxes = connection.execute("SELECT object_index, bbox_json FROM sam3_dino_objects ORDER BY object_index").fetchall()
        assert [(index, json.loads(box)[0]) for index, box in boxes] == [(0, 10.0), (2, 65.0)]
        saved = json.loads(connection.execute("SELECT annotation_json FROM sam3_review_revisions WHERE revision=?", (revision,)).fetchone()[0])
        assert saved["shapes"][0]["points"][0][0] == CORRECTED_LEFT


def test_object_review_unchecked_object_is_not_published(tmp_path: Path) -> None:
    """同一画像に未確認物体が残っていても、確認した物体だけを公開する。"""
    store, reviews, sample, revision = _three_object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")

    store.approve(revision, "approver", confirm_complete=True)

    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT object_index FROM sam3_dino_objects").fetchall() == [(0,)]


@pytest.mark.parametrize("field,value", [("tag_revision", 99), ("tag_status", "untagged"), ("sha256", "changed"), ("path", "changed.png")])
def test_object_review_changed_capture_is_excluded(tmp_path: Path, field: str, value: str | int) -> None:
    """撮影SQLの状態変更はアプリ再起動前でも公開ビューから除外する。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    store.approve(revision, "approver", confirm_complete=True)
    statements = {
        "tag_revision": "UPDATE images SET tag_revision=?",
        "tag_status": "UPDATE images SET tag_status=?",
        "sha256": "UPDATE images SET sha256=?",
        "path": "UPDATE images SET path=?",
    }

    with sqlite3.connect(reviews.database_path) as connection:
        connection.execute(statements[field], (value,))

    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_dino_objects").fetchone() == (0,)


def test_object_review_api_requires_explicit_confirmation(tmp_path: Path) -> None:
    """原本確認を省略したAPI要求は採用へ進めない。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))

    with pytest.raises(CaptureError, match="Confirm original"):
        server._set_object_review({"revision": revision, "object_index": 0, "status": "accepted", "reviewer": "operator"})

    assert reviews.objects(sample, revision)[0]["dino_status"] == "pending"


@pytest.mark.parametrize("status,reviewer", [("accepted", ""), ("unknown", "operator"), ("accepted", None)])
def test_object_review_api_invalid_input_rejects_decision(tmp_path: Path, status: str, reviewer: str | None) -> None:
    """担当者未入力や不正採否を記録しない。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))

    with pytest.raises(CaptureError):
        server._set_object_review({"revision": revision, "object_index": 0, "status": status, "reviewer": reviewer, "confirm_original": True})

    assert reviews.objects(sample, revision)[0]["dino_status"] == "pending"


def test_object_review_api_preserves_decision_across_restart(tmp_path: Path) -> None:
    """現行草案の再同期や一覧再取得で確認結果を未確認に戻さない。"""
    store, _, sample, revision = _object_review(tmp_path)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))
    server._set_object_review({"revision": revision, "object_index": 0, "status": "accepted", "reviewer": "operator", "confirm_original": True})

    store.sync_object_reviews()
    restarted = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))

    assert restarted._items()[0]["objects"][0]["dino_status"] == "accepted"


def test_object_review_overlay_preserves_original_mask(tmp_path: Path) -> None:
    """可視化PNGは元maskの領域だけに色を付け、保存済み原本は変えない。"""
    store, _, sample, revision = _object_review(tmp_path)
    before = {path: digest_file(path) for path in (tmp_path / "annotations").rglob("*.png")}
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))

    content = server._object_mask(f"/api/mask/{revision}/0")

    with Image.open(io.BytesIO(content)) as image:
        alpha = np.asarray(image)[..., -1]
        assert np.array_equal(alpha > 0, _candidate(0, 10, 30).mask.pixels)
    assert {path: digest_file(path) for path in before} == before


def test_object_review_mask_corruption_rejects_acceptance(tmp_path: Path) -> None:
    """正式保存maskの改変を、再起動後の確認済みデータとして受け入れない。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    store.approve(revision, "operator", confirm_complete=True)
    with sqlite3.connect(reviews.database_path) as connection:
        relative = connection.execute("SELECT mask_path FROM sam3_approved_masks WHERE revision=?", (revision,)).fetchone()[0]
    path = tmp_path / relative
    Image.new("L", (sample.width, sample.height), color=255).save(path)
    reviews.predictions = _prediction_store(tmp_path)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))

    with pytest.raises(CaptureError, match="hash"):
        server._set_object_review({"revision": revision, "object_index": 0, "status": "accepted", "reviewer": "operator", "confirm_original": True})

    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT dino_status FROM sam3_object_reviews WHERE revision=?", (revision,)).fetchone() == ("accepted",)


def test_object_review_unresolved_assignment_rejects_acceptance(tmp_path: Path) -> None:
    """器具対応が未確定の物体を採用しない。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    metadata = read_json(store.root / "drafts" / revision / "metadata.json")
    metadata["matching"]["assignments"] = []
    new_revision = content_hash(metadata)
    reviews.sync_revision(sample, new_revision, metadata, read_json(store.root / "reviews" / revision / "annotation.json"))

    with pytest.raises(CaptureError, match="unresolved"):
        reviews.decide(sample, new_revision, 0, status="accepted", reviewer="operator")

    assert reviews.objects(sample, new_revision)[0]["dino_status"] == "pending"


def test_object_review_reapproval_does_not_restore_revoked_acceptance(tmp_path: Path) -> None:
    """再承認しても取消前の採用を自動復活させない。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    store.approve(revision, "approver", confirm_complete=True)
    store.revoke(revision)

    store.approve(revision, "approver", confirm_complete=True)
    store.sync_object_reviews()

    assert reviews.objects(sample, revision)[0]["dino_status"] == "pending"
    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_dino_objects").fetchone() == (0,)


def test_object_review_progress_uses_sql_after_restart(tmp_path: Path) -> None:
    """一覧確認状態は再起動後もSQLから復元する。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))

    server._set_reviewed({"revision": revision, "reviewed": True})
    (store.root / "review_progress.json").unlink()

    assert BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))._items()[0]["reviewed"] is True
    assert reviews.progress()[revision]["reviewed"] is True


def test_object_review_initialize_preserves_decisions(tmp_path: Path) -> None:
    """schema初期化を繰り返しても物体確認を失わない。"""
    _, reviews, sample, revision = _object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")

    reviews.initialize()

    assert reviews.objects(sample, revision)[0]["dino_status"] == "accepted"


def test_object_review_reacceptance_changes_public_event_id(tmp_path: Path) -> None:
    """同一秒・同一承認でも再採用を別イベントとして識別する。"""
    store, reviews, sample, revision = _object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    store.approve(revision, "approver", confirm_complete=True)
    with sqlite3.connect(reviews.database_path) as connection:
        previous = connection.execute("SELECT review_event_id FROM sam3_dino_objects").fetchone()[0]

    reviews.decide(sample, revision, 0, status="rejected", reviewer="operator")
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")

    with sqlite3.connect(reviews.database_path) as connection:
        assert connection.execute("SELECT review_event_id FROM sam3_dino_objects").fetchone()[0] > previous


def test_object_review_schema_v1_upgrade_preserves_decisions(tmp_path: Path) -> None:
    """公開ビューを拡張しても既存の確認状態と履歴は維持する。"""
    _, reviews, sample, revision = _object_review(tmp_path)
    reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    with sqlite3.connect(reviews.database_path) as connection:
        connection.execute("DELETE FROM sam3_review_schema_migrations")
        connection.execute("INSERT INTO sam3_review_schema_migrations(version) VALUES (1)")

    reviews.initialize()

    assert reviews.objects(sample, revision)[0]["dino_status"] == "accepted"
    with sqlite3.connect(reviews.database_path) as connection:
        assert {row[0] for row in connection.execute("SELECT version FROM sam3_review_schema_migrations")} == {1, REVIEW_SCHEMA_VERSION}


def test_annotation_lifecycle_records_approval_and_manifest_in_session_sqlite(tmp_path: Path) -> None:
    """所属セッションDBへ画像承認と公開履歴を記録する。"""
    store, revision, sample = _draft(tmp_path, SqliteAnnotationLedger(tmp_path))
    database = sample.image_path.parents[1] / "metadata.sqlite3"

    # Act
    store.record_review_state(revision, True)
    store.approve(revision, "reviewer_a", confirm_complete=True)
    manifest = store.export_approved(tmp_path / "artifacts" / "exports" / sample.session_id)

    # Assert
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT status, reviewer, manifest_path, exported_at FROM annotation_revisions WHERE revision = ?",
            (revision,),
        ).fetchone()
    assert row is not None
    assert row[0] == "exported" and row[1] == "reviewer_a"
    assert row[2] == str(manifest.resolve()) and row[3]


def test_existing_approved_manifest_syncs_to_session_metadata(tmp_path: Path) -> None:
    """既存manifestの承認を所属セッションDBへ同期する。"""
    store, revision, sample = _draft(tmp_path)
    store.approve(revision, "reviewer_a", confirm_complete=True)
    store.export_approved(store.root / "exports" / sample.session_id)
    database = sample.image_path.parents[1] / "metadata.sqlite3"
    synced_store = FileAnnotationStore(store.root, store.capture_source, SqliteAnnotationLedger(tmp_path))

    # Act
    synced_count = synced_store.sync_existing_exports()

    # Assert
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT status, manifest_path FROM annotation_revisions WHERE revision = ?",
            (revision,),
        ).fetchone()
    assert synced_count == 1
    assert row is not None and row[0] == "exported"
    assert Path(row[1]).is_file()


def test_annotate_updated_tags_reuses_model_candidates(tmp_path: Path) -> None:
    """Changing tag revision must not trigger another expensive model inference."""
    store, _, sample = _draft(tmp_path)
    predictor = FakePredictor()

    annotate(MemorySource(replace(sample, tag_revision=2)), predictor, ScipyInstanceMatcher(MatchSettings(0.03, 0.05)), store)

    assert predictor.calls == 0


def test_batch_review_publish_preserves_yolo_manifest_contract(tmp_path: Path) -> None:
    """選択セッション名で公開し、下流YOLO validatorの契約を維持する。"""
    store, revision, _ = _draft(tmp_path)
    atomic_json(store.root / "review_progress.json", {revision: {"reviewed": True, "exception_note": "", "allow_empty": False}})
    review_server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), tuple(store.capture_source.iter_samples()))

    manifest = review_server._publish(
        {
            "reviewer": "reviewer_a",
            "session_id": "session_a",
            "confirm_complete": True,
        }
    )
    samples = ApprovedBatchSource((manifest,), tmp_path).load()

    assert len(samples) == 1
    assert samples[0].labels[0].instrument_uid == "a"
    assert samples[0].image_path == Path(tmp_path / "2026-10-23" / "session_a" / "raw" / "source.png")
    assert manifest.parent.name == "session_a"
    assert not list(manifest.parent.rglob("*.png"))


def test_batch_review_publish_includes_only_selected_session(tmp_path: Path) -> None:
    """セッション別公開に他セッションの画像を混在させない。"""
    store, revision, sample = _draft(tmp_path)
    second_path = "2026-10-24/session_b/raw/second.png"
    second_image = tmp_path / Path(second_path)
    second_image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "blue").save(second_image)
    second_sample = replace(
        sample,
        session_id="session_b",
        image_id="image_b",
        image_path=second_image,
        sha256=digest_file(second_image),
        source_path=second_path,
    )
    _register_sample(tmp_path, second_sample)
    store.capture_source.add(second_sample)
    second_revision = annotate(
        MemorySource(second_sample),
        FakePredictor(),
        ScipyInstanceMatcher(MatchSettings(0.03, 0.05)),
        store,
    )[0]
    atomic_json(
        store.root / "review_progress.json",
        {
            revision: {"reviewed": True, "exception_note": "", "allow_empty": False},
            second_revision: {"reviewed": True, "exception_note": "", "allow_empty": False},
        },
    )
    review_server = BatchReviewServer(
        store,
        ConfidenceThresholds(0.65, 0.85),
        (sample, second_sample),
    )

    manifest = review_server._publish({"reviewer": "reviewer_a", "session_id": "session_a", "confirm_complete": True})

    assert [record["source"]["session_id"] for record in read_json(manifest)["samples"]] == ["session_a"]


@pytest.mark.parametrize(
    "reviewer,confirmed,message",
    [
        ("", True, "一括公開欄の確認担当者"),
        ("  ", True, "一括公開欄の確認担当者"),
        (None, True, "一括公開欄の確認担当者"),
        ("operator", False, "チェックを入れて"),
        ("operator", True, "未確認画像が1件"),
    ],
)
def test_review_server_publish_missing_requirement_reports_specific_reason(tmp_path: Path, reviewer: str | None, confirmed: bool, message: str) -> None:
    """公開条件の不足箇所を区別し、承認やexportを行わず通知する。"""
    store, _, sample = _draft(tmp_path)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))

    with pytest.raises(CaptureError, match=message):
        server._publish({"reviewer": reviewer, "session_id": sample.session_id, "confirm_complete": confirmed})

    assert not (store.root / "exports").exists()
    assert not list((store.root / "approvals").glob("*.json"))


def test_review_server_sql_snapshot_hides_rows_deleted_before_start(tmp_path: Path) -> None:
    """SQLから消えた画像に過去ドラフトだけで再アクセスできない。"""
    store, _, sample = _draft(tmp_path)
    store.capture_source.remove(sample)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))

    assert server._items() == []


def test_review_server_snapshot_does_not_grow_after_start(tmp_path: Path) -> None:
    """レビュー画面の対象数を起動時のSQL集合に固定する。"""
    store, _, sample = _draft(tmp_path)
    server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), (sample,))
    source_path = "2026-10-23/session_b/raw/second.png"
    image_path = tmp_path / Path(source_path)
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "blue").save(image_path)
    added = replace(
        sample,
        session_id="session_b",
        image_id="image_b",
        image_path=image_path,
        sha256=digest_file(image_path),
        tags=(_tag("b", 0.2, 0.5),),
        instruments=(("b", "b"),),
        source_path=source_path,
    )
    _register_sample(tmp_path, added)
    store.capture_source.add(added)
    annotate(
        MemorySource(added),
        FakePredictor(),
        ScipyInstanceMatcher(MatchSettings(0.03, 0.05)),
        store,
    )

    assert len(server._items()) == 1


def test_approval_rejects_capture_deleted_from_sql(tmp_path: Path) -> None:
    """承認時にもSQL正本を照合し、削除済み画像を固定しない。"""
    store, revision, sample = _draft(tmp_path)
    store.capture_source.remove(sample)

    with pytest.raises(CaptureError, match="no longer present and unchanged in SQL"):
        store.approve(revision, "reviewer", confirm_complete=True)


def test_export_ignores_approval_deleted_from_sql(tmp_path: Path) -> None:
    """過去の承認コピーが残っていても、SQL削除後のmanifestへ含めない。"""
    store, revision, sample = _draft(tmp_path)
    store.approve(revision, "reviewer", confirm_complete=True)
    store.capture_source.remove(sample)

    with pytest.raises(CaptureError, match="No currently approved"):
        store.export_approved(tmp_path / "stale_batch")


def test_review_draft_does_not_copy_capture_image(tmp_path: Path) -> None:
    """レビュー成果物はSQL上の画像を参照し、重複画像ファイルを作らない。"""
    store, revision, sample = _draft(tmp_path)

    review_files = list((store.root / REVIEWS_DIR / revision).iterdir())

    assert [path.name for path in review_files] == [ANNOTATION_FILE]
    assert read_json(store.root / REVIEWS_DIR / revision / ANNOTATION_FILE)["imagePath"] == sample.source_path


def test_batch_review_save_allows_unresolved_draft_label(tmp_path: Path) -> None:
    """Keep a partial review editable while final approval remains strict."""
    store, revision, _ = _draft(tmp_path)
    annotation = read_json(store.root / REVIEWS_DIR / revision / ANNOTATION_FILE)
    annotation["shapes"][0]["label"] = "__unresolved__"
    review_server = BatchReviewServer(store, ConfidenceThresholds(0.65, 0.85), tuple(store.capture_source.iter_samples()))

    review_server._save_annotation({"revision": revision, "annotation": annotation})

    assert read_json(store.root / REVIEWS_DIR / revision / ANNOTATION_FILE)["shapes"][0]["label"] == "__unresolved__"


def test_annotate_repeat_run_preserves_manual_corrections(tmp_path: Path) -> None:
    """Existing Labelme workspaces belong to reviewers, not regeneration."""
    store, revision, sample = _draft(tmp_path)
    path = store.root / REVIEWS_DIR / revision / ANNOTATION_FILE
    data = read_json(path)
    data["shapes"][0]["points"][0][0] = 12
    atomic_json(path, data)

    annotate(MemorySource(sample), FakePredictor(), ScipyInstanceMatcher(MatchSettings(0.03, 0.05)), store)

    assert read_json(path)["shapes"][0]["points"][0][0] == CORRECTED_LEFT


def test_approve_unchanged_correct_draft_exports_image(tmp_path: Path) -> None:
    """Approval is explicit and does not require changing an already correct box."""
    store, revision, _ = _draft(tmp_path)
    store.approve(revision, "reviewer", confirm_complete=True)

    manifest = store.export_approved(tmp_path / "batch")

    assert len(read_json(manifest)["samples"]) == 1


def test_export_modified_after_approval_requires_reapproval(tmp_path: Path) -> None:
    """Do not infer approval from the mere existence of a historical review record."""
    store, revision, _ = _draft(tmp_path)
    store.approve(revision, "reviewer", confirm_complete=True)
    path = store.root / REVIEWS_DIR / revision / ANNOTATION_FILE
    data = read_json(path)
    data["shapes"][0]["points"][0][0] = 12
    atomic_json(path, data)

    with pytest.raises(CaptureError, match="changed after approval"):
        store.export_approved(tmp_path / "batch")


def test_export_without_approval_rejects_drafts(tmp_path: Path) -> None:
    """Raw SAM drafts cannot enter training merely because inference finished."""
    store, _, _ = _draft(tmp_path)

    with pytest.raises(CaptureError, match="No currently approved"):
        store.export_approved(tmp_path / "batch")


def test_approve_unresolved_label_rejects_input(tmp_path: Path) -> None:
    """Unknown objects must be identified or deliberately corrected by a reviewer."""
    store, revision, _ = _draft(tmp_path)
    path = store.root / REVIEWS_DIR / revision / ANNOTATION_FILE
    data = read_json(path)
    data["shapes"][0]["label"] = "__unresolved__"
    atomic_json(path, data)

    with pytest.raises(CaptureError, match="Resolve all shapes"):
        store.approve(revision, "reviewer", confirm_complete=True)


def test_capture_cli_status_requires_no_model_environment(tmp_path: Path) -> None:
    """Status reads SQL-backed artifacts without importing SAM3 or Torch."""
    _bundle(tmp_path / "capture")
    config = tmp_path / "config.json"
    atomic_json(config, {"capture_root": "capture", "output_root": "artifacts", "sqlite_timeout": 1.0})

    result = main(["--config", str(config), "status"])

    assert result == 0


def test_export_new_tag_revision_rejects_old_approval(tmp_path: Path) -> None:
    """A new tag snapshot invalidates approval even when model candidates are reused."""
    store, revision, sample = _draft(tmp_path)
    store.approve(revision, "reviewer", confirm_complete=True)
    annotate(MemorySource(replace(sample, tag_revision=2)), FakePredictor(), ScipyInstanceMatcher(MatchSettings(0.03, 0.05)), store)

    with pytest.raises(CaptureError, match="Source tags or matching changed"):
        store.export_approved(tmp_path / "batch")


class FakeTrainingBackend:
    """Implement the cross-project training port without any model runtime."""

    def train(self, data_config: Path, weights: Path, run_dir: Path, settings: TrainingSettings) -> dict[str, float]:
        """Publish a test checkpoint after the real training integrity gates."""
        output = run_dir / "weights" / "best.pt"
        output.parent.mkdir()
        output.write_bytes(b"integration-checkpoint")
        return {"mAP50": 0.8}


def test_capture_cycle_approved_batch_reaches_yolo_training(tmp_path: Path) -> None:
    """Bridge SAM review artifacts to independently imported YOLO batch contracts."""
    store, first_revision, sample = _draft(tmp_path)
    second_source_path = "2026-10-23/session_b/raw/second.png"
    second_image = tmp_path / Path(second_source_path)
    second_image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "blue").save(second_image)
    second = replace(sample, session_id="session_b", image_id="image_b", image_path=second_image, sha256=digest_file(second_image), tags=(_tag("b", 0.2, 0.5),), instruments=(("b", "b"),), source_path=second_source_path)
    _register_sample(tmp_path, second)
    store.capture_source.add(second)
    second_revision = annotate(MemorySource(second), FakePredictor(), ScipyInstanceMatcher(MatchSettings(0.03, 0.05)), store)[0]
    store.approve(first_revision, "reviewer", confirm_complete=True)
    store.approve(second_revision, "reviewer", confirm_complete=True)
    manifest = store.export_approved(tmp_path / "batch")
    with sqlite3.connect(tmp_path / "metadata.sqlite3") as connection:
        connection.executescript(
            """CREATE TABLE declarations(declaration_id TEXT, items_json TEXT);
            CREATE TABLE images(session_id TEXT, image_id TEXT, declaration_id TEXT, arrangement_no INTEGER,
                path TEXT, sha256 TEXT, width INTEGER, height INTEGER, tag_revision INTEGER, tag_status TEXT,
                PRIMARY KEY(session_id, image_id));
            CREATE TABLE hints(image_id TEXT, instrument_uid TEXT, instance_no INTEGER, visible INTEGER,
                hint_source TEXT, center_json TEXT, end_a_json TEXT, end_b_json TEXT, occluded INTEGER);
                CREATE TABLE annotation_revisions(session_id TEXT, image_id TEXT, tag_revision INTEGER,
                    revision TEXT, status TEXT, approval_id TEXT, reviewer TEXT, approved_at TEXT,
                    manifest_path TEXT, annotation_path TEXT, annotation_sha256 TEXT, exported_at TEXT,
                    updated_at TEXT, PRIMARY KEY(session_id, image_id, revision));
            CREATE TABLE session_splits(session_id TEXT PRIMARY KEY, split TEXT, assigned_at TEXT,
                policy_version TEXT, image_count_at_assignment INTEGER);
            CREATE TABLE instrument_classes(instrument_uid TEXT PRIMARY KEY, class_id INTEGER UNIQUE,
                instrument_name TEXT, created_at TEXT);
            CREATE TABLE dataset_versions(version TEXT PRIMARY KEY, status TEXT, created_at TEXT, updated_at TEXT,
                split_policy_json TEXT, dataset_path TEXT, manifest_path TEXT, manifest_sha256 TEXT);
            CREATE TABLE dataset_samples(version TEXT, session_id TEXT, image_id TEXT, split TEXT,
                tag_revision INTEGER, annotation_revision TEXT, approval_id TEXT, image_sha256 TEXT,
                label_sha256 TEXT, image_path TEXT, label_path TEXT, PRIMARY KEY(version, session_id, image_id));
            CREATE TABLE training_runs(run_id TEXT PRIMARY KEY, version TEXT, status TEXT, parent_run_id TEXT,
                initial_weights_sha256 TEXT, best_weights_sha256 TEXT, best_weights_path TEXT,
                settings_json TEXT, metrics_json TEXT, error TEXT, started_at TEXT, finished_at TEXT);"""
        )
        records = {(record["source"]["session_id"], record["source"]["image_id"]): record for record in read_json(manifest)["samples"]}
        for item, split in ((sample, "train"), (second, "val")):
            declaration_id = item.declaration_id
            connection.execute(
                "INSERT INTO declarations VALUES (?, ?)",
                (declaration_id, json.dumps([{"instrument_uid": uid, "instrument_name": name} for uid, name in item.instruments])),
            )
            connection.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item.session_id, item.image_id, declaration_id, item.arrangement_no, item.source_path, item.sha256, item.width, item.height, item.tag_revision, "tagged"),
            )
            connection.execute(
                "INSERT INTO session_splits VALUES (?, ?, ?, ?, ?)",
                (item.session_id, split, "test", "test", 1),
            )
            record = records[(item.session_id, item.image_id)]
            connection.execute(
                """INSERT INTO annotation_revisions(
                    session_id, image_id, tag_revision, revision, status, approval_id,
                    reviewer, approved_at, manifest_path, annotation_path, annotation_sha256,
                    exported_at, updated_at
                ) VALUES (?, ?, ?, ?, 'exported', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.session_id,
                    item.image_id,
                    item.tag_revision,
                    record["revision"],
                    record["approval_id"],
                    record["reviewer"],
                    record["approved_at"],
                    str(manifest.resolve()),
                    record["annotation"],
                    record["labelme_sha256"],
                    record["approved_at"],
                    record["approved_at"],
                ),
            )
    writer = FrozenDatasetWriter(tmp_path / "datasets", TrainingMetadataLedger(tmp_path), SplitSettings(42, 0.2, 0.1))
    weights = tmp_path / "initial.pt"
    weights.write_bytes(b"initial-checkpoint")

    dataset = build_dataset(ApprovedBatchSource((manifest,), tmp_path), writer, "capture_v1")
    best = train_frozen_dataset(
        dataset,
        weights,
        tmp_path / "run",
        TrainingSettings(1, 1, 640, "cpu", 42),
        FakeTrainingBackend(),
        capture_root=tmp_path,
        ledger=TrainingMetadataLedger(tmp_path),
    )

    assert best.read_bytes() == b"integration-checkpoint"


@pytest.mark.parametrize("shape_type", ["polygon", "point"])
def test_approve_nonrectangle_shape_rejects_conversion(tmp_path: Path, shape_type: str) -> None:
    """Avoid silently turning keypoint guides or polygon vertices into YOLO boxes."""
    store, revision, _ = _draft(tmp_path)
    path = store.root / REVIEWS_DIR / revision / ANNOTATION_FILE
    data = read_json(path)
    data["shapes"][0]["shape_type"] = shape_type
    atomic_json(path, data)

    with pytest.raises(CaptureError, match="rectangles"):
        store.approve(revision, "reviewer", confirm_complete=True)
