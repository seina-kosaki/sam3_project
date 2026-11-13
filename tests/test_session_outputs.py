"""一時セッションDBでDINO・YOLO出力の分離と公開状態を検証する。

責務:
    SAM3側の保存契約だけを検証し、DINO・YOLOの実モデルや運用データは使用しない。
処理の流れ:
    1. 同一画像IDを持つ二つのセッションとタグを作成する。
    2. 推論保存・承認・採否・公開を実行する。
    3. 専用テーブルの原本・フラグ・履歴・保存先を照合する。
関連ファイル:
    capture_workflow.py、infrastructure/handoff_store.py、infrastructure/session_database.py。
設計意図:
    ファイルとSQLを実際に作成し、セッション境界と取消の回帰をGPUなしで確認する。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import capture_workflow
from capture_workflow import main
from sam3_pipeline.domain.annotation import Assignment, Candidate, MatchResult
from sam3_pipeline.domain.capture import CaptureError, CaptureSample
from sam3_pipeline.domain.confidence import ConfidenceThresholds
from sam3_pipeline.infrastructure.annotation_ledger import SqliteAnnotationLedger
from sam3_pipeline.infrastructure.annotation_store import FileAnnotationStore, atomic_json, content_hash, digest_file, read_json
from sam3_pipeline.infrastructure.capture_source import SqliteCaptureSource
from sam3_pipeline.infrastructure.matching import ArrayMask
from sam3_pipeline.infrastructure.object_review_store import SqliteObjectReviewStore
from sam3_pipeline.infrastructure.prediction_store import SqlitePredictionStore
from sam3_pipeline.infrastructure.review_server import BatchReviewServer
from sam3_pipeline.infrastructure.selection_server import SelectiveInferenceServer
from sam3_pipeline.infrastructure.session_database import session_database

IMAGE_SIZE = 8
ORIGINAL_LEFT = 2.0
SIGNATURE = "session-model"
MODEL_METADATA = {"prompt": "metal tool", "min_score": 0.5}
CAPTURE_SCHEMA = """
CREATE TABLE declarations(declaration_id TEXT, items_json TEXT);
CREATE TABLE images(session_id TEXT, image_id TEXT, declaration_id TEXT, arrangement_no INTEGER,
    path TEXT, sha256 TEXT, width INTEGER, height INTEGER, tag_revision INTEGER, tag_status TEXT, depth_path TEXT);
CREATE TABLE hints(image_id TEXT, instrument_uid TEXT, instance_no INTEGER, visible INTEGER,
    hint_source TEXT, center_json TEXT, end_a_json TEXT, end_b_json TEXT, occluded INTEGER);
"""


@dataclass
class Workflow:
    """テストごとの共有ルートと、実際に接続した各保存先を束ねる。"""

    root: Path
    store: FileAnnotationStore
    predictions: SqlitePredictionStore
    reviews: SqliteObjectReviewStore
    samples: tuple[CaptureSample, ...]
    revisions: dict[str, str]


@pytest.fixture
def workflow(tmp_path: Path) -> Workflow:
    """同じ画像IDを持つ二つのセッションへ推論と初期レビューを保存する。

    Args:
        tmp_path: pytestの一時領域。

    Returns:
        GPUを使わない二セッションのワークフロー。
    """
    root = tmp_path / "capture"
    # 同一ID・同一画像hashでも、セッションをまたいだ保存先の混同を検出する。
    for session_id in ("session_a", "session_b"):
        bundle = root / "2026-10-30" / session_id
        image = bundle / "raw" / "image.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)).save(image)
        with closing(sqlite3.connect(bundle / "metadata.sqlite3")) as connection, connection:
            connection.executescript(CAPTURE_SCHEMA)
            connection.execute("INSERT INTO declarations VALUES (?, ?)", ("declaration", json.dumps([{"instrument_uid": "tool", "instrument_name": "器具"}])))
            connection.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (session_id, "image", "declaration", 1, image.relative_to(root).as_posix(), digest_file(image), IMAGE_SIZE, IMAGE_SIZE, 1, "tagged", None))
            connection.execute("INSERT INTO hints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", ("image", "tool", 1, 1, "manual", json.dumps({"x": 0.5, "y": 0.5}), None, None, 0))
    source = SqliteCaptureSource(root, 1.0)
    samples = tuple(source.iter_samples())
    predictions = SqlitePredictionStore(root, root, root / "annotations" / "sam3", 1.0, model_metadata=MODEL_METADATA)
    predictions.initialize()
    reviews = SqliteObjectReviewStore(root, 1.0, predictions)
    reviews.initialize()
    store = FileAnnotationStore(tmp_path / "work", source, SqliteAnnotationLedger(root, 1.0), review_store=reviews)
    revisions: dict[str, str] = {}
    pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    pixels[2:6, 2:6] = True
    candidates = (Candidate(0, (2.0, 2.0, 6.0, 6.0), 0.9, ArrayMask(pixels)),)
    # 下流と同じSQL・PNGを用い、ダミー候補だけで承認までの入口を作る。
    for sample in samples:
        prediction_id = hashlib.sha256(f"{sample.sha256}:{SIGNATURE}".encode()).hexdigest()
        predictions.save(sample, prediction_id, candidates, SIGNATURE)
        revisions[sample.session_id] = store.save_draft(sample, candidates, MatchResult((Assignment(0, "tool", 1),), ()), prediction_id, "matching")
    return Workflow(root, store, predictions, reviews, samples, revisions)


def _flags(workflow: Workflow, sample: CaptureSample) -> tuple[int, int]:
    """所属セッションに保存された物体・画像の公開可否を取得する。

    Args:
        workflow: テスト用の保存先。
        sample: 照合する撮影画像。

    Returns:
        DINOとYOLOの公開可否。
    """
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        dino = connection.execute("SELECT is_dino_ready FROM sam3_dino_results WHERE revision=?", (workflow.revisions[sample.session_id],)).fetchone()
        yolo = connection.execute("SELECT is_yolo_ready FROM sam3_yolo_results WHERE revision=?", (workflow.revisions[sample.session_id],)).fetchone()
    return int(dino[0]), int(yolo[0])


def _publish(workflow: Workflow, sample: CaptureSample) -> Path:
    """画像承認・物体採用・セッションexportを完了する。

    Args:
        workflow: テスト用の保存先。
        sample: 公開する撮影画像。

    Returns:
        公開したmanifestのパス。
    """
    revision = workflow.revisions[sample.session_id]
    workflow.reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")
    workflow.store.approve(revision, "operator", confirm_complete=True)
    return workflow.store.export_approved(workflow.store.root / "exports" / sample.session_id, sample.session_id)


def test_complete_review_saves_accepted_decision_without_export(workflow: Workflow) -> None:
    """採用候補を一括保存し、DINOだけを条件付き公開する。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review({"revision": revision, "reviewer": " operator ", "confirm_complete": True,
                             "decisions": [{"object_index": 0, "status": "accepted"}]})

    assert workflow.reviews.approved(revision)
    assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "accepted"
    item = next(item for item in server._items() if item["revision"] == revision)
    assert item["dino_approved"]
    assert item["objects"][0]["dino_status"] == "accepted"
    assert item["objects"][0]["saved_mask_exists"]
    workflow.store.sync_object_reviews()
    item = next(item for item in server._items() if item["revision"] == revision)
    assert item["dino_approved"]
    assert item["objects"][0]["dino_status"] == "accepted"
    assert item["objects"][0]["saved_mask_exists"]
    assert read_json(workflow.store.root / "approvals" / f"{revision}.json")["reviewer"] == "operator"
    assert _flags(workflow, sample) == (1, 0)
    assert _flags(workflow, workflow.samples[1]) == (0, 0)
    assert not (workflow.store.root / "exports").exists()
    target = session_database(workflow.root, sample.session_id).parent / "masks" / "image_000.png"
    candidates = workflow.predictions.load(sample, workflow.reviews.objects(sample, revision)[0]["prediction_id"])
    assert candidates is not None
    with Image.open(target) as mask:
        assert np.array_equal(np.asarray(mask) > 0, candidates[0].mask.pixels)
    assert not list((workflow.root / "annotations").rglob("*.png"))
    assert workflow.reviews.objects(sample, revision)[0]["saved_mask_exists"]
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        expected = (target.relative_to(workflow.root).as_posix(), digest_file(target))
        assert connection.execute("SELECT mask_path, mask_sha256 FROM sam3_dino_results WHERE revision=?", (revision,)).fetchone() == expected
        assert connection.execute("SELECT mask_path, mask_sha256 FROM sam3_dino_objects WHERE revision=?", (revision,)).fetchone() == expected


def test_complete_review_rejects_save_without_an_accepted_mask(workflow: Workflow) -> None:
    """採用maskが0件なら承認・採否・maskを保存しない。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    with pytest.raises(CaptureError, match="採用されたmaskがない"):
        server._complete_review({
            "revision": revision,
            "reviewer": "operator",
            "confirm_complete": True,
            "decisions": [{"object_index": 0, "status": "rejected"}],
            "assignments": [{"object_index": 0, "instrument_uid": None, "instance_no": None}],
        })

    assert not workflow.reviews.approved(revision)
    assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "pending"
    assert not (workflow.store.root / "approvals" / f"{revision}.json").exists()
    assert not (session_database(workflow.root, sample.session_id).parent / "masks" / "image_000.png").exists()


def test_complete_review_saves_selected_tag_and_leaves_other_candidates_pending(workflow: Workflow) -> None:
    """採用タグだけ保存し、未確認の別候補はpendingのまま残す。"""
    sample = workflow.samples[0]
    pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    pixels[2:6, 2:6] = True
    candidates = tuple(
        Candidate(index, (1.0 + index, 2.0, 5.0 + index, 6.0), 0.9, ArrayMask(np.roll(pixels, index, axis=1)))
        for index in range(3)
    )
    signature = f"{SIGNATURE}-partial-review"
    prediction_id = hashlib.sha256(f"{sample.sha256}:{signature}".encode()).hexdigest()
    workflow.predictions.save(sample, prediction_id, candidates, signature)
    revision = workflow.store.save_draft(
        sample, candidates, MatchResult((Assignment(0, "tool", 1),), ()), prediction_id, "partial-review"
    )
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review({
        "revision": revision,
        "reviewer": "operator",
        "confirm_complete": True,
        "decisions": [{"object_index": 0, "status": "accepted"}, {"object_index": 1, "status": "rejected"}],
        "assignments": [
            {"object_index": 0, "instrument_uid": "tool", "instance_no": 1},
            {"object_index": 1, "instrument_uid": None, "instance_no": None},
        ],
    })

    objects = workflow.reviews.objects(sample, revision)
    assert [item["dino_status"] for item in objects] == ["accepted", "rejected", "pending"]
    assert [item["saved_mask_exists"] for item in objects] == [True, False, False]
    item = next(item for item in server._items() if item["revision"] == revision)
    assert item["dino_approved"] and not item["approved"]
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute(
            "SELECT object_index FROM sam3_dino_results WHERE revision=? AND is_dino_ready=1", (revision,)
        ).fetchall() == [(0,)]


@pytest.mark.parametrize("consent", [False, True])
def test_complete_review_existing_mask_requires_overwrite_consent(workflow: Workflow, consent: bool) -> None:
    """既存maskはYesだけで更新し、Noや未指定なら採否も承認も変更しない。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    target = session_database(workflow.root, sample.session_id).parent / "masks" / "image_000.png"
    target.parent.mkdir()
    target.write_bytes(b"previous-mask")
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection, connection:
        connection.execute("INSERT INTO sam3_approved_masks VALUES (?, ?, ?, ?)", (revision, 0, target.relative_to(workflow.root).as_posix(), digest_file(target)))
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    payload = {"revision": revision, "reviewer": "operator", "confirm_complete": True, "decisions": [{"object_index": 0, "status": "accepted"}], "overwrite_masks": consent}
    assert workflow.reviews.objects(sample, revision)[0]["saved_mask_exists"]

    if consent:
        server._complete_review(payload)
        assert target.read_bytes() != b"previous-mask"
        assert _flags(workflow, sample) == (1, 0)
    else:
        with pytest.raises(CaptureError, match="上書き"):
            server._complete_review(payload)
        assert target.read_bytes() == b"previous-mask"
        assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "pending"
        assert not workflow.reviews.approved(revision)
        assert not (workflow.store.root / "approvals" / f"{revision}.json").exists()
    assert not list(target.parent.glob(".mask-save-*"))


def test_complete_review_overwrite_sql_failure_restores_previous_mask(workflow: Workflow) -> None:
    """上書き後のSQL失敗で元の承認ファイルとDINO公開状態を維持する。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    payload = {"revision": revision, "reviewer": "operator", "confirm_complete": True, "decisions": [{"object_index": 0, "status": "accepted"}]}
    server._complete_review(payload)
    target = session_database(workflow.root, sample.session_id).parent / "masks" / "image_000.png"
    # 置換前後の内容を区別し、復元処理が実際に元バイト列を戻すことを確認する。
    target.write_bytes(b"previous-approved-content")
    previous = target.read_bytes()
    # 既存承認を保持したまま、ファイル配置後の画像更新だけを失敗させる。
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection, connection:
        connection.execute("""CREATE TRIGGER fail_replace BEFORE UPDATE ON sam3_review_revisions
            WHEN NEW.status='approved' BEGIN SELECT RAISE(ABORT, '承認更新失敗'); END""")

    with pytest.raises(CaptureError, match="承認更新失敗"):
        server._complete_review({**payload, "overwrite_masks": True})

    assert target.read_bytes() == previous
    assert _flags(workflow, sample) == (1, 0)
    assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "accepted"
    assert not list(target.parent.glob(".mask-save-*"))


def test_complete_review_new_prediction_overwrites_only_approved_copy(workflow: Workflow) -> None:
    """再推論の保存同意後に承認用だけを置換し、旧推論原本を保持する。"""
    sample = workflow.samples[0]
    old_revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    server._complete_review({"revision": old_revision, "reviewer": "operator", "confirm_complete": True, "decisions": [{"object_index": 0, "status": "accepted"}]})
    target = session_database(workflow.root, sample.session_id).parent / "masks" / "image_000.png"
    old_bytes = target.read_bytes()
    pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    pixels[3:5, 3:5] = True
    candidates = (Candidate(0, (3.0, 3.0, 5.0, 5.0), 0.95, ArrayMask(pixels)),)
    signature = f"{SIGNATURE}-updated"
    prediction_id = hashlib.sha256(f"{sample.sha256}:{signature}".encode()).hexdigest()
    workflow.predictions.save(sample, prediction_id, candidates, signature)
    new_revision = workflow.store.save_draft(sample, candidates, MatchResult((Assignment(0, "tool", 1),), ()), prediction_id, "updated")
    payload = {"revision": new_revision, "reviewer": "operator", "confirm_complete": True, "decisions": [{"object_index": 0, "status": "accepted"}]}

    with pytest.raises(CaptureError, match="上書き"):
        server._complete_review(payload)
    server._complete_review({**payload, "overwrite_masks": True})

    target = session_database(workflow.root, sample.session_id).parent / "masks" / "image_000.png"
    with Image.open(target) as mask:
        assert np.array_equal(np.asarray(mask) > 0, pixels)
    assert target.read_bytes() != old_bytes
    assert not list((workflow.root / "annotations").rglob("*.png"))
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT is_dino_ready FROM sam3_dino_results WHERE revision=?", (old_revision,)).fetchone() == (0,)
        assert connection.execute("SELECT is_dino_ready FROM sam3_dino_results WHERE revision=?", (new_revision,)).fetchone() == (1,)


@pytest.mark.parametrize(
    "changes",
    [
        {"reviewer": ""},
        {"reviewer": None},
        {"confirm_complete": False},
        {"decisions": None},
        {"decisions": []},
        {"decisions": [None]},
        {"decisions": [{"object_index": True, "status": "accepted"}]},
        {"decisions": [{"object_index": "0", "status": "accepted"}]},
        {"decisions": [{"object_index": 0, "status": "pending"}]},
        {"decisions": [{"object_index": 1, "status": "accepted"}]},
        {"decisions": [{"object_index": 0, "status": "accepted"}, {"object_index": 0, "status": "rejected"}]},
    ],
)
def test_complete_review_invalid_payload_preserves_pending_state(workflow: Workflow, changes: dict[str, Any]) -> None:
    """不完全な一括確定から承認JSONや部分採用を生成しない。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    payload = {"revision": revision, "reviewer": "operator", "confirm_complete": True, "decisions": [{"object_index": 0, "status": "accepted"}], **changes}

    with pytest.raises(CaptureError):
        server._complete_review(payload)

    assert not workflow.reviews.approved(revision)
    assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "pending"
    assert not (workflow.store.root / "approvals" / f"{revision}.json").exists()
    assert _flags(workflow, sample) == (0, 0)
    assert not list(session_database(workflow.root, sample.session_id).parent.glob("masks/*.png"))


def test_complete_review_sql_failure_rolls_back_object_and_image(workflow: Workflow) -> None:
    """画像承認の書込失敗時に、直前の物体採否と履歴もロールバックする。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    # 物体更新後の画像更新で失敗させ、途中の採用だけが残らないことを検証する。
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection, connection:
        connection.execute("""CREATE TRIGGER fail_approval BEFORE UPDATE ON sam3_review_revisions
            WHEN NEW.status='approved' BEGIN SELECT RAISE(ABORT, '承認保存のテスト失敗'); END""")

    with pytest.raises(CaptureError, match="承認保存のテスト失敗"):
        server._complete_review({"revision": revision, "reviewer": "operator", "confirm_complete": True, "decisions": [{"object_index": 0, "status": "accepted"}]})

    assert not workflow.reviews.approved(revision)
    assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "pending"
    assert not (workflow.store.root / "approvals" / f"{revision}.json").exists()
    assert _flags(workflow, sample) == (0, 0)
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_object_review_events WHERE revision=?", (revision,)).fetchone()[0] == 0


def test_complete_review_multiple_objects_are_saved_together(workflow: Workflow) -> None:
    """複数物体の選択を一括保存し、使わない物体はDINOだけから除外する。"""
    sample = workflow.samples[0]
    pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    pixels[2:6, 2:6] = True
    candidates = tuple(Candidate(index, (2.0 + index, 2.0, 6.0 + index, 6.0), 0.9, ArrayMask(np.roll(pixels, index, axis=1))) for index in (0, 1))
    signature = f"{SIGNATURE}-two-objects"
    prediction_id = hashlib.sha256(f"{sample.sha256}:{signature}".encode()).hexdigest()
    workflow.predictions.save(sample, prediction_id, candidates, signature)
    # タグ未対応のbboxも比較用候補として保全し、DINO採否だけを分離する。
    revision = workflow.store.save_draft(sample, candidates, MatchResult((Assignment(0, "tool", 1),), ()), prediction_id, "two-objects")
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review(
        {"revision": revision, "reviewer": "operator", "confirm_complete": True, "exception_note": "重複検出のためDINOでは片方を使わない", "decisions": [{"object_index": 0, "status": "accepted"}, {"object_index": 1, "status": "rejected"}]}
    )

    assert [item["dino_status"] for item in workflow.reviews.objects(sample, revision)] == ["accepted", "rejected"]
    assert workflow.reviews.approved(revision)
    saved_annotation = read_json(workflow.store.root / "reviews" / revision / "annotation.json")
    assert len(saved_annotation["shapes"]) + len(saved_annotation["review_candidates"]) == len(candidates)
    assert len(saved_annotation["review_candidates"]) == 1
    assert saved_annotation["review_candidates"][0]["label"] == "__unresolved__"
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT object_index FROM sam3_dino_results WHERE revision=? AND is_dino_ready=1", (revision,)).fetchall() == [(0,)]


def _ambiguous_dino(workflow: Workflow) -> tuple[str, dict[str, Any]]:
    """自動対応が空の三候補と、一候補だけを手動採用する要求を返す。

    Args:
        workflow: 一時DBと実際の保存処理を持つテスト環境。

    Returns:
        新レビュー版とDINO専用の保存要求。
    """
    sample = workflow.samples[0]
    pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    pixels[2:6, 2:6] = True
    candidates = tuple(Candidate(index, (1.0 + index, 2.0, 5.0 + index, 6.0), 0.55, ArrayMask(np.roll(pixels, index, axis=1))) for index in range(3))
    signature = f"{SIGNATURE}-ambiguous"
    prediction_id = hashlib.sha256(f"{sample.sha256}:{signature}".encode()).hexdigest()
    workflow.predictions.save(sample, prediction_id, candidates, signature)
    revision = workflow.store.save_draft(sample, candidates, MatchResult((), ("ambiguous:tool:1",)), prediction_id, "ambiguous")
    workflow.revisions[sample.session_id] = revision
    # 未対応のYOLO候補は変更せず、DINOだけを一候補に絞る。
    payload = {
        "revision": revision,
        "reviewer": "operator",
        "confirm_complete": True,
        "decisions": [{"object_index": index, "status": "accepted" if index == 1 else "rejected"} for index in range(3)],
        "assignments": [{"object_index": index, "instrument_uid": "tool" if index == 1 else None, "instance_no": 1 if index == 1 else None} for index in range(3)],
    }
    return revision, payload


def test_dino_manual_selection_accepts_one_low_score_candidate_and_survives_restart(workflow: Workflow) -> None:
    """低スコアの曖昧候補をタグへ手動対応し、YOLO未解決でもDINOだけを公開する。

    Args:
        workflow: 一時セッションの保存先。
    """
    revision, payload = _ambiguous_dino(workflow)
    sample = workflow.samples[0]
    annotation = workflow.store.root / "reviews" / revision / "annotation.json"
    before = annotation.read_bytes()
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review(payload)
    masks = session_database(workflow.root, sample.session_id).parent / "masks"
    assert sorted(path.name for path in masks.glob("*.png")) == ["image_001.png"]
    workflow.reviews.initialize()
    workflow.store.sync_object_reviews()
    workflow.store.sync_existing_exports()

    objects = workflow.reviews.objects(sample, revision)
    assert [item["dino_status"] for item in objects] == ["rejected", "accepted", "rejected"]
    assert objects[1]["instrument_uid"] == "tool"
    assert objects[1]["instance_no"] == 1
    assert annotation.read_bytes() == before
    assert not (workflow.store.root / "approvals" / f"{revision}.json").exists()
    item = next(item for item in server._items() if item["revision"] == revision)
    assert item["dino_approved"] and not item["approved"]
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT object_index FROM sam3_dino_objects WHERE revision=?", (revision,)).fetchall() == [(1,)]
        assert connection.execute("SELECT object_index FROM sam3_dino_results WHERE revision=? AND is_dino_ready=1", (revision,)).fetchall() == [(1,)]
        assert connection.execute("SELECT is_yolo_ready FROM sam3_yolo_results WHERE revision=?", (revision,)).fetchone()[0] == 0
        assert connection.execute("SELECT object_index, previous_uid, instrument_uid, instance_no FROM sam3_object_assignment_events WHERE revision=?", (revision,)).fetchall() == [(1, None, "tool", 1)]


def test_dino_review_allows_assigning_only_some_decided_candidates(workflow: Workflow) -> None:
    """判定する候補のうち一部だけタグ対応を指定できる。"""
    revision, payload = _ambiguous_dino(workflow)
    sample = workflow.samples[0]
    payload["decisions"] = [
        {"object_index": 1, "status": "accepted"},
        {"object_index": 2, "status": "rejected"},
    ]
    payload["assignments"] = [
        {"object_index": 1, "instrument_uid": "tool", "instance_no": 1},
    ]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review(payload)

    objects = workflow.reviews.objects(sample, revision)
    assert [item["dino_status"] for item in objects] == ["pending", "accepted", "rejected"]


def test_dino_acceptance_rejects_unmentioned_candidates_for_the_same_tag(workflow: Workflow) -> None:
    """APIで1候補だけ採用した場合も、同じタグの未指定候補を自動で不採用にする。"""
    revision, _ = _ambiguous_dino(workflow)
    sample = workflow.samples[0]
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection, connection:
        connection.execute(
            "UPDATE sam3_object_reviews SET instrument_uid='tool', instance_no=1 WHERE revision=? AND object_index IN (0, 2)",
            (revision,),
        )
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review({
        "revision": revision,
        "reviewer": "operator",
        "confirm_complete": True,
        "decisions": [{"object_index": 1, "status": "accepted"}],
        "assignments": [{"object_index": 1, "instrument_uid": "tool", "instance_no": 1}],
    })

    objects = workflow.reviews.objects(sample, revision)
    assert [item["dino_status"] for item in objects] == ["rejected", "accepted", "rejected"]
    assert [item["saved_mask_exists"] for item in objects] == [False, True, False]


def _seed_old_masks(workflow: Workflow, revision: str) -> Path:
    """旧バグで全候補を保存した状態を一時DBに作る。"""
    database = session_database(workflow.root, workflow.samples[0].session_id)
    directory = database.parent / "masks"
    directory.mkdir(exist_ok=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        for index in range(3):
            target = directory / f"image_{index:03d}.png"
            target.write_bytes(f"old-{index}".encode())
            connection.execute("INSERT INTO sam3_approved_masks VALUES (?, ?, ?, ?)", (revision, index, target.relative_to(workflow.root).as_posix(), digest_file(target)))
    return directory


@pytest.mark.parametrize("selected", [None, 1, 2])
def test_dino_cleanup_removes_rejected_masks_without_overwrite_consent(workflow: Workflow, selected: int | None) -> None:
    """不採用PNGだけの存在は同意を要求せず、今回採用だけを保存する。"""
    revision, payload = _ambiguous_dino(workflow)
    sample = workflow.samples[0]
    directory = _seed_old_masks(workflow, revision)
    if selected is not None:
        (directory / f"image_{selected:03d}.png").unlink()
    for decision, assignment in zip(payload["decisions"], payload["assignments"], strict=True):
        accepted = decision["object_index"] == selected
        decision["status"] = "accepted" if accepted else "rejected"
        assignment.update(instrument_uid="tool" if accepted else None, instance_no=1 if accepted else None)
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    if selected is None:
        previous_masks = {path: path.read_bytes() for path in directory.glob("*.png")}
        with pytest.raises(CaptureError, match="採用されたmaskがない"):
            server._complete_review(payload)
        assert {path: path.read_bytes() for path in directory.glob("*.png")} == previous_masks
        assert all(item["dino_status"] == "pending" for item in workflow.reviews.objects(sample, revision))
        with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM sam3_object_review_events WHERE revision=?", (revision,)
            ).fetchone() == (0,)
        return

    server._complete_review(payload)

    expected = [f"image_{selected:03d}.png"]
    assert sorted(path.name for path in directory.glob("*.png")) == expected
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT object_index FROM sam3_approved_masks WHERE revision=?", (revision,)).fetchall() == [(selected,)]
        rows = connection.execute("SELECT object_index, mask_path, is_dino_ready FROM sam3_dino_results WHERE revision=? ORDER BY object_index", (revision,)).fetchall()
        assert len(rows) == len(payload["decisions"])
        for index, path, ready in rows:
            assert ready == int(index == selected)
            assert path.startswith("memory:") == (index != selected)
        assert connection.execute("SELECT COUNT(*) FROM sam3_object_review_events WHERE revision=?", (revision,)).fetchone() == (3,)


def test_dino_switch_cleans_old_revision_ledger_but_keeps_history(workflow: Workflow) -> None:
    """採用切替で旧版の正式参照も整理し、推論と監査履歴は保持する。"""
    sample = workflow.samples[0]
    old_revision = workflow.revisions[sample.session_id]
    _publish(workflow, sample)
    other_manifest = _publish(workflow, workflow.samples[1])
    other_bytes = other_manifest.read_bytes()
    other_mask = session_database(workflow.root, workflow.samples[1].session_id).parent / "masks" / "image_000.png"
    other_png = other_mask.read_bytes()
    database = session_database(workflow.root, sample.session_id)
    with closing(sqlite3.connect(database)) as connection:
        originals = connection.execute("SELECT * FROM sam3_prediction_objects").fetchall()
        old_events = connection.execute("SELECT * FROM sam3_object_review_events WHERE revision=?", (old_revision,)).fetchall()
    revision, payload = _ambiguous_dino(workflow)
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review(payload)
    payload["decisions"][1]["status"] = "rejected"
    payload["decisions"][2]["status"] = "accepted"
    payload["assignments"][1].update(instrument_uid=None, instance_no=None)
    payload["assignments"][2].update(instrument_uid="tool", instance_no=1)
    server._complete_review(payload)

    directory = database.parent / "masks"
    assert sorted(path.name for path in directory.glob("*.png")) == ["image_002.png"]
    assert other_mask.read_bytes() == other_png
    assert other_manifest.read_bytes() == other_bytes
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT * FROM sam3_prediction_objects WHERE prediction_id=?", (originals[0][2],)).fetchall() == originals
        assert connection.execute("SELECT * FROM sam3_object_review_events WHERE revision=?", (old_revision,)).fetchall() == old_events
        assert connection.execute("SELECT revision, object_index FROM sam3_approved_masks").fetchall() == [(revision, 2)]
        assert connection.execute("SELECT mask_path, is_dino_ready FROM sam3_dino_results WHERE revision=?", (old_revision,)).fetchone()[0].startswith("memory:")
        assert connection.execute("SELECT object_index FROM sam3_dino_objects").fetchall() == [(2,)]


def test_dino_cleanup_preserves_other_image_in_same_session(workflow: Workflow) -> None:
    """同じセッションの前方一致名の他画像と未登録ファイルを保護する。"""
    sample = workflow.samples[0]
    database = session_database(workflow.root, sample.session_id)
    other_path = sample.image_path.with_name("image_extra.png")
    other_path.write_bytes(sample.image_path.read_bytes())
    other = replace(sample, image_id="other", image_path=other_path, source_path=other_path.relative_to(workflow.root).as_posix())
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            """INSERT INTO images SELECT session_id, 'other', declaration_id, arrangement_no,
            ?, sha256, width, height, tag_revision, tag_status, depth_path FROM images WHERE image_id=?""",
            (other.source_path, sample.image_id),
        )
        connection.execute("INSERT INTO hints SELECT 'other', instrument_uid, instance_no, visible, hint_source, center_json, end_a_json, end_b_json, occluded FROM hints WHERE image_id=?", (sample.image_id,))
    prediction_id = workflow.reviews.objects(sample, workflow.revisions[sample.session_id])[0]["prediction_id"]
    candidates = workflow.predictions.load(sample, prediction_id)
    assert candidates is not None
    workflow.predictions.save(other, prediction_id, candidates, SIGNATURE)
    other_revision = workflow.store.save_draft(other, candidates, MatchResult((Assignment(0, "tool", 1),), ()), prediction_id, "matching")
    workflow.reviews.decide(other, other_revision, 0, status="accepted", reviewer="operator")
    workflow.store.approve(other_revision, "operator", confirm_complete=True)
    manifest = workflow.store.export_approved(workflow.store.root / "other-export", sample.session_id)
    revision, payload = _ambiguous_dino(workflow)
    directory = _seed_old_masks(workflow, revision)
    (directory / "image_001.png").unlink()
    unrelated = directory / "image_999.png"
    unrelated.write_bytes(b"untracked")
    protected_paths = [other_path, directory / "image_extra_000.png", unrelated, manifest]
    protected = {path: path.read_bytes() for path in protected_paths}
    with closing(sqlite3.connect(database)) as connection:
        yolo = connection.execute("SELECT * FROM sam3_yolo_results WHERE revision=?", (other_revision,)).fetchall()
        dino = connection.execute("SELECT * FROM sam3_dino_results WHERE revision=?", (other_revision,)).fetchall()
        masks = connection.execute("SELECT * FROM sam3_approved_masks WHERE revision=?", (other_revision,)).fetchall()
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review(payload)

    assert {path: path.read_bytes() for path in protected_paths} == protected
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT * FROM sam3_yolo_results WHERE revision=?", (other_revision,)).fetchall() == yolo
        assert connection.execute("SELECT * FROM sam3_dino_results WHERE revision=?", (other_revision,)).fetchall() == dino
        assert connection.execute("SELECT * FROM sam3_approved_masks WHERE revision=?", (other_revision,)).fetchall() == masks


@pytest.mark.parametrize("failure", ["no_consent", "replace", "delete", "sql", "commit"])
def test_dino_cleanup_failure_restores_all_files_and_database(workflow: Workflow, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """削除を伴う保存の通常失敗とcommit失敗でファイル・全SQLを復元する。"""
    revision, payload = _ambiguous_dino(workflow)
    directory = _seed_old_masks(workflow, revision)
    database = session_database(workflow.root, workflow.samples[0].session_id)
    if failure in {"sql", "commit"}:
        with closing(sqlite3.connect(database)) as connection, connection:
            if failure == "sql":
                connection.execute("""CREATE TRIGGER fail_cleanup BEFORE UPDATE ON sam3_review_revisions
                    BEGIN SELECT RAISE(ABORT, 'cleanup SQL failure'); END""")
            else:
                connection.executescript("""
                    CREATE TABLE commit_parent(id INTEGER PRIMARY KEY);
                    CREATE TABLE commit_child(id INTEGER REFERENCES commit_parent(id) DEFERRABLE INITIALLY DEFERRED);
                    CREATE TRIGGER fail_commit AFTER UPDATE ON sam3_review_revisions
                    BEGIN INSERT INTO commit_child VALUES (1); END;
                """)
    before_files = {path: path.read_bytes() for path in directory.glob("*.png")}
    with closing(sqlite3.connect(database)) as connection:
        before_sql = list(connection.iterdump())
    if failure == "replace":
        original_replace = Path.replace

        def fail_replace(path: Path, target: Path) -> Path:
            if path.name.startswith("new-"):
                raise OSError("test replacement failure")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", fail_replace)
    elif failure == "delete":
        original_unlink = Path.unlink

        def fail_unlink(path: Path, missing_ok: bool = False) -> None:
            if path == directory / "image_002.png":
                raise OSError("test deletion failure")
            original_unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", fail_unlink)
    if failure != "no_consent":
        payload["overwrite_masks"] = True
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    with pytest.raises(CaptureError):
        server._complete_review(payload)

    assert {path: path.read_bytes() for path in directory.glob("*.png")} == before_files
    assert not list(directory.glob(".mask-save-*"))
    with closing(sqlite3.connect(database)) as connection:
        assert list(connection.iterdump()) == before_sql
    assert not (workflow.store.root / "approvals" / f"{revision}.json").exists()


@pytest.mark.parametrize("conflict", ["unknown", "prediction", "foreign", "outside"])
def test_dino_cleanup_refuses_unsafe_ownership(workflow: Workflow, conflict: str) -> None:
    """所有不明・共有原本・他画像・領域外のファイルは同意があっても変更しない。"""
    revision, payload = _ambiguous_dino(workflow)
    directory = _seed_old_masks(workflow, revision)
    database = session_database(workflow.root, workflow.samples[0].session_id)
    path = (directory / "image_000.png").relative_to(workflow.root).as_posix()
    with closing(sqlite3.connect(database)) as connection, connection:
        if conflict == "unknown":
            connection.execute("DELETE FROM sam3_approved_masks WHERE revision=? AND object_index=0", (revision,))
        elif conflict == "prediction":
            connection.execute("UPDATE sam3_prediction_objects SET mask_path=? WHERE prediction_id<>?", (path, workflow.reviews.objects(workflow.samples[0], revision)[0]["prediction_id"]))
        elif conflict == "foreign":
            old_revision = connection.execute("SELECT revision FROM sam3_review_revisions WHERE is_current=0").fetchone()[0]
            connection.execute("UPDATE sam3_review_revisions SET image_id='other' WHERE revision=?", (old_revision,))
            connection.execute("INSERT INTO sam3_approved_masks VALUES (?, 0, ?, 'old')", (old_revision, path))
        else:
            connection.execute("UPDATE sam3_approved_masks SET mask_path=? WHERE revision=? AND object_index=0", (workflow.samples[0].source_path, revision))
    payload["overwrite_masks"] = True
    before = {item: item.read_bytes() for item in workflow.root.rglob("*.png")}
    with closing(sqlite3.connect(database)) as connection:
        before_sql = list(connection.iterdump())
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    with pytest.raises(CaptureError):
        server._complete_review(payload)

    assert {item: item.read_bytes() for item in workflow.root.rglob("*.png")} == before
    with closing(sqlite3.connect(database)) as connection:
        assert list(connection.iterdump()) == before_sql


@pytest.mark.parametrize("damage", ["duplicate", "unknown", "boolean", "unresolved", "index"])
def test_dino_invalid_manual_selection_keeps_assignment_history_empty(workflow: Workflow, damage: str) -> None:
    """不正な手動対応は採否・履歴・maskを部分保存しない。

    Args:
        workflow: 一時セッションの保存先。
        damage: 不正に変更する要求項目。
    """
    revision, payload = _ambiguous_dino(workflow)
    if damage == "duplicate":
        payload["assignments"][0].update(instrument_uid="tool", instance_no=1)
        payload["decisions"][0]["status"] = "accepted"
    elif damage == "unknown":
        payload["assignments"][1]["instrument_uid"] = "missing"
    elif damage == "boolean":
        payload["assignments"][1]["instance_no"] = True
    elif damage == "unresolved":
        payload["assignments"][1].update(instrument_uid=None, instance_no=None)
    else:
        payload["assignments"][0]["object_index"] = True
    sample = workflow.samples[0]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    with pytest.raises(CaptureError):
        server._complete_review(payload)

    assert all(item["instrument_uid"] is None and item["dino_status"] == "pending" for item in workflow.reviews.objects(sample, revision))
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_object_assignment_events WHERE revision=?", (revision,)).fetchone()[0] == 0
    assert not (session_database(workflow.root, sample.session_id).parent / "masks").exists()


def test_dino_manual_selection_sql_failure_rolls_back_assignments(workflow: Workflow) -> None:
    """公開反映の失敗時に手動対応・履歴・maskも元へ戻す。

    Args:
        workflow: 一時セッションの保存先。
    """
    revision, payload = _ambiguous_dino(workflow)
    sample = workflow.samples[0]
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection, connection:
        connection.execute("""CREATE TRIGGER reject_manual_handoff BEFORE UPDATE ON sam3_dino_results
            BEGIN SELECT RAISE(ABORT, 'manual test failure'); END""")
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    with pytest.raises(CaptureError, match="manual test failure"):
        server._complete_review(payload)

    assert all(item["instrument_uid"] is None for item in workflow.reviews.objects(sample, revision))
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_object_assignment_events WHERE revision=?", (revision,)).fetchone()[0] == 0
    assert not list(session_database(workflow.root, sample.session_id).parent.glob("masks/*.png"))


def test_dino_only_approval_never_restores_stale_yolo_export(workflow: Workflow) -> None:
    """DINO専用確定後に古いYOLO manifestの同期で承認を復活させない。

    Args:
        workflow: 一時セッションの保存先。
    """
    sample = workflow.samples[0]
    _publish(workflow, sample)
    revision = workflow.revisions[sample.session_id]
    workflow.reviews.record_annotation(revision, read_json(workflow.store.root / "reviews" / revision / "annotation.json"), "revoked")
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    payload = {
        "revision": revision,
        "reviewer": "operator",
        "confirm_complete": True,
        "overwrite_masks": True,
        "decisions": [{"object_index": 0, "status": "accepted"}],
        "assignments": [{"object_index": 0, "instrument_uid": "tool", "instance_no": 1}],
    }

    server._complete_review(payload)
    workflow.store.sync_existing_exports()
    workflow.store.sync_object_reviews()

    assert _flags(workflow, sample) == (1, 0)
    with pytest.raises(CaptureError, match="承認が失効"):
        workflow.store.export_approved(workflow.store.root / "stale-manual-export", sample.session_id)


def test_dino_manual_selection_preserves_current_yolo_publication(workflow: Workflow) -> None:
    """有効なYOLO承認と公開はDINO採否の変更で失効させず、担当者も混同しない。

    Args:
        workflow: 一時セッションの保存先。
    """
    sample = workflow.samples[0]
    manifest = _publish(workflow, sample)
    revision = workflow.revisions[sample.session_id]
    approval = workflow.store.root / "approvals" / f"{revision}.json"
    before = approval.read_bytes()
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review(
        {"revision": revision, "reviewer": "dino-operator", "confirm_complete": True, "overwrite_masks": True, "decisions": [{"object_index": 0, "status": "accepted"}], "assignments": [{"object_index": 0, "instrument_uid": "tool", "instance_no": 1}]}
    )

    assert _flags(workflow, sample) == (1, 1)
    assert approval.read_bytes() == before
    assert manifest.is_file()
    assert workflow.reviews.objects(sample, revision)[0]["reviewer"] == "dino-operator"
    item = next(item for item in server._items() if item["revision"] == revision)
    assert item["approved"]


def test_selection_catalog_marks_dino_approved_with_preserved_yolo_scope(workflow: Workflow) -> None:
    """有効なYOLO承認scopeを維持したDINO保存を選択一覧にも反映する。"""
    sample = workflow.samples[0]
    _publish(workflow, sample)
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._complete_review(
        {
            "revision": revision,
            "reviewer": "dino-operator",
            "confirm_complete": True,
            "overwrite_masks": True,
            "decisions": [{"object_index": 0, "status": "accepted"}],
            "assignments": [{"object_index": 0, "instrument_uid": "tool", "instance_no": 1}],
        }
    )

    assert workflow.reviews.objects(sample, revision)[0]["saved_mask_exists"]
    selection = SelectiveInferenceServer(workflow.store.capture_source, workflow.store, workflow.reviews)
    selected_item = selection._with_status(selection.entry_by_identity[(sample.session_id, sample.image_id)])
    assert selected_item["yolo"] == "承認済み"
    assert selected_item["dino"] == "承認済み"


def test_review_items_capture_date_comes_from_source_path(workflow: Workflow) -> None:
    """時刻だけのセッションIDではなく、撮影元の上位日付を画面へ渡す。"""
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    items = server._items()

    assert {item["capture_date"] for item in items} == {"2026-10-30"}
    assert {item["session_id"] for item in items} == {"session_a", "session_b"}


def test_review_restart_lists_unapproved_images_without_mask_files(workflow: Workflow) -> None:
    """承認前PNGがない状態でも一覧を表示し、確認時だけ再推論を要求する。"""
    restarted = SqlitePredictionStore(workflow.root, workflow.root, workflow.root / "annotations" / "sam3", 1.0)
    reviews = SqliteObjectReviewStore(workflow.root, 1.0, restarted)
    workflow.store.review_store = reviews
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    items = server._items()

    assert len(items) == len(workflow.samples)
    assert all(not obj["saved_mask_exists"] for item in items for obj in item["objects"])
    assert not list(workflow.root.rglob("masks/*.png"))
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    with pytest.raises(CaptureError, match="Missing original mask"):
        server._object_mask(f"/api/mask/{revision}/0")
    assert not workflow.reviews.approved(revision)


def test_review_restart_reads_approved_png_without_model(workflow: Workflow) -> None:
    """承認後は正式PNGだけから確認を再開し、別の原本PNGを要求しない。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    server._complete_review({"revision": revision, "reviewer": "operator", "confirm_complete": True, "decisions": [{"object_index": 0, "status": "accepted"}]})
    restarted = SqlitePredictionStore(workflow.root, workflow.root, workflow.root / "annotations" / "sam3", 1.0)
    reviews = SqliteObjectReviewStore(workflow.root, 1.0, restarted)

    png = reviews.mask_png(sample, revision, 0)

    assert png.startswith(b"\x89PNG")
    assert reviews.objects(sample, revision)[0]["saved_mask_exists"]
    assert not list((workflow.root / "annotations").rglob("*.png"))


def test_session_outputs_prediction_details_and_initial_flags(workflow: Workflow) -> None:
    """各セッションに推論条件・原本を保存し、初期公開を拒否する。"""
    for sample in workflow.samples:
        with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM sam3_dino_results").fetchone()
            assert row["session_id"] == sample.session_id
            assert (row["prompt"], row["min_score"], row["sam3_score"]) == ("metal tool", 0.5, 0.9)
            assert json.loads(row["bbox_json"]) == [2.0, 2.0, 6.0, 6.0]
            assert row["mask_path"].startswith("memory:")
            assert not list(workflow.root.rglob("masks/*.png"))
            assert row["review_status"] == "pending"
            assert connection.execute("SELECT COUNT(*) FROM sam3_predictions").fetchone()[0] == 1
        assert _flags(workflow, sample) == (0, 0)
    assert not (workflow.root / "metadata.sqlite3").exists()


def test_sync_object_reviews_migrates_unmodified_legacy_initial_annotation(workflow: Workflow) -> None:
    """レビュー起動時の同期でも、未編集の旧初期草案を移行する。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    annotation_path = workflow.store.root / "reviews" / revision / "annotation.json"
    annotation = read_json(annotation_path)
    annotation["shapes"].extend(annotation.pop("review_candidates"))
    annotation.pop("tag_review_version")
    for shape in annotation["shapes"]:
        shape.pop("tag_assignment")
    atomic_json(annotation_path, annotation)

    workflow.store.sync_object_reviews()

    migrated = read_json(annotation_path)
    assert migrated["tag_review_version"] == 1
    assert len(migrated["shapes"]) == 1
    assert migrated["shapes"][0]["tag_assignment"] == {"instrument_uid": "tool", "instance_no": 1}


def test_sync_object_reviews_preserves_edited_legacy_initial_annotation(workflow: Workflow) -> None:
    """旧初期草案でも人手編集があればレビュー起動時に変更しない。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    annotation_path = workflow.store.root / "reviews" / revision / "annotation.json"
    annotation = read_json(annotation_path)
    annotation["shapes"].extend(annotation.pop("review_candidates"))
    annotation.pop("tag_review_version")
    for shape in annotation["shapes"]:
        shape.pop("tag_assignment")
    annotation["shapes"][0]["points"][0][0] = 1.0
    atomic_json(annotation_path, annotation)

    workflow.store.sync_object_reviews()

    saved = read_json(annotation_path)
    assert "tag_review_version" not in saved
    assert saved["shapes"][0]["points"][0][0] == 1.0


@pytest.mark.parametrize("target", ["dino", "yolo"])
def test_session_outputs_publication_is_independent(workflow: Workflow, target: str) -> None:
    """DINO公開とYOLO公開は別条件であり、他セッションへ波及しない。"""
    first, second = workflow.samples
    revision = workflow.revisions[first.session_id]
    workflow.store.approve(revision, "operator", confirm_complete=True)
    assert _flags(workflow, first) == (0, 0)

    if target == "dino":
        server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
        server._complete_review({"revision": revision, "reviewer": "operator", "confirm_complete": True, "decisions": [{"object_index": 0, "status": "accepted"}]})
    else:
        workflow.store.export_approved(workflow.store.root / "exports" / first.session_id, first.session_id)

    assert _flags(workflow, first) == ((1, 0) if target == "dino" else (0, 1))
    assert _flags(workflow, second) == (0, 0)


def test_session_outputs_yolo_export_preserves_contract(workflow: Workflow) -> None:
    """従来のmanifestと注釈JSONを維持し、画像を公開先へ複製しない。"""
    sample = workflow.samples[0]

    manifest = _publish(workflow, sample)

    record = read_json(manifest)["samples"][0]
    assert record["source_path"] == sample.source_path
    assert (manifest.parent / record["annotation"]).is_file()
    assert not list(manifest.parent.rglob("*.png"))
    assert _flags(workflow, sample) == (1, 1)
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        row = connection.execute("SELECT annotation_json, annotation_sha256, allow_empty, exception_note, manifest_path FROM sam3_yolo_results").fetchone()
    assert json.loads(row[0])["shapes"][0]["points"] == [[2.0, 2.0], [6.0, 6.0]]
    assert row[1:] == (record["labelme_sha256"], 0, "", str(manifest.resolve()))


def test_tag_review_manual_box_exports_without_changing_dino_original(workflow: Workflow) -> None:
    """手描きの採用矩形だけを学習注釈へ渡し、DINOの推論原本を維持する。

    Args:
        workflow: 一時DBと実際の保存処理を持つテスト環境。
    """
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    annotation = read_json(workflow.store.root / "reviews" / revision / "annotation.json")
    candidate = annotation["shapes"][0]
    candidate["label"] = "__unresolved__"
    candidate["tag_assignment"] = None
    annotation.update(
        tag_review_version=1,
        review_candidates=[candidate],
        shapes=[
            {
                "label": "tool",
                "points": [[1.0, 1.0], [7.0, 7.0]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "description": json.dumps({"manual": True}),
                "tag_assignment": {"instrument_uid": "tool", "instance_no": 1},
            }
        ],
    )
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    # 画面のAPIと同じ保存境界を通し、再読込・承認・公開までを確認する。
    server._save_annotation({"revision": revision, "annotation": annotation})
    reloaded = next(item for item in server._items() if item["revision"] == revision)
    workflow.store.approve(revision, "operator", confirm_complete=True)
    manifest = workflow.store.export_approved(workflow.store.root / "tag-export", sample.session_id)
    record = read_json(manifest)["samples"][0]
    exported = read_json(manifest.parent / record["annotation"])

    assert reloaded["annotation"] == annotation
    assert len(exported["shapes"]) == 1
    assert exported["shapes"][0]["points"] == [[1.0, 1.0], [7.0, 7.0]]
    assert len(exported["review_candidates"]) == 1
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        original = connection.execute("SELECT bbox_json FROM sam3_prediction_objects").fetchone()[0]
        assert json.loads(original) == [2.0, 2.0, 6.0, 6.0]
    assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "pending"


def test_tag_review_duplicate_assignment_cannot_be_saved_or_approved(workflow: Workflow) -> None:
    """保存・直接承認のどちらでも同一タグへの複数bbox採用を拒否する。

    Args:
        workflow: 一時DBと実際の保存処理を持つテスト環境。
    """
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    path = workflow.store.root / "reviews" / revision / "annotation.json"
    annotation = read_json(path)
    annotation.update(tag_review_version=1, review_candidates=[])
    first = annotation["shapes"][0]
    first["tag_assignment"] = {"instrument_uid": "tool", "instance_no": 1}
    annotation["shapes"].append({**first, "group_id": None, "points": [[1.0, 1.0], [7.0, 7.0]]})
    before = path.read_bytes()
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    with pytest.raises(CaptureError, match="複数のbbox"):
        server._save_annotation({"revision": revision, "annotation": annotation})
    assert path.read_bytes() == before
    # ファイルを直接編集されても、承認時の独立した検証を省略しない。
    atomic_json(path, annotation)
    with pytest.raises(CaptureError, match="複数のbbox"):
        workflow.store.approve(revision, "operator", confirm_complete=True)


@pytest.mark.parametrize("change", ["tag_revision", "untagged", "hash", "path", "dimensions", "delete"])
def test_session_outputs_capture_changes_invalidate_immediately(workflow: Workflow, change: str) -> None:
    """撮影側からの更新でもSAM3の再起動を待たずに両公開フラグを失効する。"""
    sample = workflow.samples[0]
    _publish(workflow, sample)
    statements = {
        "tag_revision": "UPDATE images SET tag_revision=2",
        "untagged": "UPDATE images SET tag_status='untagged'",
        "hash": "UPDATE images SET sha256='changed'",
        "path": "UPDATE images SET path='changed'",
        "dimensions": "UPDATE images SET width=1",
        "delete": "DELETE FROM images",
    }

    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection, connection:
        connection.execute(statements[change])

    assert _flags(workflow, sample) == (0, 0)


def test_session_outputs_revocation_retains_history(workflow: Workflow) -> None:
    """取消は公開を停止し、再承認だけで物体採用やYOLO公開を復活させない。"""
    sample = workflow.samples[0]
    _publish(workflow, sample)
    revision = workflow.revisions[sample.session_id]

    workflow.store.revoke(revision)
    assert _flags(workflow, sample) == (0, 0)
    workflow.store.approve(revision, "operator", confirm_complete=True)

    assert _flags(workflow, sample) == (0, 0)
    assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "pending"


def test_session_outputs_yolo_edit_preserves_dino_original(workflow: Workflow) -> None:
    """矩形編集で画像承認を失効するが、推論bbox・maskは不変に保つ。"""
    sample = workflow.samples[0]
    _publish(workflow, sample)
    revision = workflow.revisions[sample.session_id]
    before = {path: digest_file(path) for path in workflow.root.rglob("*.png")}
    annotation = read_json(workflow.store.root / "reviews" / revision / "annotation.json")
    annotation["shapes"][0]["points"][0][0] = 1.0
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)

    server._save_annotation({"revision": revision, "annotation": annotation})
    assert _flags(workflow, sample) == (0, 0)
    workflow.store.approve(revision, "operator", confirm_complete=True)

    assert _flags(workflow, sample) == (1, 0)
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert json.loads(connection.execute("SELECT bbox_json FROM sam3_dino_results").fetchone()[0])[0] == ORIGINAL_LEFT
        assert json.loads(connection.execute("SELECT annotation_json FROM sam3_yolo_results").fetchone()[0])["shapes"][0]["points"][0][0] == 1.0
    assert {path: digest_file(path) for path in before} == before


def test_session_outputs_new_revision_disables_old_results(workflow: Workflow) -> None:
    """新しい対応付け版の作成時は旧版の両公開フラグを無効にする。"""
    sample = workflow.samples[0]
    _publish(workflow, sample)
    revision = workflow.revisions[sample.session_id]
    metadata = read_json(workflow.store.root / "drafts" / revision / "metadata.json")
    metadata["matching_signature"] = "new-matching"
    annotation = read_json(workflow.store.root / "reviews" / revision / "annotation.json")

    workflow.reviews.sync_revision(sample, content_hash(metadata), metadata, annotation)

    assert _flags(workflow, sample) == (0, 0)


@pytest.mark.parametrize("old_database", [False, True])
def test_session_outputs_restart_restores_both_publications(workflow: Workflow, monkeypatch: pytest.MonkeyPatch, old_database: bool) -> None:
    """CLIのreview再起動でも中央DBなしに公開状態を復元する。"""
    if old_database:
        # 残存する旧単一DBの破損は、現行のセッションDB運用へ影響させない。
        (workflow.root / "metadata.sqlite3").write_bytes(b"ignored old database")
    for sample in workflow.samples:
        _publish(workflow, sample)
    config = workflow.store.root.parent / "config.json"
    atomic_json(config, {"capture_root": str(workflow.root), "output_root": str(workflow.store.root), "sqlite_timeout": 1.0, "review_confidence": {"medium_score": 0.65, "high_score": 0.85}})
    monkeypatch.setattr(BatchReviewServer, "serve", lambda _self: None)

    result = main(["--config", str(config), "review"])

    assert result == 0
    assert all(_flags(workflow, sample) == (1, 1) for sample in workflow.samples)
    if old_database:
        assert (workflow.root / "metadata.sqlite3").read_bytes() == b"ignored old database"
    else:
        assert not (workflow.root / "metadata.sqlite3").exists()


def test_session_outputs_sql_revocation_survives_stale_approval_json(workflow: Workflow) -> None:
    """古い承認JSONが残ってもSQLの取消状態を正本として維持する。"""
    sample = workflow.samples[0]
    _publish(workflow, sample)
    revision = workflow.revisions[sample.session_id]
    annotation = read_json(workflow.store.root / "reviews" / revision / "annotation.json")
    workflow.reviews.record_annotation(revision, annotation, "revoked")

    workflow.store.sync_object_reviews()
    workflow.store.sync_existing_exports()

    assert _flags(workflow, sample) == (0, 0)
    with pytest.raises(CaptureError, match="承認が失効"):
        workflow.store.export_approved(workflow.store.root / "stale-export", sample.session_id)


def test_session_outputs_publication_failure_rolls_back_review(workflow: Workflow) -> None:
    """専用テーブルへの反映失敗時は採否とイベント履歴もロールバックする。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    workflow.store.approve(revision, "operator", confirm_complete=True)
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection, connection:
        connection.execute("""CREATE TRIGGER reject_handoff BEFORE UPDATE ON sam3_dino_results
            BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")

    with pytest.raises(CaptureError, match="simulated failure"):
        workflow.reviews.decide(sample, revision, 0, status="accepted", reviewer="operator")

    assert _flags(workflow, sample) == (0, 0)
    assert workflow.reviews.objects(sample, revision)[0]["dino_status"] == "pending"
    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_object_review_events").fetchone()[0] == 0


def test_session_outputs_empty_prediction_allows_explicit_yolo_negative(workflow: Workflow) -> None:
    """検出0件を保持し、明示承認した負例をYOLOだけへ公開する。"""
    sample = workflow.samples[0]
    signature = "empty-model"
    prediction_id = hashlib.sha256(f"{sample.sha256}:{signature}".encode()).hexdigest()
    workflow.predictions.save(sample, prediction_id, (), signature)
    revision = workflow.store.save_draft(sample, (), MatchResult((), ()), prediction_id, "empty-matching")
    workflow.store.approve(revision, "operator", confirm_complete=True, allow_empty=True, exception_note="器具なしの負例として確認済み")

    workflow.store.export_approved(workflow.store.root / "empty-export", sample.session_id)

    with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_dino_results WHERE revision=?", (revision,)).fetchone()[0] == 0
        assert connection.execute("SELECT allow_empty, is_yolo_ready FROM sam3_yolo_results WHERE revision=?", (revision,)).fetchone() == (1, 1)


def test_session_outputs_missing_database_never_recreates(workflow: Workflow) -> None:
    """削除されたセッションDBへ保存せず、他DBや旧単一DBへ退避しない。"""
    sample = workflow.samples[0]
    database = session_database(workflow.root, sample.session_id)
    database.unlink()

    with pytest.raises(CaptureError, match="セッションDB"):
        workflow.reviews.decide(sample, workflow.revisions[sample.session_id], 0, status="accepted", reviewer="operator")

    assert not database.exists()
    assert not (workflow.root / "metadata.sqlite3").exists()


def test_session_outputs_cli_inference_uses_session_databases(workflow: Workflow, monkeypatch: pytest.MonkeyPatch) -> None:
    """通常runは選択画像だけを各セッションの専用テーブルへ保存する。"""
    signature = "cli-native-model"
    pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    pixels[2:6, 2:6] = True
    candidates = (Candidate(0, (2.0, 2.0, 6.0, 6.0), 0.9, ArrayMask(pixels)),)

    class Predictor:
        """GPU依存を置き換え、通常runの保存経路だけを通す。"""

        def __init__(self) -> None:
            """推論キーに使用するモデル署名を設定する。"""
            self.signature: str = signature

        def initialize(self) -> None:
            """実モデルの事前初期化APIをテスト用予測器に合わせる。"""

        def infer(self, _sample: CaptureSample) -> tuple[Candidate, ...]:
            """入力画像へ確定済みの候補を返す。

            Args:
                _sample: 推論対象画像。

            Returns:
                原本保存を検証するダミー候補。
            """
            return candidates

    def create_predictor(checkpoint: Path | None, prompt: str, min_score: float, bpe_path: Path | None) -> Predictor:
        """モデル生成時に設定が引き渡されることを確認する。

        Args:
            checkpoint: 任意の重みパス。
            prompt: 全画像共通プロンプト。
            min_score: 推論スコア閾値。
            bpe_path: 任意の語彙パス。

        Returns:
            実モデルをロードしない推論代替。
        """
        assert checkpoint is None and bpe_path is None
        assert (prompt, min_score) == ("instrument", 0.45)
        return Predictor()

    config = workflow.store.root.parent / "run-config.json"
    atomic_json(
        config,
        {
            "capture_root": str(workflow.root),
            "output_root": str(workflow.store.root),
            "prompt": "instrument",
            "min_score": 0.45,
            "matching": {"max_distance": 0.03, "ambiguity_margin": 0.05},
            "review_confidence": {"medium_score": 0.65, "high_score": 0.85},
        },
    )
    monkeypatch.setattr(capture_workflow, "Sam3CapturePredictor", create_predictor)
    chosen = workflow.samples[0]
    def select_chosen_image(selection: SelectiveInferenceServer) -> list[tuple[str, str]]:
        selection.url = "http://127.0.0.1:12345/"
        return [(chosen.session_id, chosen.image_id)]

    monkeypatch.setattr(capture_workflow.SelectiveInferenceServer, "select", select_chosen_image)
    monkeypatch.setattr(BatchReviewServer, "serve", lambda _self, **_kwargs: None)

    result = main(["--config", str(config), "run"])

    assert result == 0
    for sample in workflow.samples:
        with closing(sqlite3.connect(session_database(workflow.root, sample.session_id))) as connection:
            actual = connection.execute("SELECT prompt, min_score, is_dino_ready FROM sam3_dino_results WHERE model_signature=?", (signature,)).fetchall()
            expected = [("instrument", 0.45, 0)] if sample == chosen else []
            assert actual == expected
    assert not (workflow.root / "metadata.sqlite3").exists()


def test_review_page_embeds_live_selection_return_url(workflow: Workflow) -> None:
    """runレビュー画面だけに稼働中selection serverのURLを注入する。"""
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    server.selection_return_url = "http://127.0.0.1:12345/?review_complete=1"

    page = server._review_page()

    assert b"window.selectionReturnUrl = \"http://127.0.0.1:12345/?review_complete=1\"" in page


def test_review_server_partial_publish_keeps_yolo_independent_of_dino_and_edit_clears(workflow: Workflow) -> None:
    """DINO部分保存はYOLO確認・公開を変えず、YOLO編集は確認と承認表示を解除する。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    server._set_reviewed({"revision": revision, "reviewed": True})
    manifest = server._publish({"reviewer": "operator", "session_id": sample.session_id, "confirm_complete": True})
    before = manifest.read_bytes()

    server._complete_review({"revision": revision, "reviewer": "operator", "confirm_complete": True,
                             "decisions": [{"object_index": 0, "status": "accepted"}],
                             "assignments": [{"object_index": 0, "instrument_uid": "tool", "instance_no": 1}]})

    item = next(item for item in server._items() if item["revision"] == revision)
    assert item["reviewed"] and item["approved"] and item["published"]
    assert item["dino_approved"]
    assert item["objects"][0]["dino_status"] == "accepted"
    assert item["objects"][0]["saved_mask_exists"]
    assert _flags(workflow, sample)[1] == 1
    assert manifest.read_bytes() == before
    workflow.store.sync_object_reviews()
    item = next(item for item in server._items() if item["revision"] == revision)
    assert item["dino_approved"]
    assert item["objects"][0]["dino_status"] == "accepted"
    assert item["objects"][0]["saved_mask_exists"]
    annotation = json.loads((workflow.store.root / "reviews" / revision / "annotation.json").read_text(encoding="utf-8"))
    annotation["shapes"][0]["points"][1][0] -= 1
    server._save_annotation({"revision": revision, "annotation": annotation})
    item = next(item for item in server._items() if item["revision"] == revision)
    assert not item["reviewed"] and not item["approved"] and not item["published"]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("label", "not-a-declared-uid"), ("shape_type", "polygon")],
)
def test_batch_publish_error_identifies_image_shape_and_field(
    workflow: Workflow,
    field: str,
    invalid_value: str,
) -> None:
    """公開検証エラーに画像ID、shape index、違反フィールドを含め、承認前に停止する。"""
    sample = workflow.samples[0]
    revision = workflow.revisions[sample.session_id]
    server = BatchReviewServer(workflow.store, ConfidenceThresholds(0.65, 0.85), workflow.samples)
    server._set_reviewed({"revision": revision, "reviewed": True})

    annotation_path = workflow.store.root / "reviews" / revision / "annotation.json"
    annotation = read_json(annotation_path)
    annotation["shapes"][0][field] = invalid_value
    atomic_json(annotation_path, annotation)

    with pytest.raises(
        CaptureError,
        match=rf"image ID {sample.image_id}: shape index 0 field '{field}'",
    ):
        server._publish({
            "reviewer": "operator",
            "session_id": sample.session_id,
            "confirm_complete": True,
        })

    assert not workflow.reviews.approved(revision, scope="yolo")
    assert not (workflow.store.root / "exports").exists()
