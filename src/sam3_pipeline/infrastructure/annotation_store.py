"""撮影原本を複製せず、推論キャッシュと人手レビューの記録を別領域に保持する。

JSONとNPZを成果物形式として使い、NPZはpickleを無効にして読む。レビュー修正を推論再実行で
上書きせず、承認・公開時もSQL上の撮影原本を正本として照合する。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sam3_pipeline.domain.annotation import RECTANGLE, UNRESOLVED_LABEL, Candidate, MatchResult, ObjectReviewStore, PredictionStore
from sam3_pipeline.domain.capture import SCHEMA_VERSION, AnnotationLedger, CaptureError, CaptureSample, CaptureSource
from sam3_pipeline.infrastructure.matching import ArrayMask
from sam3_pipeline.infrastructure.tag_annotations import TAG_REVIEW_VERSION, validate_tag_review

CACHE_DIR = "cache"
DRAFTS_DIR = "drafts"
REVIEWS_DIR = "reviews"
APPROVALS_DIR = "approvals"
LATEST_DIR = "latest"
METADATA_FILE = "metadata.json"
ANNOTATION_FILE = "annotation.json"
LABELME_VERSION = "5.5.0"
ENCODING = "utf-8"
SHA256_LENGTH = 64
BOX_POINT_COUNT = 2
POINT_COORDINATE_COUNT = 2


def digest_file(path: Path) -> str:
    """Hash file contents for provenance and mutation checks.

    Args:
        path: Existing artifact.

    Returns:
        SHA-256 digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(data: Any) -> str:
    """Hash canonical structured data rather than formatting-dependent JSON text.

    Args:
        data: JSON-compatible data, with paths encoded as strings.

    Returns:
        Stable SHA-256 digest.
    """
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True, allow_nan=False, default=str).encode(ENCODING)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject malformed artifact roots.

    Args:
        path: Artifact path.

    Returns:
        Parsed JSON object.

    Raises:
        CaptureError: The document is not a JSON object.
    """
    data = json.loads(path.read_text(encoding=ENCODING))
    if not isinstance(data, dict):
        raise CaptureError(f"Expected a JSON object: {path}")
    return data


def atomic_json(path: Path, data: Any) -> None:
    """Publish complete JSON files through same-directory atomic replacement.

    Args:
        path: Destination artifact.
        data: JSON-compatible data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding=ENCODING, dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, ensure_ascii=True, allow_nan=False, default=str)
    temporary.replace(path)


def validate_rectangles(data: dict[str, Any], allowed_uids: set[str], width: int, height: int, *, allow_empty: bool = False, allow_unresolved: bool = False) -> list[dict[str, Any]]:
    """Validate the detection-only Labelme contract before approval or training.

    Args:
        data: Labelme document.
        allowed_uids: Stable instrument identities allowed in this image.
        width: Original RGB width.
        height: Original RGB height.
        allow_empty: Explicit confirmation that an image is a true negative.
        allow_unresolved: Permit unresolved draft labels before final approval.

    Returns:
        Validated rectangle shapes.

    Raises:
        CaptureError: A label, geometry, duplicate, or image dimension is invalid.
    """
    if data.get("imageWidth") != width or data.get("imageHeight") != height:
        raise CaptureError("Labelme dimensions differ from the captured image")
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or (not shapes and not allow_empty):
        raise CaptureError("Empty annotations require explicit true-negative approval")
    groups: set[int] = set()
    boxes: set[str] = set()
    for shape_index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise CaptureError(f"shape index {shape_index} field 'shape' must be an object")
        label = shape.get("label")
        is_unresolved = label == UNRESOLVED_LABEL
        if is_unresolved and not allow_unresolved:
            raise CaptureError(f"shape index {shape_index} field 'label' is unresolved")
        if not is_unresolved and (not isinstance(label, str) or label not in allowed_uids):
            raise CaptureError(f"shape index {shape_index} field 'label' must be a declared instrument UID")
        shape_type = shape.get("shape_type")
        if shape_type != RECTANGLE:
            raise CaptureError(f"shape index {shape_index} field 'shape_type' must be '{RECTANGLE}'")
        points = shape.get("points", [])
        has_two_corners = isinstance(points, list) and len(points) == BOX_POINT_COUNT
        has_valid_corner_lengths = has_two_corners and all(
            isinstance(point, list) and len(point) == POINT_COORDINATE_COUNT
            for point in points
        )
        if not has_valid_corner_lengths:
            raise CaptureError(f"shape index {shape_index} field 'points' must contain exactly two corners")
        has_finite_coordinates = all(
            isinstance(value, (float, int)) and math.isfinite(value)
            for point in points
            for value in point
        )
        if not has_finite_coordinates:
            raise CaptureError(f"shape index {shape_index} field 'points' must contain finite coordinates")
        (first_x, first_y), (second_x, second_y) = points
        is_inside_image = (
            0 <= min(first_x, second_x) < max(first_x, second_x) <= width
            and 0 <= min(first_y, second_y) < max(first_y, second_y) <= height
        )
        if not is_inside_image:
            raise CaptureError(f"shape index {shape_index} field 'points' must define positive area inside the image")
        group = shape.get("group_id")
        if group is not None:
            if not isinstance(group, int) or group in groups:
                raise CaptureError(f"shape index {shape_index} field 'group_id' is duplicated or invalid")
            groups.add(group)
        box = content_hash([label, min(first_x, second_x), min(first_y, second_y), max(first_x, second_x), max(first_y, second_y)])
        if box in boxes:
            raise CaptureError(f"shape index {shape_index} duplicates an annotation rectangle")
        boxes.add(box)
    return shapes


class FileAnnotationStore:
    """注釈・承認の履歴を保存し、撮影画像はSQL正本から参照する。

    Args:
        root: 撮影原本と分離したアノテーション出力ルート。
        capture_source: 読み取り専用の撮影SQLアダプター。
        ledger: 既存YOLO互換の画像承認台帳。
        review_store: 推論原本とは分離したSQLレビュー・物体採否の保存先。

    Note:
        出力ルートへの書き込みは単一プロセスに限定する。承認後の注釈変更には再承認が必要である。
    """

    def __init__(self, root: Path, capture_source: CaptureSource, ledger: AnnotationLedger | None = None, *, review_store: ObjectReviewStore | None = None) -> None:
        """成果物領域とSQL正本への読み取り専用ポートを結び付ける。"""
        self.root = root.resolve()
        self.capture_source = capture_source
        self.ledger = ledger
        self.review_store = review_store

    def load_candidates(self, key: str) -> tuple[Candidate, ...] | None:
        """Read a completed inference cache without executing serialized Python.

        Args:
            key: Prediction fingerprint.

        Returns:
            Cached candidates, including an empty tuple for zero detections.
        """
        path = self.root / CACHE_DIR / f"{self._key(key)}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            return tuple(Candidate(item["index"], tuple(item["box"]), item["score"], ArrayMask(archive[f"mask_{item['index']}"])) for item in metadata)

    def save_candidates(self, key: str, candidates: tuple[Candidate, ...]) -> None:
        """Publish masks and boxes atomically, preserving zero-detection cache entries.

        Args:
            key: Prediction fingerprint.
            candidates: Validated model candidates using array-backed masks.

        Raises:
            CaptureError: The mask adapter is incompatible with NPZ persistence.
        """
        path = self.root / CACHE_DIR / f"{self._key(key)}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        metadata = []
        for candidate in candidates:
            if not isinstance(candidate.mask, ArrayMask):
                raise CaptureError("Cache requires ArrayMask adapters")
            arrays[f"mask_{candidate.index}"] = candidate.mask.pixels
            metadata.append({"index": candidate.index, "box": candidate.box, "score": candidate.score})
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, allow_pickle=False, metadata=np.array(json.dumps(metadata)), **arrays)
        temporary.replace(path)

    def migrate_predictions(self, target: PredictionStore) -> int:
        """旧草案が参照するNPZを、撮影画像を照合して新保存先へ非破壊移行する。

        Args:
            target: PNGとSQLを正本とする推論保存先。

        Returns:
            同一画像・同一推論を重複排除した処理件数。移行済みも含む。

        Raises:
            CaptureError: 対象キャッシュが欠落・破損しているか、保存に失敗した場合。

        Note:
            タグ変更前の推論も画像hashが同一なら移行できる。承認やDINO採否は変更しない。
            旧NPZには元の推論署名がないため、不明として保持し推測で復元しない。
        """
        migrated: set[tuple[str, str, str]] = set()
        try:
            for path in sorted((self.root / DRAFTS_DIR).glob(f"*/{METADATA_FILE}")):
                metadata = read_json(path)
                source = metadata["source"]
                sample = self.capture_source.current_sample(source["session_id"], source["image_id"], source["source_path"])
                if sample is None or (sample.sha256, sample.width, sample.height) != (source["sha256"], source["width"], source["height"]):
                    continue
                key = self._key(metadata["prediction_key"])
                identity = (sample.session_id, sample.image_id, key)
                if identity in migrated:
                    continue
                if target.load(sample, key) is None:
                    candidates = self.load_candidates(key)
                    if candidates is None:
                        raise CaptureError(f"Legacy prediction cache is missing: {key}")
                    target.save(sample, key, candidates, None)
                migrated.add(identity)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CaptureError(f"Invalid legacy prediction cache: {error}") from error
        return len(migrated)

    def save_draft(self, sample: CaptureSample, candidates: tuple[Candidate, ...], result: MatchResult, prediction_key: str, matching_signature: str) -> str:
        """Create one editable workspace per source and matching revision.

        Args:
            sample: Source snapshot.
            candidates: Cached model output.
            result: Tag correspondence and uncertainty.
            prediction_key: Model output fingerprint.
            matching_signature: Matching configuration fingerprint.

        Returns:
            Stable revision ID; existing edits are preserved and only an exact initial legacy draft may migrate.

        Raises:
            CaptureError: Source image contents or dimensions have changed.
        """
        source = self._sample_record(sample)
        metadata = {"schema_version": SCHEMA_VERSION, "source": source, "prediction_key": prediction_key, "matching_signature": matching_signature, "matching": asdict(result)}
        revision = content_hash(metadata)
        draft_dir = self.root / DRAFTS_DIR / revision
        review_dir = self.root / REVIEWS_DIR / revision
        if (draft_dir / METADATA_FILE).exists():
            if not (review_dir / ANNOTATION_FILE).exists():
                raise CaptureError(f"Review workspace is missing: {revision}")
            annotation_path = review_dir / ANNOTATION_FILE
            annotation = read_json(annotation_path)
            was_migrated = False
            if "tag_review_version" not in annotation:
                local_progress_path = self.root / "review_progress.json"
                local_progress = read_json(local_progress_path) if local_progress_path.exists() else {}
                persisted_progress = self.review_store.progress() if self.review_store is not None else {}
                if self._can_migrate_initial_annotation(revision, sample, local_progress.get(revision), persisted_progress):
                    candidate_records = [{"index": candidate.index, "box": candidate.box, "score": candidate.score} for candidate in candidates]
                    upgraded = self._upgrade_initial_annotation(sample, annotation, metadata, candidate_records)
                    if upgraded is not None:
                        atomic_json(annotation_path, upgraded)
                        annotation = upgraded
                        was_migrated = True
            self._mark_latest(sample, revision)
            if self.review_store is not None:
                self.review_store.sync_revision(sample, revision, metadata, annotation)
            if was_migrated:
                self._record_annotation(source, revision, "inferred", annotation_path)
            return revision
        if not sample.source_path or digest_file(sample.image_path) != sample.sha256:
            raise CaptureError("Source image changed while processing")
        with Image.open(sample.image_path) as image:
            if image.size != (sample.width, sample.height):
                raise CaptureError("Source image dimensions differ from capture metadata")
        review_dir.mkdir(parents=True, exist_ok=False)
        assignments = {assignment.candidate_index: assignment for assignment in result.assignments}
        shapes: list[dict[str, Any]] = []
        review_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            assignment = assignments.get(candidate.index)
            xmin, ymin, xmax, ymax = candidate.box
            shape = {
                "label": assignment.instrument_uid if assignment else UNRESOLVED_LABEL,
                "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": candidate.index,
                "shape_type": RECTANGLE,
                "flags": {"needs_review": True},
                "description": json.dumps({"score": candidate.score, "instance_no": assignment.instance_no if assignment else None}),
                "tag_assignment": {"instrument_uid": assignment.instrument_uid, "instance_no": assignment.instance_no} if assignment else None,
            }
            if assignment:
                shapes.append(shape)
            else:
                review_candidates.append(shape)
        labelme = {
            "version": LABELME_VERSION,
            "flags": {"needs_review": True},
            "tag_review_version": TAG_REVIEW_VERSION,
            "shapes": shapes,
            "review_candidates": review_candidates,
            "imagePath": sample.source_path,
            "imageData": None,
            "imageWidth": sample.width,
            "imageHeight": sample.height,
        }
        atomic_json(review_dir / ANNOTATION_FILE, labelme)
        atomic_json(draft_dir / METADATA_FILE, metadata)
        self._mark_latest(sample, revision)
        if self.review_store is not None:
            self.review_store.sync_revision(sample, revision, metadata, labelme)
        self._record_annotation(source, revision, "inferred", review_dir / ANNOTATION_FILE)
        return revision

    def sync_object_reviews(
        self,
        identities: set[tuple[str, str]] | None = None,
        source_paths: set[str] | None = None,
    ) -> int:
        """既存草案をSQLへ同期し、未編集の初期旧注釈だけをタグレビュー形式へ移行する。

        Returns:
            同期した現行レビュー版の数。

        Raises:
            CaptureError: 原本・承認の不整合、または保存失敗が発生した場合。
        """
        if self.review_store is None:
            return 0
        progress_path = self.root / "review_progress.json"
        progress = read_json(progress_path) if progress_path.exists() else {}
        persisted_progress = self.review_store.progress()
        synced = 0
        for path in sorted((self.root / DRAFTS_DIR).glob(f"*/{METADATA_FILE}")):
            metadata = read_json(path)
            source = metadata["source"]
            if identities is not None and (source["session_id"], source["image_id"]) not in identities:
                continue
            if source_paths is not None and source["source_path"] not in source_paths:
                continue
            if not self._source_is_current(source):
                continue
            revision = self._key(path.parent.name)
            if not self.is_latest_revision(source, revision):
                continue
            annotation_path = self.root / REVIEWS_DIR / revision / ANNOTATION_FILE
            sample = self.current_sample(source)
            annotation = read_json(annotation_path)
            self.review_store.sync_revision(sample, revision, metadata, annotation)
            was_migrated = False
            is_legacy_annotation = "tag_review_version" not in annotation
            can_migrate = is_legacy_annotation and self._can_migrate_initial_annotation(revision, sample, progress.get(revision), persisted_progress)
            if can_migrate:
                candidate_records = [
                    {"index": item["object_index"], "box": item["bbox"], "score": item["sam3_score"]}
                    for item in self.review_store.objects(sample, revision)
                ]
                upgraded = self._upgrade_initial_annotation(sample, annotation, metadata, candidate_records)
                if upgraded is not None:
                    atomic_json(annotation_path, upgraded)
                    annotation = upgraded
                    self.review_store.sync_revision(sample, revision, metadata, annotation)
                    was_migrated = True
            if not persisted_progress.get(revision):
                state = progress.get(revision, {})
                self.review_store.set_progress(revision, {"reviewed": state} if isinstance(state, bool) else state)
            approval_path = self.root / APPROVALS_DIR / f"{revision}.json"
            if approval_path.exists():
                approval = read_json(approval_path)
                if digest_file(annotation_path) == approval["labelme_sha256"]:
                    self._validate_approval(approval)
                    # 既にSQLへ記録済みの取消・編集状態を、残存JSONの承認で復活させない。
                    if revision not in persisted_progress:
                        self.review_store.record_annotation(revision, read_json(annotation_path), "approved", approval)
                else:
                    self.review_store.record_annotation(revision, read_json(annotation_path), "inferred")
            if was_migrated:
                self._record_annotation(source, revision, "inferred", annotation_path)
            synced += 1
        return synced

    def _can_migrate_initial_annotation(
        self,
        revision: str,
        sample: CaptureSample,
        local_progress: Any,
        persisted_progress: dict[str, dict[str, Any]],
    ) -> bool:
        """未確認・未承認の版だけを初期注釈の安全な移行対象にする。"""
        if (self.root / APPROVALS_DIR / f"{revision}.json").exists():
            return False
        if self._was_reviewed(local_progress):
            return False
        if self.review_store is None or revision not in persisted_progress:
            return True
        if self._was_reviewed(persisted_progress[revision]):
            return False
        if self.review_store.approved(revision, scope="yolo") or self.review_store.approved(revision, scope="dino"):
            return False
        return all(item["dino_status"] == "pending" for item in self.review_store.objects(sample, revision))

    @staticmethod
    def _was_reviewed(state: Any) -> bool:
        """旧形式の真偽値と現行の進捗objectから確認状態を読む。"""
        if isinstance(state, bool):
            return state
        return isinstance(state, dict) and state.get("reviewed") is True

    @staticmethod
    def _upgrade_initial_annotation(
        sample: CaptureSample,
        annotation: dict[str, Any],
        metadata: dict[str, Any],
        candidate_records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """保存内容が初期生成物と完全一致する旧注釈だけをタグレビュー形式へ移す。"""
        if "tag_review_version" in annotation:
            return None
        assignments = {item["candidate_index"]: item for item in metadata["matching"]["assignments"]}
        expected = FileAnnotationStore._legacy_initial_annotation(sample, candidate_records, assignments)
        try:
            saved_json = json.dumps(annotation, sort_keys=True, allow_nan=False)
            expected_json = json.dumps(expected, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return None
        if saved_json != expected_json:
            return None
        return FileAnnotationStore._versioned_initial_annotation(annotation, assignments)

    @staticmethod
    def _legacy_initial_annotation(
        sample: CaptureSample,
        candidate_records: list[dict[str, Any]],
        assignments: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """再構成した旧初期注釈を手元のファイルとの厳密比較に使う。"""
        shapes = []
        for candidate in candidate_records:
            index = candidate["index"]
            assignment = assignments.get(index)
            xmin, ymin, xmax, ymax = candidate["box"]
            shapes.append(
                {
                    "label": assignment["instrument_uid"] if assignment is not None else UNRESOLVED_LABEL,
                    "points": [[xmin, ymin], [xmax, ymax]],
                    "group_id": index,
                    "shape_type": RECTANGLE,
                    "flags": {"needs_review": True},
                    "description": json.dumps({"score": candidate["score"], "instance_no": assignment["instance_no"] if assignment is not None else None}),
                }
            )
        return {
            "version": LABELME_VERSION,
            "flags": {"needs_review": True},
            "shapes": shapes,
            "imagePath": sample.source_path,
            "imageData": None,
            "imageWidth": sample.width,
            "imageHeight": sample.height,
        }

    @staticmethod
    def _versioned_initial_annotation(
        annotation: dict[str, Any],
        assignments: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """未編集の旧shapeを内容を保ったまま採用shapeとレビュー候補に分ける。"""
        shapes = []
        review_candidates = []
        for original_shape in annotation["shapes"]:
            shape = dict(original_shape)
            assignment = assignments.get(shape["group_id"])
            shape["tag_assignment"] = (
                {"instrument_uid": assignment["instrument_uid"], "instance_no": assignment["instance_no"]}
                if assignment is not None
                else None
            )
            if assignment is not None:
                shapes.append(shape)
            else:
                review_candidates.append(shape)
        return {
            **annotation,
            "tag_review_version": TAG_REVIEW_VERSION,
            "shapes": shapes,
            "review_candidates": review_candidates,
        }

    def record_review_state(self, revision: str, reviewed: bool) -> None:
        """Persist an explicit human review check for the current annotation revision."""
        revision = self._key(revision)
        metadata = read_json(self.root / DRAFTS_DIR / revision / METADATA_FILE)
        path = self.root / REVIEWS_DIR / revision / ANNOTATION_FILE
        self._record_annotation(metadata["source"], revision, "reviewed" if reviewed else "inferred", path)

    def approve(
        self,
        revision: str,
        reviewer: str,
        *,
        confirm_complete: bool,
        allow_empty: bool = False,
        exception_note: str = "",
        decisions: dict[int, str] | None = None,
        overwrite_masks: bool = False,
        assignments: dict[int, tuple[str, int] | None] | None = None,
    ) -> str:
        """画像承認を検証し、指定された物体の判定も同じ保存処理で確定する。

        Args:
            revision: Source draft revision.
            reviewer: Responsible reviewer identity.
            confirm_complete: Attestation that all visible target objects were checked.
            allow_empty: Attestation that this is a true-negative image.
            exception_note: Required explanation when visible tag counts differ.
            decisions: 今回確定する物体の採否。省略時は既存の物体判定を変更しない。
            overwrite_masks: 既存承認用maskの上書きへの明示同意。
            assignments: DINO候補の手動対応。指定時はYOLO注釈を承認せずDINOだけを確定する。

        Returns:
            承認ID。画像本体は複製せず、SQL上の識別情報と注釈ハッシュを記録する。

        Raises:
            CaptureError: Review evidence, identity, rectangles, or image is invalid.
        """
        revision = self._key(revision)
        if not reviewer.strip() or not confirm_complete:
            raise CaptureError("Reviewer and explicit complete-image confirmation are required")
        metadata = read_json(self.root / DRAFTS_DIR / revision / METADATA_FILE)
        review_dir = self.root / REVIEWS_DIR / revision
        path = review_dir / ANNOTATION_FILE
        data = read_json(path)
        source = metadata["source"]
        self._require_latest(source, revision)
        # DINO手動確認を未解決のYOLO矩形で妨げず、逆にYOLOの承認にも昇格させない。
        if assignments is None:
            allowed = {item[0] for item in source["instruments"]} | {tag["instrument_uid"] for tag in source["tags"]}
            shapes = validate_rectangles(data, allowed, source["width"], source["height"], allow_empty=allow_empty)
            validate_tag_review(data, source["tags"])
            expected = Counter(tag["instrument_uid"] for tag in source["tags"] if tag["visible"])
            actual = Counter(shape["label"] for shape in shapes)
            if expected != actual and not exception_note.strip():
                raise CaptureError("Tag/rectangle counts differ; provide an explicit exception note")
        self._review_image(data, source)
        content: dict[str, Any] = {
            "revision": revision,
            "labelme_sha256": digest_file(path),
            "image_sha256": source["sha256"],
            "reviewer": reviewer.strip(),
            "confirmed_complete": True,
            "allow_empty": allow_empty,
            "exception_note": exception_note.strip(),
        }
        if assignments is not None:
            if decisions is None or self.review_store is None:
                raise CaptureError("DINO手動対応には物体レビュー保存先と採否が必要である")
            content["scope"] = "dino"
            content["dino_assignments"] = {str(index): identity for index, identity in assignments.items()}
            content["dino_decisions"] = {str(index): decision for index, decision in decisions.items()}
        approval_id = content_hash(content)
        record = {"schema_version": SCHEMA_VERSION, **content, "approval_id": approval_id, "approved_at": datetime.now(UTC).isoformat(), "source": source}
        if assignments is not None and self.review_store is not None:
            previous_path = self.root / APPROVALS_DIR / f"{revision}.json"
            if previous_path.is_file():
                previous = read_json(previous_path)
                # YOLOが既に現行承認なら、その承認ID・担当者・公開状態はDINO操作で変更しない。
                if previous.get("labelme_sha256") == content["labelme_sha256"] and self.review_store.approved(revision, approval_id=previous["approval_id"], scope="yolo"):
                    record = previous
                    approval_id = previous["approval_id"]
        # 不正な一括判定から承認JSONだけが残らないよう、SQL側の検証・確定を先に行う。
        if decisions is not None and self.review_store is None:
            raise CaptureError("物体レビュー保存先が設定されていない")
        self._record_annotation(source, revision, "approved", path, record, decisions=decisions, overwrite_masks=overwrite_masks, assignments=assignments, decision_reviewer=reviewer.strip() if assignments is not None else None)
        if assignments is None:
            atomic_json(self.root / APPROVALS_DIR / f"{revision}.json", record)
        return approval_id

    def export_approved(self, destination: Path, session_id: str | None = None, revisions: set[str] | None = None) -> Path:
        """SQL上で現行の承認を、画像を複製せずにmanifestへ公開する。

        Args:
            destination: 新規公開先。既存の出力は上書きしない。
            session_id: 指定時に公開対象をそのセッションの承認だけに限定する。
            revisions: 指定時に公開対象をこのレビュー版の承認だけに限定する（部分・追加公開用）。

        Returns:
            yolo_trainが読み込むmanifest.jsonのパス。

        Raises:
            CaptureError: 現行承認がない、レビュー内容が変化した、または識別子が重複した場合。
        """
        stored_records = [read_json(path) for path in sorted((self.root / APPROVALS_DIR).glob("*.json"))]
        records = [record for record in stored_records if (session_id is None or record["source"]["session_id"] == session_id) and (revisions is None or record["revision"] in revisions) and self._source_is_current(record["source"])]
        if not records:
            raise CaptureError("No currently approved SQL capture images are available")
        samples: list[dict[str, Any]] = []
        identities: set[str] = set()
        for record in records:
            identity = record["source"]["source_path"]
            if identity in identities:
                raise CaptureError("Multiple source revisions approved for one image; revoke obsolete approvals first")
            identities.add(identity)
            self._validate_approval(record)
            if self.review_store is not None and not self.review_store.approved(record["revision"], approval_id=record["approval_id"], scope="yolo"):
                raise CaptureError("セッションDB上で画像承認が失効している")
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise CaptureError(f"Export destination already exists: {destination}")
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "batch"
            staging.mkdir()
            for record in records:
                source = record["source"]
                key = content_hash(source["source_path"])
                relative = Path("items") / key
                item_dir = staging / relative
                item_dir.mkdir(parents=True)
                shutil.copy2(self.root / REVIEWS_DIR / record["revision"] / ANNOTATION_FILE, item_dir / ANNOTATION_FILE)
                samples.append({**record, "image": source["source_path"], "source_path": source["source_path"], "annotation": (relative / ANNOTATION_FILE).as_posix()})
            atomic_json(staging / "manifest.json", {"schema_version": SCHEMA_VERSION, "samples": samples})
            staging.rename(destination)
        manifest_path = str((destination / "manifest.json").resolve())
        if self.ledger is not None:
            for record in records:
                self.ledger.mark_annotation_exported(
                    session_id=record["source"]["session_id"],
                    image_id=record["source"]["image_id"],
                    revision=record["revision"],
                    manifest_path=manifest_path,
                    source_path=record["source"]["source_path"],
                )
        return destination / "manifest.json"

    def revoke(self, revision: str) -> None:
        """Remove current approval while retaining immutable provenance artifacts.

        Args:
            revision: Obsolete source revision to remove from future exports.
        """
        revision = self._key(revision)
        approval_path = self.root / APPROVALS_DIR / f"{revision}.json"
        if approval_path.is_file():
            record = read_json(approval_path)
            annotation_path = self.root / REVIEWS_DIR / revision / ANNOTATION_FILE
            self._record_annotation(record["source"], revision, "revoked", annotation_path, record)
        elif self.review_store is not None:
            self.review_store.record_annotation(revision, read_json(self.root / REVIEWS_DIR / revision / ANNOTATION_FILE), "revoked")
        approval_path.unlink(missing_ok=True)

    def sync_existing_exports(self) -> int:
        """Reconcile previously published manifests with the central metadata ledger."""
        if self.ledger is None:
            return 0
        synced = 0
        for manifest_path in sorted((self.root / "exports").glob("*/manifest.json")):
            manifest = read_json(manifest_path)
            for record in manifest.get("samples", []):
                source = record["source"]
                revision = self._key(record["revision"])
                annotation_path = self.root / REVIEWS_DIR / revision / ANNOTATION_FILE
                approval_path = self.root / APPROVALS_DIR / f"{revision}.json"
                if not self._source_is_current(source) or not annotation_path.is_file() or not approval_path.is_file():
                    continue
                if self.review_store is not None and not self.review_store.approved(revision, approval_id=record["approval_id"], scope="yolo"):
                    continue
                if digest_file(annotation_path) != record["labelme_sha256"]:
                    raise CaptureError(f"Published annotation changed after export: {revision}")
                approval = read_json(approval_path)
                if approval.get("approval_id") != record.get("approval_id"):
                    continue
                self._record_annotation(source, revision, "approved", annotation_path, record)
                self.ledger.mark_annotation_exported(
                    session_id=source["session_id"],
                    image_id=source["image_id"],
                    revision=revision,
                    manifest_path=str(manifest_path.resolve()),
                    source_path=source["source_path"],
                )
                synced += 1
        return synced

    def _record_annotation(
        self,
        source: dict[str, Any],
        revision: str,
        status: str,
        annotation_path: Path,
        approval: dict[str, Any] | None = None,
        *,
        decisions: dict[int, str] | None = None,
        overwrite_masks: bool = False,
        assignments: dict[int, tuple[str, int] | None] | None = None,
        decision_reviewer: str | None = None,
    ) -> None:
        """SQLレビューと従来の承認台帳を、YOLO互換ファイルと同期する。"""
        if self.review_store is not None:
            self.review_store.record_annotation(
                revision,
                read_json(annotation_path),
                status,
                approval,
                decisions=decisions,
                sample=self.current_sample(source) if status == "approved" else None,
                overwrite_masks=overwrite_masks,
                assignments=assignments,
                decision_reviewer=decision_reviewer,
            )
        if assignments is not None and approval is not None and approval.get("scope") != "dino":
            return
        if self.ledger is None:
            return
        self.ledger.upsert_annotation_revision(
            session_id=source["session_id"],
            image_id=source["image_id"],
            tag_revision=source["tag_revision"],
            revision=revision,
            status=status,
            annotation_path=annotation_path.relative_to(self.root).as_posix(),
            annotation_sha256=digest_file(annotation_path),
            source_path=source["source_path"],
            approval_id=approval.get("approval_id") if approval else None,
            reviewer=approval.get("reviewer") if approval else None,
            approved_at=approval.get("approved_at") if approval else None,
        )

    def _validate_approval(self, record: dict[str, Any]) -> None:
        """承認後の注釈変更とSQL正本との不一致を検出する。"""
        revision = self._key(record["revision"])
        self._require_latest(record["source"], revision)
        path = self.root / REVIEWS_DIR / revision / ANNOTATION_FILE
        if digest_file(path) != record["labelme_sha256"]:
            raise CaptureError(f"Annotation changed after approval; approve again: {revision}")
        data = read_json(path)
        self._review_image(data, record["source"])

    def _mark_latest(self, sample: CaptureSample, revision: str) -> None:
        """Invalidate old approvals when the current source or matcher changes."""
        identity = content_hash(sample.source_path)
        atomic_json(self.root / LATEST_DIR / f"{identity}.json", {"revision": revision})

    def is_latest_revision(self, source: dict[str, Any], revision: str) -> bool:
        """Check the latest revision using its unique source path, with legacy-key fallback."""
        identity = content_hash(source["source_path"])
        latest_path = self.root / LATEST_DIR / f"{identity}.json"
        if latest_path.is_file():
            return read_json(latest_path).get("revision") == revision

        legacy_identity = content_hash([source["session_id"], source["image_id"]])
        legacy_path = self.root / LATEST_DIR / f"{legacy_identity}.json"
        if not legacy_path.is_file() or read_json(legacy_path).get("revision") != revision:
            return False

        metadata_path = self.root / DRAFTS_DIR / revision / METADATA_FILE
        if not metadata_path.is_file():
            return False
        saved_source = read_json(metadata_path).get("source", {})
        return saved_source.get("source_path") == source["source_path"]

    def _require_latest(self, source: dict[str, Any], revision: str) -> None:
        """Do not export a historical approval after a newer draft has been generated."""
        if not self.is_latest_revision(source, revision):
            raise CaptureError("Source tags or matching changed; revoke the old approval and review the latest draft")

    def _review_image(self, data: dict[str, Any], source: dict[str, Any]) -> Path:
        """SQLの現行パスだけを参照し、撮影画像を出力領域へ複製しない。"""
        if data.get("imagePath") != source.get("source_path"):
            raise CaptureError("Review image must reference its SQL source path")
        sample = self._require_current_source(source)
        path = sample.image_path
        with Image.open(path) as image:
            if image.size != (source["width"], source["height"]):
                raise CaptureError("SQL source image dimensions changed")
        return path

    def _source_is_current(self, source: dict[str, Any]) -> bool:
        """比較用スナップショットが現行SQLと完全一致するか返す。"""
        source_path = source.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            return False
        sample = self.capture_source.current_sample(source["session_id"], source["image_id"], source_path)
        if sample is None:
            return False
        return self._sample_record(sample) == source

    def is_current_snapshot(self, source: dict[str, Any], sample: CaptureSample) -> bool:
        """固定した起動時集合とSQLの現在行を画像全体の再ハッシュなしで照合する。

        Args:
            source: 成果物に保存された撮影スナップショット。
            sample: レビュー開始時点で固定したサンプル。

        Returns:
            保存済み内容、スナップショット、SQL行が一致する場合はTrue。
        """
        return self._sample_record(sample) == source and self.capture_source.is_current(sample)

    def current_sample(self, source: dict[str, Any]) -> CaptureSample:
        """SQL正本から現行画像を再読込し、内容ハッシュも検証する。

        Args:
            source: ドラフト等に保存された撮影スナップショット。

        Returns:
            SQLと一致する現行サンプル。

        Raises:
            CaptureError: SQLレコードまたは画像が削除・変更されている場合。
        """
        return self._require_current_source(source)

    def _require_current_source(self, source: dict[str, Any]) -> CaptureSample:
        """承認や画像配信で古いSQLスナップショットを拒否する。"""
        sample = self._sample_from_record(source)
        if sample is None:
            raise CaptureError("Capture source is no longer present and unchanged in SQL")
        return sample

    def _sample_from_record(self, source: dict[str, Any]) -> CaptureSample | None:
        """SQL行を読んでハッシュを検証し、保存済みタグ版と完全一致させる。"""
        source_path = source.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            return None
        sample = self.capture_source.current_sample(source["session_id"], source["image_id"], source_path)
        if sample is None or self._sample_record(sample) != source:
            return None
        return sample

    @staticmethod
    def _sample_record(sample: CaptureSample) -> dict[str, Any]:
        """DBルート相対パスを記録し、PCごとの絶対パスを成果物へ残さない。"""
        record = asdict(sample)
        record["image_path"] = sample.source_path
        return json.loads(json.dumps(record, ensure_ascii=True, default=str))

    @staticmethod
    def _key(value: str) -> str:
        """Prevent artifact identifiers from escaping their designated directories."""
        if len(value) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
            raise CaptureError("Invalid artifact fingerprint")
        return value
