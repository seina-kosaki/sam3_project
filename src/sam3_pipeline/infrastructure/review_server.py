"""Serve a loopback-only batch review UI over existing Labelme drafts.

The server uses Python's standard HTTP library to avoid adding a web runtime
for a local reviewer. Publishing delegates to the existing immutable approval
store, preserving the downstream YOLO manifest contract.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from sam3_pipeline.domain.capture import CaptureError, CaptureSample
from sam3_pipeline.domain.confidence import ConfidenceThresholds, assess_review_confidence
from sam3_pipeline.infrastructure.annotation_store import (
    ANNOTATION_FILE,
    APPROVALS_DIR,
    DRAFTS_DIR,
    LATEST_DIR,
    REVIEWS_DIR,
    FileAnnotationStore,
    atomic_json,
    content_hash,
    read_json,
    validate_rectangles,
)
from sam3_pipeline.infrastructure.tag_annotations import validate_tag_review

LOGGER = logging.getLogger(__name__)
HTML_FILE = Path(__file__).parents[1] / "interface" / "review.html"
TAG_REVIEW_FILE = HTML_FILE.with_name("tag_review.js")
PROGRESS_FILE = "review_progress.json"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MASK_ROUTE_PARTS = 2
REVISION_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class BatchReviewServer:
    """起動時のSQL対象だけをローカル画面でレビューできるようにする。

    Args:
        store: 注釈ストアと撮影SQLアダプター。
        thresholds: レビュー優先度を分類する設定済みスコア境界。
        samples: サーバー起動時点で固定したtagged撮影データ。

    Note:
        HTTPリスナーはloopbackだけにbindする。信頼できる作業端末内で使用し、ネットワークサービスとして公開しない。
    """

    def __init__(self, store: FileAnnotationStore, thresholds: ConfidenceThresholds, samples: tuple[CaptureSample, ...]) -> None:
        """成果物領域と起動時点のSQL対象集合を結び付ける。"""
        self.store = store
        self.thresholds = thresholds
        self.samples = {(sample.session_id, sample.image_id): sample for sample in samples}
        self._progress_lock = threading.Lock()

    def serve(self) -> None:
        """Start the local review server and open its page in the default browser."""
        handler = self._handler_type()
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            address = server.server_address[0]
            port = server.server_address[1]
            if isinstance(address, bytes):
                address = address.decode("ascii")
            url = f"http://{address}:{port}/"
            LOGGER.info("Batch review: %s (Ctrl+C to stop)", url)
            webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                LOGGER.info("Batch review stopped")

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        """Bind request methods to this store without global mutable state."""
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                service._handle_get(self)

            def do_POST(self) -> None:
                service._handle_post(self)

            def log_message(self, format: str, *args: Any) -> None:
                LOGGER.debug("Review HTTP: " + format, *args)

        return Handler

    def _handle_get(self, request: BaseHTTPRequestHandler) -> None:
        """UI、現行レビュー一覧、またはSQL上の原本画像を返す。"""
        route = urlsplit(request.path).path
        if route == "/":
            self._respond(request, HTML_FILE.read_bytes(), "text/html; charset=utf-8")
            return
        if route == "/tag_review.js":
            self._respond(request, TAG_REVIEW_FILE.read_bytes(), "text/javascript; charset=utf-8")
            return
        if route == "/api/items":
            try:
                items = self._items()
                sessions = sorted({item["session_id"] for item in items})
                self._respond_json(request, {"items": items, "sessions": sessions})
            except (CaptureError, OSError, ValueError) as error:
                LOGGER.warning("レビュー一覧を読み込めない: %s", error)
                self._respond_json(request, {"error": str(error)}, status=400)
            return
        if route.startswith("/api/mask/"):
            try:
                self._respond(request, self._object_mask(route), "image/png")
            except (CaptureError, ValueError, OSError) as error:
                self._respond_json(request, {"error": str(error)}, status=404)
            return
        if route.startswith("/api/image/"):
            revision = unquote(route.removeprefix("/api/image/"))
            self._require_revision(revision)
            review_dir = self.store.root / REVIEWS_DIR / revision
            annotation = read_json(review_dir / ANNOTATION_FILE)
            source = read_json(self.store.root / DRAFTS_DIR / revision / "metadata.json")["source"]
            identity = (source["session_id"], source["image_id"])
            sample = self.samples.get(identity)
            if sample is None or not self.store.is_current_snapshot(source, sample):
                self._respond_json(request, {"error": "Capture source is no longer current"}, status=404)
                return
            try:
                image_path = self.store.current_sample(source).image_path
            except CaptureError as error:
                LOGGER.warning("Review image is no longer available: %s", error)
                self._respond_json(request, {"error": str(error)}, status=404)
                return
            if annotation["imagePath"] != source["source_path"]:
                raise CaptureError("Review image path differs from its SQL source")
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self._respond(request, image_path.read_bytes(), content_type)
            return
        self._respond_json(request, {"error": "Not found"}, status=404)

    def _handle_post(self, request: BaseHTTPRequestHandler) -> None:
        """Persist a review edit, checkbox state, or complete batch publication."""
        route = urlsplit(request.path).path
        try:
            payload = self._read_payload(request)
            if route == "/api/reviewed":
                self._set_reviewed(payload)
                self._respond_json(request, {"ok": True})
            elif route == "/api/annotation":
                self._save_annotation(payload)
                self._respond_json(request, {"ok": True})
            elif route == "/api/object-review":
                self._set_object_review(payload)
                self._respond_json(request, {"ok": True})
            elif route == "/api/approve-image":
                self._approve_image(payload)
                self._respond_json(request, {"ok": True})
            elif route == "/api/complete-review":
                self._complete_review(payload)
                self._respond_json(request, {"ok": True})
            elif route == "/api/publish":
                manifest = self._publish(payload)
                self._respond_json(request, {"ok": True, "manifest": str(manifest)})
            else:
                self._respond_json(request, {"error": "Not found"}, status=404)
        except (CaptureError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._respond_json(request, {"error": str(error)}, status=400)
        except OSError as error:
            LOGGER.exception("Review request failed")
            self._respond_json(request, {"error": str(error)}, status=500)

    def _items(self) -> list[dict[str, Any]]:
        """起動時SQL集合に存在する最新リビジョンだけを一覧化する。"""
        progress = self._read_progress()
        items = []
        for metadata_path in sorted((self.store.root / DRAFTS_DIR).glob("*/metadata.json")):
            revision = metadata_path.parent.name
            source = read_json(metadata_path)["source"]
            sample = self.samples.get((source["session_id"], source["image_id"]))
            if sample is None or not self.store.is_current_snapshot(source, sample):
                continue
            identity = content_hash([source["session_id"], source["image_id"]])
            latest_path = self.store.root / LATEST_DIR / f"{identity}.json"
            if not latest_path.is_file() or read_json(latest_path).get("revision") != revision:
                continue
            review_dir = self.store.root / REVIEWS_DIR / revision
            annotation = read_json(review_dir / ANNOTATION_FILE)
            metadata = read_json(metadata_path)
            confidence = assess_review_confidence(metadata, annotation, self.thresholds)
            state = self._progress_state(progress.get(revision))
            approved = (self.store.root / APPROVALS_DIR / f"{revision}.json").is_file()
            objects: list[dict[str, Any]] = []
            dino_approved = approved
            if self.store.review_store is not None:
                dino_approved = self.store.review_store.approved(revision)
                approved = self.store.review_store.approved(revision, scope="yolo")
                objects = self.store.review_store.objects(sample, revision)
                for item in objects:
                    item["mask_url"] = f"/api/mask/{revision}/{item['object_index']}"
            items.append(
                {
                    "revision": revision,
                    "image_id": source["image_id"],
                    "session_id": source["session_id"],
                    "capture_date": PurePosixPath(source["source_path"]).parts[0],
                    "image_url": f"/api/image/{revision}",
                    "width": source["width"],
                    "height": source["height"],
                    "tags": source["tags"],
                    "instruments": source["instruments"],
                    "annotation": annotation,
                    "confidence": confidence.level.value,
                    "reasons": list(confidence.reasons),
                    "candidate_confidence": [{"index": item.candidate_index, "level": item.level.value, "score": item.score, "reasons": list(item.reasons)} for item in confidence.candidates],
                    "reviewed": state["reviewed"] or approved,
                    "exception_note": state["exception_note"],
                    "allow_empty": state["allow_empty"],
                    "approved": approved,
                    "dino_approved": dino_approved,
                    "objects": objects,
                }
            )
        return items

    def _current_object_revision(self, revision: str) -> tuple[str, CaptureSample]:
        """起動時集合、現在の撮影SQL、最新レビューの三者を照合する。"""
        revision = self._require_revision(revision)
        source = read_json(self.store.root / DRAFTS_DIR / revision / "metadata.json")["source"]
        sample = self.samples.get((source["session_id"], source["image_id"]))
        if sample is None or not self.store.is_current_snapshot(source, sample):
            raise CaptureError("Review source is no longer current")
        identity = content_hash([sample.session_id, sample.image_id])
        if read_json(self.store.root / LATEST_DIR / f"{identity}.json")["revision"] != revision:
            raise CaptureError("Review revision is no longer current")
        return revision, self.store.current_sample(source)

    def _object_mask(self, route: str) -> bytes:
        """ルートの物体番号を検証し、現行の原本マスクを取得する。"""
        parts = route.removeprefix("/api/mask/").split("/")
        if len(parts) != MASK_ROUTE_PARTS or not parts[1].isdigit() or self.store.review_store is None:
            raise CaptureError("Invalid object mask request")
        revision, sample = self._current_object_revision(parts[0])
        return self.store.review_store.mask_png(sample, revision, int(parts[1]))

    def _approve_image(self, payload: dict[str, Any]) -> None:
        """再exportせず画像承認を記録し、DINOの物体別判定とは区別する。"""
        revision, _ = self._current_object_revision(str(payload.get("revision", "")))
        if payload.get("confirm_complete") is not True or not isinstance(payload.get("reviewer"), str):
            raise CaptureError("Complete-image confirmation and reviewer are required")
        self.store.approve(revision, payload["reviewer"], confirm_complete=True, allow_empty=payload.get("allow_empty") is True, exception_note=str(payload.get("exception_note", "")))

    def _complete_review(self, payload: dict[str, Any]) -> None:
        """全物体の確定判定と画像承認を一回の操作で保存する。

        Args:
            payload: 画像版、共通担当者、全物体の採否と明示確認。

        Raises:
            CaptureError: 担当者、全件判定、現行性、原本または注釈が不正な場合。
        """
        revision, _ = self._current_object_revision(str(payload.get("revision", "")))
        reviewer = payload.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise CaptureError("画面上部の確認担当者を入力してください")
        if payload.get("confirm_complete") is not True:
            raise CaptureError("画像全体を確認してから保存してください")
        records = payload.get("decisions")
        if not isinstance(records, list):
            raise CaptureError("画像内の全物体を判定してください")
        # 重複番号や真偽値を辞書化で見落とさず、保存前に拒否する。
        decisions: dict[int, str] = {}
        for record in records:
            if not isinstance(record, dict):
                raise CaptureError("物体判定の形式が不正である")
            index, status = record.get("object_index"), record.get("status")
            if isinstance(index, bool) or not isinstance(index, int) or index in decisions or status not in ("accepted", "rejected"):
                raise CaptureError("全物体の「使う／使わない」を重複なく選択してください")
            decisions[index] = status
        assignments: dict[int, tuple[str, int] | None] | None = None
        if "assignments" in payload:
            assignments = {}
            raw_assignments = payload["assignments"]
            if not isinstance(raw_assignments, list):
                raise CaptureError("全候補のタグ対応を指定してください")
            for record in raw_assignments:
                if not isinstance(record, dict):
                    raise CaptureError("タグ対応の形式が不正である")
                index = record.get("object_index")
                if type(index) is not int or index in assignments:
                    raise CaptureError("候補番号を重複なく指定してください")
                uid, instance = record.get("instrument_uid"), record.get("instance_no")
                if uid is None and instance is None:
                    assignments[index] = None
                elif isinstance(uid, str) and uid and type(instance) is int:
                    assignments[index] = (uid, instance)
                else:
                    raise CaptureError("器具UIDと個体番号を指定してください")
        self.store.approve(
            revision,
            reviewer,
            confirm_complete=True,
            decisions=decisions,
            overwrite_masks=payload.get("overwrite_masks") is True,
            allow_empty=payload.get("allow_empty") is True,
            exception_note=str(payload.get("exception_note", "")),
            assignments=assignments,
        )

    def _set_object_review(self, payload: dict[str, Any]) -> None:
        """画像確認とは独立して、元bbox・maskに対する物体別採否を記録する。"""
        if self.store.review_store is None:
            raise CaptureError("Object review storage is not configured")
        revision, sample = self._current_object_revision(str(payload.get("revision", "")))
        index = payload.get("object_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise CaptureError("Object index must be an integer")
        if payload.get("confirm_original") is not True or not isinstance(payload.get("reviewer"), str):
            raise CaptureError("Confirm original bbox and mask before recording a decision")
        self.store.review_store.decide(sample, revision, index, status=str(payload.get("status", "")), reviewer=str(payload.get("reviewer", "")))

    def _set_reviewed(self, payload: dict[str, Any]) -> None:
        """Persist one explicit review check for the current revision."""
        revision = self._require_revision(str(payload.get("revision", "")))
        if not any(item["revision"] == revision for item in self._items()):
            raise CaptureError("Revision is not the latest review item")
        with self._progress_lock:
            progress = self._read_progress()
            state = self._progress_state(progress.get(revision))
            for key in ("reviewed", "allow_empty"):
                if key in payload:
                    if not isinstance(payload[key], bool):
                        raise TypeError(f"{key} must be a boolean")
                    state[key] = payload[key]
            if "exception_note" in payload:
                if not isinstance(payload["exception_note"], str):
                    raise TypeError("exception_note must be text")
                state["exception_note"] = payload["exception_note"]
            progress[revision] = state
            if self.store.review_store is not None:
                self.store.review_store.set_progress(revision, state)
            atomic_json(self.store.root / PROGRESS_FILE, progress)
        self.store.record_review_state(revision, state["reviewed"])

    def _save_annotation(self, payload: dict[str, Any]) -> None:
        """Validate and save browser edits without changing the original image."""
        revision = self._require_revision(str(payload.get("revision", "")))
        self._current_object_revision(revision)
        annotation = payload.get("annotation")
        if not isinstance(annotation, dict):
            raise TypeError("annotation must be a JSON object")
        metadata = read_json(self.store.root / DRAFTS_DIR / revision / "metadata.json")
        source = metadata["source"]
        sample = self.samples.get((source["session_id"], source["image_id"]))
        if sample is None or not self.store.is_current_snapshot(source, sample):
            raise CaptureError("Review source is no longer present in the SQL snapshot")
        path = self.store.root / REVIEWS_DIR / revision / ANNOTATION_FILE
        original = read_json(path)
        if "tag_review_version" in original and annotation.get("tag_review_version") != original["tag_review_version"]:
            raise CaptureError("タグ単位のレビュー形式を旧形式へ戻すことはできない")
        if (annotation.get("imagePath"), annotation.get("imageWidth"), annotation.get("imageHeight")) != (original["imagePath"], source["width"], source["height"]):
            raise CaptureError("Review image reference and dimensions are immutable")
        allowed = {item[0] for item in source["instruments"]}
        allowed.update(tag["instrument_uid"] for tag in source["tags"])
        validate_rectangles(annotation, allowed, source["width"], source["height"], allow_empty=True, allow_unresolved=True)
        validate_tag_review(annotation, source["tags"])
        if "tag_review_version" in annotation:
            # 学習対象外の候補にも同じ矩形検証を適用し、不正な座標を画面へ戻さない。
            validate_rectangles({**annotation, "shapes": annotation["review_candidates"]}, allowed, source["width"], source["height"], allow_empty=True, allow_unresolved=True)
        atomic_json(path, annotation)
        with self._progress_lock:
            progress = self._read_progress()
            state = self._progress_state(progress.get(revision))
            state["reviewed"] = False
            progress[revision] = state
            if self.store.review_store is not None:
                self.store.review_store.set_progress(revision, state)
            atomic_json(self.store.root / PROGRESS_FILE, progress)
        self.store.record_review_state(revision, False)

    def _publish(self, payload: dict[str, Any]) -> Path:
        """選択セッションの確認済み草案を承認し、セッション名で公開する。"""
        raw_reviewer = payload.get("reviewer")
        if not isinstance(raw_reviewer, str) or not raw_reviewer.strip():
            raise CaptureError("一括公開欄の確認担当者を入力してください")
        reviewer = raw_reviewer.strip()
        if payload.get("confirm_complete") is not True:
            raise CaptureError("「選択セッションの全画像を確認済み」にチェックを入れてください")
        session_id = str(payload.get("session_id", "")).strip()
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("有効なセッションを選択してください")
        items = [item for item in self._items() if item["session_id"] == session_id]
        if not items:
            raise CaptureError("選択セッションに現行レビュー画像がありません")
        destination = self.store.root / "exports" / session_id
        if destination.exists():
            raise CaptureError(f"Export destination already exists: {destination}")
        progress = self._read_progress()
        unchecked = sum(not self._progress_state(progress.get(item["revision"]))["reviewed"] and not item["approved"] for item in items)
        if unchecked:
            raise CaptureError(f"選択セッションに未確認画像が{unchecked}件ある。各画像の「一覧確認済」をチェックしてください")
        notes = {revision: self._progress_state(progress.get(revision))["exception_note"] for revision in progress}
        approved_items: list[tuple[str, str, bool]] = []
        for item in items:
            revision = item["revision"]
            metadata = read_json(self.store.root / DRAFTS_DIR / revision / "metadata.json")
            annotation = read_json(self.store.root / REVIEWS_DIR / revision / ANNOTATION_FILE)
            source = metadata["source"]
            allowed = {entry[0] for entry in source["instruments"]}
            allowed.update(tag["instrument_uid"] for tag in source["tags"])
            state = self._progress_state(progress.get(revision))
            shapes = validate_rectangles(annotation, allowed, source["width"], source["height"], allow_empty=state["allow_empty"])
            expected = Counter(tag["instrument_uid"] for tag in source["tags"] if tag["visible"])
            actual = Counter(shape["label"] for shape in shapes)
            note = notes.get(revision, "")
            if expected != actual and not note.strip():
                raise CaptureError(f"{revision}: tag and rectangle counts differ; enter an exception note")
            approved_items.append((revision, note, state["allow_empty"]))

        for revision, note, empty_ok in approved_items:
            self.store.approve(revision, reviewer, confirm_complete=True, allow_empty=empty_ok, exception_note=note)
        return self.store.export_approved(destination, session_id=session_id)

    def _read_progress(self) -> dict[str, Any]:
        """Read reviewer checkboxes without altering immutable annotation records."""
        if self.store.review_store is not None:
            return self.store.review_store.progress()
        path = self.store.root / PROGRESS_FILE
        return read_json(path) if path.is_file() else {}

    @staticmethod
    def _progress_state(value: Any) -> dict[str, Any]:
        """Normalize persisted review state, including the earlier bool format."""
        if isinstance(value, bool):
            return {"reviewed": value, "exception_note": "", "allow_empty": False}
        if not isinstance(value, dict):
            return {"reviewed": False, "exception_note": "", "allow_empty": False}
        return {
            "reviewed": bool(value.get("reviewed", False)),
            "exception_note": str(value.get("exception_note", "")),
            "allow_empty": bool(value.get("allow_empty", False)),
        }

    @staticmethod
    def _require_revision(revision: str) -> str:
        """Accept only content-addressed revision IDs, never arbitrary paths."""
        if not REVISION_PATTERN.fullmatch(revision):
            raise ValueError("Invalid review revision")
        return revision

    @staticmethod
    def _read_payload(request: BaseHTTPRequestHandler) -> dict[str, Any]:
        """Parse bounded JSON requests from the local browser UI."""
        length = int(request.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise ValueError("Request size is invalid")
        payload = json.loads(request.rfile.read(length))
        if not isinstance(payload, dict):
            raise TypeError("Request body must be a JSON object")
        return payload

    @staticmethod
    def _respond_json(request: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
        """Write a same-origin JSON response for browser requests."""
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        BatchReviewServer._respond(request, body, "application/json; charset=utf-8", status)

    @staticmethod
    def _respond(request: BaseHTTPRequestHandler, body: bytes, content_type: str, status: int = 200) -> None:
        """Write a bounded response with explicit content type and length."""
        request.send_response(status)
        request.send_header("Content-Type", content_type)
        request.send_header("Content-Length", str(len(body)))
        request.send_header("X-Content-Type-Options", "nosniff")
        request.send_header("Cache-Control", "no-store")
        request.end_headers()
        request.wfile.write(body)
