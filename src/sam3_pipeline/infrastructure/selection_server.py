"""推論前に処理対象を選択するローカルviewer。"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from sam3_pipeline.domain.capture import CaptureError
from sam3_pipeline.infrastructure.annotation_store import DRAFTS_DIR, LATEST_DIR, FileAnnotationStore, content_hash, read_json
from sam3_pipeline.infrastructure.capture_source import SqliteCaptureSource
from sam3_pipeline.infrastructure.object_review_store import SqliteObjectReviewStore

LOGGER = logging.getLogger(__name__)
HTML_FILE = Path(__file__).parents[1] / "interface" / "selection.html"
MAX_SELECTION_BYTES = 1024 * 1024
REVIEW_RETURN_QUERY = "review_complete=1"


class SelectiveInferenceServer:
    """撮影SQLカタログを提示し、選択完了時に画像IDだけを返す。"""

    def __init__(
        self,
        source: SqliteCaptureSource,
        store: FileAnnotationStore,
        reviews: SqliteObjectReviewStore,
    ) -> None:
        """撮影入力、成果物、独立レビュー状態を選択一覧へ接続する。"""
        self.source = source
        self.store = store
        self.reviews = reviews
        self.entries = source.catalog()
        self.entry_by_identity = {(item["session_id"], item["image_id"]): item for item in self.entries}
        self.selected: list[tuple[str, str]] | None = None
        self.server: ThreadingHTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self.selection_ready = threading.Event()
        self.review_complete = threading.Event()
        self.state_lock = threading.Lock()
        self.reviewing = False
        self.accepting_selection = True
        self.url: str | None = None

    def select(self) -> list[tuple[str, str]]:
        """選択画面を開き、選択されたセッション・画像IDで応答する。"""
        self.start()
        with self.state_lock:
            has_pending_selection = self.selected is not None
        if not has_pending_selection:
            self.selection_ready.wait()
        with self.state_lock:
            selected = self.selected
            self.selected = None
            self.selection_ready.clear()
            self.accepting_selection = False
        if not selected:
            raise CaptureError("推論対象が選択されませんでした")
        return selected

    def start(self) -> None:
        """選択serverを起動し、レビュー後にも同じURLを使えるようにする。"""
        if self.server is not None:
            return
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                service._get(self)

            def do_POST(self) -> None:
                service._post(self)

            def log_message(self, format: str, *args: Any) -> None:
                LOGGER.debug("Selection HTTP: " + format, *args)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        host, port = self.server.server_address[:2]
        if isinstance(host, bytes):
            host = host.decode("ascii")
        self.url = f"http://{host}:{port}/"
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        LOGGER.info("画像選択viewer: %s", self.url)
        if not webbrowser.open(self.url):
            LOGGER.warning("ブラウザーを自動起動できません。URLを開いてください: %s", self.url)

    def begin_review(self) -> None:
        """レビュー中の選択画面への遷移を完了signalとして扱う。"""
        with self.state_lock:
            self.reviewing = True
            self.accepting_selection = False
            self.review_complete.clear()

    def close(self) -> None:
        """run終了時に選択serverと待受socketを閉じる。"""
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        if self.server_thread is not None:
            self.server_thread.join()
        self.server = None
        self.server_thread = None

    def _get(self, request: BaseHTTPRequestHandler) -> None:
        route = urlsplit(request.path)
        if route.path == "/":
            with self.state_lock:
                is_review_return = parse_qs(route.query).get("review_complete") == ["1"]
                if self.reviewing and is_review_return:
                    self.reviewing = False
                    self.accepting_selection = True
                    self.review_complete.set()
            self._respond(request, HTML_FILE.read_bytes(), "text/html; charset=utf-8")
        elif route.path == "/api/catalog":
            try:
                items = [self._with_status(item) for item in self.entries]
                self._respond_json(request, {"items": items, "sessions": sorted({item["session_id"] for item in items})})
            except (CaptureError, OSError, ValueError, KeyError) as error:
                self._respond_json(request, {"error": str(error)}, 400)
        elif route.path == "/api/image":
            query = parse_qs(route.query)
            identity = (query.get("session_id", [""])[0], query.get("image_id", [""])[0])
            item = self.entry_by_identity.get(identity)
            if item is None:
                self._respond_json(request, {"error": "画像が選択一覧にありません"}, 404)
                return
            path = (self.source.root / item["source_path"]).resolve()
            if not path.is_relative_to(self.source.root) or not path.is_file():
                self._respond_json(request, {"error": "画像ファイルがありません"}, 404)
                return
            try:
                body = path.read_bytes()
            except OSError as error:
                self._respond_json(request, {"error": f"画像を読み込めません: {error}"}, 404)
                return
            self._respond(request, body, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        else:
            self._respond_json(request, {"error": "Not found"}, 404)

    def _with_status(self, item: dict[str, Any]) -> dict[str, Any]:
        identity = content_hash([item["session_id"], item["image_id"]])
        latest_path = self.store.root / LATEST_DIR / f"{identity}.json"
        result = {key: item[key] for key in ("session_id", "image_id", "capture_date", "width", "height")}
        result.update({"draft": False, "yolo": "未処理", "dino": "未処理"})
        if not latest_path.is_file():
            return result
        revision = read_json(latest_path).get("revision")
        metadata_path = self.store.root / DRAFTS_DIR / str(revision) / "metadata.json"
        if not metadata_path.is_file():
            return result
        source = read_json(metadata_path).get("source", {})
        if (source.get("session_id"), source.get("image_id")) != (item["session_id"], item["image_id"]):
            return result
        result["draft"] = True
        review_state = self.reviews.catalog_state(item["session_id"], item["image_id"], str(revision))
        if review_state is None:
            return result
        if review_state["yolo_approved"]:
            result["yolo"] = "承認済み"
        elif review_state["progress"].get("reviewed") is True:
            result["yolo"] = "確認済み（未承認）"
        else:
            result["yolo"] = "未確認"
        result["dino"] = "承認済み" if review_state["dino_approved"] else review_state["dino_state"]
        return result

    def _post(self, request: BaseHTTPRequestHandler) -> None:
        try:
            length = int(request.headers.get("Content-Length", "0"))
        except (TypeError, ValueError) as error:
            self._respond_json(request, {"error": str(error)}, 400)
            return
        if length <= 0 or length > MAX_SELECTION_BYTES:
            self._respond_json(request, {"error": "選択データのサイズが不正です"}, 400)
            return
        try:
            payload = json.loads(request.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            self._respond_json(request, {"error": str(error)}, 400)
            return
        if not isinstance(payload, dict):
            self._respond_json(request, {"error": "選択データの形式が不正です"}, 400)
            return
        selected = payload.get("selected")
        if not isinstance(selected, list) or not selected:
            self._respond_json(request, {"error": "1件以上の画像を選択してください"}, 400)
            return
        identities = self._validated_selection(selected)
        if identities is None:
            self._respond_json(request, {"error": "選択画像が一覧にないか重複しています"}, 400)
            return
        with self.state_lock:
            if not self.accepting_selection:
                self._respond_json(request, {"error": "レビュー完了後に画像を選択してください"}, 409)
                return
            if self.selected is not None:
                self._respond_json(request, {"error": "前の選択処理がまだ開始していません"}, 400)
                return
            self.selected = identities
            self.selection_ready.set()
        self._respond_json(request, {"ok": True, "count": len(identities)})

    def _validated_selection(self, selected: list[Any]) -> list[tuple[str, str]] | None:
        identities = []
        seen = set()
        for value in selected:
            if not isinstance(value, dict):
                return None
            identity = (value.get("session_id"), value.get("image_id"))
            if identity not in self.entry_by_identity or identity in seen:
                return None
            seen.add(identity)
            identities.append(identity)
        return identities

    @staticmethod
    def _respond(request: BaseHTTPRequestHandler, body: bytes, content_type: str) -> None:
        request.send_response(200)
        request.send_header("Content-Type", content_type)
        request.send_header("Content-Length", str(len(body)))
        request.end_headers()
        request.wfile.write(body)

    @classmethod
    def _respond_json(cls, request: BaseHTTPRequestHandler, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        request.send_response(status)
        request.send_header("Content-Type", "application/json; charset=utf-8")
        request.send_header("Content-Length", str(len(body)))
        request.end_headers()
        request.wfile.write(body)
