"""選択前カタログと対象IDの検証。"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from PIL import Image

from sam3_pipeline.infrastructure.annotation_store import DRAFTS_DIR, LATEST_DIR, METADATA_FILE, FileAnnotationStore, atomic_json, content_hash
from sam3_pipeline.infrastructure.capture_source import SqliteCaptureSource
from sam3_pipeline.infrastructure.object_review_store import SqliteObjectReviewStore
from sam3_pipeline.infrastructure.prediction_store import SqlitePredictionStore
from sam3_pipeline.infrastructure.selection_server import SelectiveInferenceServer

CATALOG_IMAGE_COUNT = 1000
HTTP_CONFLICT_STATUS = 409
HTTP_BAD_REQUEST_STATUS = 400
HTTP_OK_STATUS = 200
SELECTION_HTML = Path(__file__).resolve().parents[1] / "src" / "sam3_pipeline" / "interface" / "selection.html"
BUNDLE_SCHEMA = """
CREATE TABLE declarations(declaration_id TEXT, items_json TEXT);
CREATE TABLE images(session_id TEXT, image_id TEXT, declaration_id TEXT, arrangement_no INTEGER,
path TEXT, sha256 TEXT, width INTEGER, height INTEGER, tag_revision INTEGER, tag_status TEXT, depth_path TEXT);
CREATE TABLE hints(image_id TEXT, instrument_uid TEXT, instance_no INTEGER, visible INTEGER,
hint_source TEXT, center_json TEXT, end_a_json TEXT, end_b_json TEXT, occluded INTEGER);
"""
NODE_SELECTION_UI = """
const vm = require("node:vm");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const elements = new Map();
const document = {
  querySelector(selector) {
    if (!elements.has(selector)) {
      const element = {
        _html: "", options: [], value: "", textContent: "", disabled: false,
        append(option) {
          this.options.push(option);
          if (this.options.length === 1) this.value = option.value;
        },
        querySelectorAll: () => []
      };
      Object.defineProperty(element, "innerHTML", {
        get() { return this._html; },
        set(value) { this._html = value; this.options = []; }
      });
      elements.set(selector, element);
    }
    return elements.get(selector);
  },
  createElement: () => ({value: "", textContent: ""})
};
const sandbox = vm.createContext({
  document,
  fetch: async () => ({json: async () => ({items: input.items})})
});
(async () => {
  vm.runInContext(input.script, sandbox);
  await new Promise(resolve => setImmediate(resolve));
  const date = elements.get("#date");
  const session = elements.get("#session");
  const dates = date.options.map(option => option.value);
  const firstDateSessions = session.options.map(option => option.value);
  const dateDisabled = date.disabled;
  const sessionDisabled = session.disabled;
  if (input.items.length) vm.runInContext("currentPage = 1; selected.add(identity(items[0]));", sandbox);
  date.value = "2026-10-24";
  date.onchange();
  const secondDate = {
    session: session.value,
    sessions: session.options.map(option => option.value),
    page: elements.get("#page").textContent,
    grid: elements.get("#grid").innerHTML,
    count: elements.get("#count").textContent
  };
  date.value = "2026-10-25";
  date.onchange();
  process.stdout.write(JSON.stringify({
    dates, firstDateSessions, dateDisabled, sessionDisabled, secondDate,
    fallbackSession: session.value,
    fallbackSessions: session.options.map(option => option.value),
    fallbackGrid: elements.get("#grid").innerHTML
  }));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


class CatalogRequest:
    """選択catalog応答を記録する最小HTTP要求。"""

    path = "/api/catalog"

    def __init__(self) -> None:
        """応答コードと本文の格納先を初期化する。"""
        self.status = 0
        self.headers: list[tuple[str, str]] = []
        self.wfile = io.BytesIO()

    def send_response(self, status: int) -> None:
        """HTTP状態コードを記録する。"""
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        """応答ヘッダーを記録する。"""
        self.headers.append((name, value))

    def end_headers(self) -> None:
        """HTTPヘッダーの終了を記録せず受け入れる。"""
        pass


class SelectionPostRequest(CatalogRequest):
    """選択POST payloadをテストする最小HTTP要求。"""

    path = "/api/select"

    def __init__(self, body: bytes) -> None:
        """本文をHTTP要求streamとして渡す。"""
        super().__init__()
        self.headers = {"Content-Length": str(len(body))}
        self.response_headers: list[tuple[str, str]] = []
        self.rfile = io.BytesIO(body)

    def send_header(self, name: str, value: str) -> None:
        """応答ヘッダーを記録する。"""
        self.response_headers.append((name, value))


def _capture_bundle(
    root: Path,
    image_count: int = 1,
    session_id: str = "session_a",
    capture_date: str = "2026-10-23",
) -> None:
    bundle = root / capture_date / session_id
    image_paths = []
    for index in range(image_count):
        image_name = "image.png" if image_count == 1 else f"image_{index:04}.png"
        image = bundle / "raw" / image_name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red").save(image)
        image_paths.append(image)
    with closing(sqlite3.connect(bundle / "metadata.sqlite3")) as connection, connection:
        connection.executescript(BUNDLE_SCHEMA)
        connection.execute("INSERT INTO declarations VALUES (?, ?)", ("decl", json.dumps([])))
        connection.executemany(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    session_id,
                    f"image_{index:04}",
                    "decl",
                    index,
                    image_paths[index].relative_to(root).as_posix(),
                    hashlib.sha256(image_paths[index].read_bytes()).hexdigest(),
                    8,
                    8,
                    1,
                    "tagged",
                    None,
                )
                for index in range(image_count)
            ],
        )


def test_catalog_is_lightweight_for_one_thousand_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """1000件を列挙し、未選択画像の内容hash検証を始めない。"""
    _capture_bundle(tmp_path, CATALOG_IMAGE_COUNT)
    with closing(sqlite3.connect(tmp_path / "2026-10-23" / "session_a" / "metadata.sqlite3")) as connection, connection:
        connection.execute("UPDATE images SET sha256='deliberately-not-the-file-hash'")
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    monkeypatch.setattr(source, "iter_samples", lambda: pytest.fail("選択一覧で全サンプル化してはならない"))

    catalog = source.catalog()

    assert len(catalog) == CATALOG_IMAGE_COUNT
    assert catalog[0]["image_id"] == "image_0000"
    assert catalog[-1]["image_id"] == "image_0999"


def test_selection_accepts_multiple_listed_ids_only(tmp_path: Path) -> None:
    """viewerから届く複数画像IDは一覧に存在し重複しない場合だけ受理する。"""
    _capture_bundle(tmp_path, 2)
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    prediction_store = SqlitePredictionStore(tmp_path, tmp_path, tmp_path / "masks", 1.0)
    prediction_store.initialize()
    reviews = SqliteObjectReviewStore(tmp_path, 1.0, prediction_store)
    reviews.initialize()
    store = FileAnnotationStore(tmp_path / "output", source, review_store=reviews)
    selection = SelectiveInferenceServer(source, store, reviews)

    selected = [
        {"session_id": "session_a", "image_id": "image_0000"},
        {"session_id": "session_a", "image_id": "image_0001"},
    ]

    assert selection._validated_selection(selected) == [
        selection.entries[0]["source_path"],
        selection.entries[1]["source_path"],
    ]
    assert selection._validated_selection([*selected, selected[0]]) is None
    assert selection._validated_selection([{"session_id": "session_a", "image_id": "missing"}]) is None


def test_selection_uses_source_path_when_session_and_image_are_reused(tmp_path: Path) -> None:
    """Require source_path when identical session/image IDs occur on multiple dates."""
    _capture_bundle(tmp_path, capture_date="2026-10-23")
    _capture_bundle(tmp_path, capture_date="2026-10-24")
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    prediction_store = SqlitePredictionStore(tmp_path, tmp_path, tmp_path / "masks", 1.0)
    prediction_store.initialize()
    reviews = SqliteObjectReviewStore(tmp_path, 1.0, prediction_store)
    reviews.initialize()
    store = FileAnnotationStore(tmp_path / "output", source, review_store=reviews)
    selection = SelectiveInferenceServer(source, store, reviews)
    first_path = "2026-10-23/session_a/raw/image.png"
    second_path = "2026-10-24/session_a/raw/image.png"

    assert selection._validated_selection([{"session_id": "session_a", "image_id": "image_0000"}]) is None
    assert selection._validated_selection([{"source_path": second_path}]) == [second_path]
    assert selection.entry_by_source_path[second_path]["capture_date"] == "2026-10-24"
    assert first_path != second_path
    request = SelectionPostRequest(
        json.dumps({"selected": [{"session_id": "session_a", "image_id": "image_0000"}]}).encode("utf-8")
    )

    selection._post(request)  # type: ignore[arg-type]

    assert request.status == HTTP_BAD_REQUEST_STATUS


def test_catalog_latest_drafts_are_keyed_by_source_path(tmp_path: Path) -> None:
    """Keep latest revision status separate for reused IDs on different capture dates."""
    _capture_bundle(tmp_path, capture_date="2026-10-23")
    _capture_bundle(tmp_path, capture_date="2026-10-24")
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    prediction_store = SqlitePredictionStore(tmp_path, tmp_path, tmp_path / "masks", 1.0)
    prediction_store.initialize()
    reviews = SqliteObjectReviewStore(tmp_path, 1.0, prediction_store)
    reviews.initialize()
    store = FileAnnotationStore(tmp_path / "output", source, review_store=reviews)
    samples_by_path = {sample.source_path: sample for sample in source.iter_samples()}
    entries = source.catalog()

    for index, item in enumerate(entries):
        revision = f"revision-{index}"
        metadata = {"source": {"session_id": item["session_id"], "image_id": item["image_id"], "source_path": item["source_path"]}}
        atomic_json(store.root / DRAFTS_DIR / revision / METADATA_FILE, metadata)
        store._mark_latest(samples_by_path[item["source_path"]], revision)

    selection = SelectiveInferenceServer(source, store, reviews)
    statuses = {item["source_path"]: selection._with_status(item) for item in entries}

    assert statuses["2026-10-23/session_a/raw/image.png"]["draft"] is True
    assert statuses["2026-10-24/session_a/raw/image.png"]["draft"] is True
    assert content_hash("2026-10-23/session_a/raw/image.png") != content_hash("2026-10-24/session_a/raw/image.png")


def test_live_selection_server_accepts_selection_after_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """レビュー後に同じ選択serverへ戻り、次の選択を受け付ける。"""
    _capture_bundle(tmp_path)
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    prediction_store = SqlitePredictionStore(tmp_path, tmp_path, tmp_path / "masks", 1.0)
    prediction_store.initialize()
    reviews = SqliteObjectReviewStore(tmp_path, 1.0, prediction_store)
    reviews.initialize()
    store = FileAnnotationStore(tmp_path / "output", source, review_store=reviews)
    selection = SelectiveInferenceServer(source, store, reviews)
    monkeypatch.setattr("sam3_pipeline.infrastructure.selection_server.webbrowser.open", lambda _url: True)
    selected = [{"session_id": "session_a", "image_id": "image_0000"}]

    def request_selection(method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
        address = urlsplit(selection.url)
        connection = http.client.HTTPConnection(address.hostname, address.port, timeout=5)
        try:
            headers = {"Content-Type": "application/json"} if body is not None else {}
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def post_selection() -> None:
        body = json.dumps({"selected": selected}).encode("utf-8")
        status, _body = request_selection("POST", "/api/select", body)
        assert status == HTTP_OK_STATUS

    try:
        selection.start()
        assert selection.url is not None
        post_selection()
        assert selection.select() == ["2026-10-23/session_a/raw/image.png"]

        body = json.dumps({"selected": selected}).encode("utf-8")
        status, _body = request_selection("POST", "/api/select", body)
        assert status == HTTP_CONFLICT_STATUS

        selection.begin_review()
        status, page = request_selection("GET", "/")
        assert status == HTTP_OK_STATUS
        assert page == SELECTION_HTML.read_bytes()
        assert not selection.review_complete.is_set()

        status, page = request_selection("GET", "/?review_complete=1")
        assert status == HTTP_OK_STATUS
        assert page == SELECTION_HTML.read_bytes()

        assert selection.review_complete.is_set()
        post_selection()
        assert selection.select() == ["2026-10-23/session_a/raw/image.png"]
        assert selection.server is not None
        assert selection.server_thread is not None and selection.server_thread.is_alive()
    finally:
        selection.close()


def test_catalog_keeps_unregistered_draft_as_unprocessed(tmp_path: Path) -> None:
    """DB未登録の通常草案を読む一覧APIはエラーにせず未処理で返す。"""
    _capture_bundle(tmp_path)
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    prediction_store = SqlitePredictionStore(tmp_path, tmp_path, tmp_path / "masks", 1.0)
    prediction_store.initialize()
    reviews = SqliteObjectReviewStore(tmp_path, 1.0, prediction_store)
    reviews.initialize()
    store = FileAnnotationStore(tmp_path / "output", source, review_store=reviews)
    revision = "unregistered-revision"
    identity = content_hash(["session_a", "image_0000"])
    metadata = {
        "source": {
            "session_id": "session_a",
            "image_id": "image_0000",
            "source_path": "2026-10-23/session_a/raw/image.png",
        }
    }
    atomic_json(store.root / DRAFTS_DIR / revision / METADATA_FILE, metadata)
    atomic_json(store.root / LATEST_DIR / f"{identity}.json", {"revision": revision})
    selection = SelectiveInferenceServer(source, store, reviews)

    request = CatalogRequest()

    selection._get(request)  # type: ignore[arg-type]

    response: dict[str, Any] = json.loads(request.wfile.getvalue())
    assert request.status == HTTP_OK_STATUS
    assert response["items"][0]["draft"] is True
    assert response["items"][0]["yolo"] == "未処理"
    assert response["items"][0]["dino"] == "未処理"


def test_catalog_does_not_read_another_sessions_revision(tmp_path: Path) -> None:
    """別セッションDBに同じrevision値があっても対象画像の状態へ混ぜない。"""
    _capture_bundle(tmp_path)
    _capture_bundle(tmp_path, session_id="session_b", capture_date="2026-10-24")
    source = SqliteCaptureSource(tmp_path, timeout=1.0)
    prediction_store = SqlitePredictionStore(tmp_path, tmp_path, tmp_path / "masks", 1.0)
    prediction_store.initialize()
    reviews = SqliteObjectReviewStore(tmp_path, 1.0, prediction_store)
    reviews.initialize()
    store = FileAnnotationStore(tmp_path / "output", source, review_store=reviews)
    revision = "same-revision-in-another-session"
    identity = content_hash(["session_a", "image_0000"])
    metadata = {"source": {"session_id": "session_a", "image_id": "image_0000"}}
    atomic_json(store.root / DRAFTS_DIR / revision / METADATA_FILE, metadata)
    atomic_json(store.root / LATEST_DIR / f"{identity}.json", {"revision": revision})
    foreign_database = tmp_path / "2026-10-24" / "session_b" / "metadata.sqlite3"
    with closing(sqlite3.connect(foreign_database)) as connection, connection:
        connection.execute(
            """INSERT INTO sam3_review_revisions(
            revision, session_id, image_id, prediction_id, tag_revision, source_sha256,
            source_path, metadata_json, annotation_json, is_current, status, approval_id, approval_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'approved', ?, ?)""",
            (
                revision,
                "session_b",
                "image_0000",
                "prediction",
                1,
                "source-hash",
                "2026-10-24/session_b/raw/image.png",
                "{}",
                "{}",
                "approval",
                json.dumps({"scope": "yolo"}),
            ),
        )
    selection = SelectiveInferenceServer(source, store, reviews)
    request = CatalogRequest()

    selection._get(request)  # type: ignore[arg-type]

    response: dict[str, Any] = json.loads(request.wfile.getvalue())
    item = next(value for value in response["items"] if value["session_id"] == "session_a")
    assert request.status == HTTP_OK_STATUS
    assert item["yolo"] == "未処理"
    assert item["dino"] == "未処理"
    with closing(sqlite3.connect(foreign_database)) as connection:
        foreign_review = connection.execute(
            "SELECT status, approval_id FROM sam3_review_revisions WHERE revision=?",
            (revision,),
        ).fetchone()
    assert foreign_review == ("approved", "approval")


def test_both_viewers_define_fifty_item_pages() -> None:
    """選択画面・レビュー画面とも一覧を50件単位で表示する。"""
    project = Path(__file__).resolve().parents[1]
    selection_html = (project / "src/sam3_pipeline/interface/selection.html").read_text(encoding="utf-8")
    review_html = (project / "src/sam3_pipeline/interface/review.html").read_text(encoding="utf-8")

    assert "const pageSize = 50" in selection_html
    assert "const galleryPageSize = 50" in review_html
    assert "let selected = new Set()" in selection_html
    assert 'id="finish-work"' in review_html
    assert "window.location.assign(window.selectionReturnUrl)" in review_html


def test_date_filters_sessions_and_resets_visible_page() -> None:
    """日付切替はセッション候補と表示を更新し、選択状態を維持する。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.jsがないため選択画面のJavaScript検証をスキップ")

    items = [
        {
            "capture_date": "2026-10-23",
            "session_id": "session_a",
            "image_id": f"image_{index:04}",
            "source_path": f"2026-10-23/session_a/raw/image_{index:04}.png",
            "width": 8,
            "height": 8,
            "draft": False,
            "yolo": "未処理",
            "dino": "未処理",
        }
        for index in range(51)
    ]
    items.append(
        {
            "capture_date": "2026-10-23",
            "session_id": "session_b",
            "image_id": "image_other_first_date_session",
            "source_path": "2026-10-23/session_b/raw/image.png",
            "width": 8,
            "height": 8,
            "draft": False,
            "yolo": "未処理",
            "dino": "未処理",
        }
    )
    items.extend(
        {
            "capture_date": "2026-10-24",
            "session_id": "session_a",
            "image_id": f"image_second_{index:04}",
            "source_path": f"2026-10-24/session_a/raw/image_second_{index:04}.png",
            "width": 8,
            "height": 8,
            "draft": False,
            "yolo": "未処理",
            "dino": "未処理",
        }
        for index in range(51)
    )
    items.extend(
        [
            {
                "capture_date": "2026-10-25",
                "session_id": "session_c",
                "image_id": "image_fallback",
                "source_path": "2026-10-25/session_c/raw/image.png",
                "width": 8,
                "height": 8,
                "draft": False,
                "yolo": "未処理",
                "dino": "未処理",
            },
            {
                "capture_date": "2026-10-25",
                "session_id": "session_z",
                "image_id": "image_other_session",
                "source_path": "2026-10-25/session_z/raw/image.png",
                "width": 8,
                "height": 8,
                "draft": False,
                "yolo": "未処理",
                "dino": "未処理",
            },
        ]
    )
    script = SELECTION_HTML.read_text(encoding="utf-8").split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run(
        [node, "-e", NODE_SELECTION_UI],
        input=json.dumps({"script": script, "items": items}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=10,
    )
    response = json.loads(result.stdout)

    assert response["dates"] == ["2026-10-23", "2026-10-24", "2026-10-25"]
    assert response["firstDateSessions"] == ["session_a", "session_b"]
    assert response["dateDisabled"] is False
    assert response["sessionDisabled"] is False
    assert response["secondDate"]["session"] == "session_a"
    assert response["secondDate"]["sessions"] == ["session_a"]
    assert response["secondDate"]["page"] == "1 / 2 ページ"
    assert "image_second_0000" in response["secondDate"]["grid"]
    assert "image_0000" not in response["secondDate"]["grid"]
    assert response["secondDate"]["count"] == "選択 1 件"
    assert response["fallbackSessions"] == ["session_c", "session_z"]
    assert response["fallbackSession"] == "session_c"
    assert "image_fallback" in response["fallbackGrid"]
    assert "image_other_session" not in response["fallbackGrid"]

    empty_result = subprocess.run(
        [node, "-e", NODE_SELECTION_UI],
        input=json.dumps({"script": script, "items": []}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=10,
    )
    empty_response = json.loads(empty_result.stdout)
    assert empty_response["dates"] == []
    assert empty_response["firstDateSessions"] == []
    assert empty_response["dateDisabled"] is True
    assert empty_response["sessionDisabled"] is True
