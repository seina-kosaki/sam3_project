"""開始画面の実際のJavaScriptで信頼度とモデル別承認表示を検証する。

責務:
    保存済み状態から生成するカード表示を検証し、実データやモデルは操作しない。
処理の流れ:
    1. レビューHTMLのスクリプトをNode.jsの隔離コンテキストへ読み込む。
    2. 最小限のDOM代替と画像データで一覧描画を実行する。
    3. 信頼度メーターとモデル別承認・物体件数を照合する。
関連ファイル:
    src/sam3_pipeline/interface/review.html。
設計意図:
    表示ロジックをPythonに複製せず、実際に配信するスクリプトを回帰検証する。

Note:
    Node.jsがない環境ではスキップする。レイアウトはブラウザーで別途検証する。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REVIEW_HTML = Path(__file__).resolve().parents[1] / "src" / "sam3_pipeline" / "interface" / "review.html"
TAG_REVIEW_JS = REVIEW_HTML.with_name("tag_review.js")
NODE_RENDER = """
const vm = require("node:vm");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const elements = new Map();
const document = {
  querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, {
      innerHTML: "", value: "", getContext: () => ({}),
      addEventListener() {}, querySelectorAll: () => []
    });
    return elements.get(selector);
  },
  querySelectorAll: () => []
};
const sandbox = vm.createContext({
  document, window: {addEventListener() {}},
  fetch: async () => ({ok: true, headers: {get: () => "application/json"},
    json: async () => ({items: [], sessions: []})})
});
vm.runInContext(input.script, sandbox);
sandbox.testItem = input.item;
vm.runInContext('items = [testItem]; selectedSession = testItem.session_id; renderGallery();', sandbox);
process.stdout.write(elements.get("#gallery").innerHTML);
"""
NODE_SAVE = """
const vm = require("node:vm");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const elements = new Map(), posts = [];
let confirmations = 0;
const document = {
  querySelector(selector) {
    if (!elements.has(selector)) {
      const callbacks = {};
      const element = {
        innerHTML: "", textContent: "", value: "operator", disabled: false,
        getContext: () => ({}), querySelectorAll: () => [],
        addEventListener(event, callback) { callbacks[event] = callback; },
        showModal() { confirmations++; element.returnValue = input.answer; callbacks.close(); },
        close() {}, callbacks
      };
      elements.set(selector, element);
    }
    return elements.get(selector);
  },
  querySelectorAll: () => []
};
const sandbox = vm.createContext({
  document, structuredClone, performance, console, window: {addEventListener() {}},
  fetch: async () => ({ok: true, headers: {get: () => "application/json"},
    json: async () => ({items: [], sessions: []})})
});
(async () => {
  vm.runInContext(input.script, sandbox);
  await new Promise(resolve => setImmediate(resolve));
  sandbox.testItem = input.item;
  sandbox.latest = input.latest;
  sandbox.posts = posts;
  vm.runInContext(`
    items = [structuredClone(testItem)];
    selectedItem = structuredClone(testItem);
    selectedSession = testItem.session_id;
    viewedObjects = new Set(testItem.objects.map(object => object.object_index));
    loadItems = async () => false;
    request = async (url, method, payload) => {
      if (method === "POST") { posts.push(payload); return {}; }
      return {items: [latest]};
    };
  `, sandbox);
  vm.runInContext("renderGallery()", sandbox);
  await elements.get("#save-object-review").callbacks.click();
  process.stdout.write(JSON.stringify({
    confirmations, posts, message: document.querySelector("#object-message").textContent,
    gallery: document.querySelector("#gallery").innerHTML,
    current: vm.runInContext("items[0]", sandbox),
    selected: vm.runInContext("selectedItem", sandbox)
  }));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.parametrize("confidence,filled", [("low", 1), ("medium", 2), ("high", 3)])
@pytest.mark.parametrize(
    "approved,statuses,label,dino_approved",
    [
        (False, [], "対象なし", False),
        (True, [], "対象なし", False),
        (False, ["pending"], "未判定", False),
        (True, ["pending"], "未判定", False),
        (True, ["accepted", "pending"], "判定途中", False),
        (False, ["accepted", "pending"], "✓ 承認済", True),
        (False, ["accepted"], "未承認", False),
        (True, ["accepted"], "✓ 承認済", True),
        (True, ["accepted", "rejected"], "✓ 承認済", True),
        (True, ["rejected"], "判定済（採用なし）", False),
    ],
)
def test_review_gallery_separates_confidence_and_approval(
    *,
    confidence: str,
    filled: int,
    approved: bool,
    statuses: list[str],
    label: str,
    dino_approved: bool,
) -> None:
    """信頼度によらずモデル別承認と物体判定の内訳を独立表示する。

    Args:
        confidence: 推論信頼度。
        filled: メーターの点灯本数。
        approved: 保存済みの画像承認状態。
        statuses: 保存済みの物体採否。
        label: DINO状態の期待表示。
        dino_approved: DINO承認色の期待有無。
    """
    # ブラウザー依存を増やさず、既存Node.jsで配信スクリプトそのものを実行する。
    node = shutil.which("node")
    if node is None:
        pytest.skip("レビューUIのJavaScript検証にはNode.jsが必要である")
    script = TAG_REVIEW_JS.read_text(encoding="utf-8") + "\n" + REVIEW_HTML.read_text(encoding="utf-8").split("<script>", 1)[1].split("</script>", 1)[0]
    item = {
        "revision": "revision",
        "session_id": "session",
        "capture_date": "2026-10-30",
        "image_id": "image",
        "width": 8,
        "height": 8,
        "tags": [],
        "annotation": {"shapes": []},
        "confidence": confidence,
        "reasons": [],
        "approved": approved,
        "dino_approved": dino_approved,
        "reviewed": True,
        "objects": [{"dino_status": status, "saved_mask_exists": True} for status in statuses],
    }

    result = subprocess.run(
        [node, "-e", NODE_RENDER],
        input=json.dumps({"script": script, "item": item}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=10,
    )

    # 色だけでなく文字・段階数・内訳を検証し、一覧確認と承認の混同も防ぐ。
    html = result.stdout
    yolo_label = "✓ 承認済" if approved else "未承認"
    yolo_class = "approved" if approved else ""
    dino_class = "approved" if dino_approved else ""
    assert f'<strong>YOLO</strong><span class="approval-badge {yolo_class}">{yolo_label}</span>' in html
    assert f'<strong>DINO</strong><span class="approval-badge {dino_class}">{label}</span>' in html
    assert f"採用 {statuses.count('accepted')} / 不採用 {statuses.count('rejected')} / 未判定 {statuses.count('pending')}" in html
    assert html.count('class="filled"') == filled
    assert f'class="confidence {confidence}"' in html
    assert " · 承認済" not in html


@pytest.mark.parametrize(
    "existing,answer,confirmed,saved",
    [
        ([], "yes", False, True),
        ([0, 2], "no", False, True),
        ([1], "yes", True, True),
        ([1], "no", True, False),
        ([1], "", True, False),
    ],
)
def test_dino_save_uses_unsaved_decisions_and_latest_index(
    existing: list[int], answer: str, *, confirmed: bool, saved: bool,
) -> None:
    """配信中の保存ハンドラーで採用先だけ確認し、No/EscapeはPOSTしない。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("レビューUIのJavaScript検証にはNode.jsが必要である")
    script = TAG_REVIEW_JS.read_text(encoding="utf-8") + "\n" + REVIEW_HTML.read_text(encoding="utf-8").split("<script>", 1)[1].split("</script>", 1)[0]
    item = {
        "revision": "revision", "session_id": "session", "image_id": "image",
        "capture_date": "2026-10-30", "width": 8, "height": 8,
        "tags": [], "annotation": {"shapes": []}, "confidence": "high",
        "reasons": [], "candidate_confidence": [], "approved": False,
        "dino_approved": False, "reviewed": False, "published": False,
        "published_other_version": False,
        "objects": [
            {"object_index": index, "dino_status": "accepted" if index == 1 else "rejected", "instrument_uid": "tool", "instance_no": 1}
            for index in range(3)
        ],
    }
    latest = {
        "revision": "revision",
        "objects": [
            {"object_index": index, "dino_status": "rejected" if index == 1 else "accepted", "saved_mask_exists": index in existing}
            for index in (2, 0, 1)
        ],
    }

    result = subprocess.run(
        [node, "-e", NODE_SAVE],
        input=json.dumps({"script": script, "item": item, "latest": latest, "answer": answer}),
        text=True, encoding="utf-8", capture_output=True, check=True, timeout=10,
    )
    actual = json.loads(result.stdout)

    assert actual["confirmations"] == int(confirmed)
    assert len(actual["posts"]) == int(saved)
    assert actual["selected"] == item
    if saved:
        assert actual["posts"][0]["overwrite_masks"] == confirmed
        assert actual["posts"][0]["decisions"] == [{"object_index": obj["object_index"], "status": obj["dino_status"]} for obj in item["objects"]]
        assert "✓ 承認済" in actual["gallery"]
    else:
        assert actual["current"] == item
        assert "保存を中止" in actual["message"]


NODE_HARNESS = """
const vm = require("node:vm");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
const elements = new Map(), log = {posts: [], closes: 0, opened: [], strokes: []};
function makeElement() {
  const callbacks = {};
  const element = {
    innerHTML: "", textContent: "", value: "operator", disabled: false, hidden: false, checked: false, style: {},
    querySelectorAll: () => [], setAttribute() {}, focus() {},
    addEventListener(event, callback) { callbacks[event] = callback; },
    showModal() { element.open = true; }, close() { element.open = false; log.closes++; }, callbacks
  };
  element.getContext = () => {
    const ctx = {clearRect() {}, drawImage() {}, setLineDash() {}, strokeText() {}, fillText() {}, beginPath() {}, arc() {}, fill() {}, stroke() {},
      strokeRect(...args) { log.strokes.push({style: ctx.strokeStyle, width: ctx.lineWidth, rect: args}); }};
    return ctx;
  };
  return element;
}
const document = {
  querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, makeElement());
    return elements.get(selector);
  },
  querySelectorAll: () => []
};
const sandbox = vm.createContext({
  document, structuredClone, performance, console, log, input, setImmediate,
  Image: class { constructor() { this.complete = false; } },
  window: {addEventListener() {}, confirm: () => true},
  fetch: async () => ({ok: true, headers: {get: () => "application/json"}, json: async () => ({items: [], sessions: []})})
});
(async () => {
  vm.runInContext(input.script, sandbox);
  await new Promise(resolve => setImmediate(resolve));
  const result = await vm.runInContext("(async () => {" + input.body + "})()", sandbox);
  process.stdout.write(JSON.stringify({result, log, message: document.querySelector("#tag-message").textContent || document.querySelector("#object-message").textContent,
    notice: document.querySelector("#notice").textContent, noticeClass: document.querySelector("#notice").className,
    counts: document.querySelector("#publish-counts").textContent, publishDisabled: document.querySelector("#publish-button").disabled}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


def _run_ui(body: str, item: dict[str, object]) -> dict[str, object]:
    """配信スクリプトを読み込んだ隔離環境で任意の操作を実行する。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("レビューUIのJavaScript検証にはNode.jsが必要である")
    script = TAG_REVIEW_JS.read_text(encoding="utf-8") + "\n" + REVIEW_HTML.read_text(encoding="utf-8").split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run(
        [node, "-e", NODE_HARNESS],
        input=json.dumps({"script": script, "body": f"const testItem = {json.dumps(item)};\n{body}"}),
        text=True, encoding="utf-8", capture_output=True, check=True, timeout=10,
    )
    return json.loads(result.stdout)


DINO_ITEM = {
    "revision": "r1", "session_id": "session", "capture_date": "2026-10-30", "image_id": "image", "image_url": "/api/image/r1",
    "confidence": "high", "reviewed": True, "approved": True, "width": 8, "height": 8, "tags": [], "annotation": {"shapes": []},
    "objects": [
        {"object_index": 0, "dino_status": "accepted", "instrument_uid": "tool", "instance_no": 1, "bbox": [0, 0, 2, 2]},
        {"object_index": 1, "dino_status": "pending", "instrument_uid": None, "instance_no": None, "bbox": [0, 0, 2, 2]},
    ],
}
DINO_SETUP = """
const next = {...structuredClone(testItem), revision: "r2"};
items = [structuredClone(testItem), structuredClone(next)];
selectedItem = structuredClone(testItem);
viewedObjects = new Set();
loadItems = async () => true;
openDetail = item => log.opened.push(item.revision);
const latest = {items: [structuredClone(testItem), next]};
"""


def test_dino_save_requires_an_accepted_mask() -> None:
    """採用候補が0件なら保存せず、現在の画像に留まる。"""
    actual = _run_ui(DINO_SETUP + """
selectedItem.objects = selectedItem.objects.map(object => ({...object, dino_status: "rejected"}));
viewedObjects = new Set();
request = async (url, method, payload) => { if (method === "POST") { log.posts.push(payload); return {}; } return latest; };
await document.querySelector("#save-object-review").callbacks.click();
""", DINO_ITEM)

    assert actual["log"]["posts"] == []
    assert "採用されたmaskがない" in actual["message"]
    assert actual["log"]["opened"] == []


def test_dino_partial_save_skips_pending_candidates_and_moves_next() -> None:
    """採用maskを確認すれば、未確認の別候補を残したまま保存できる。"""
    actual = _run_ui(DINO_SETUP + """
request = async (url, method, payload) => { if (method === "POST") log.posts.push(payload); return latest; };
viewedObjects = new Set([0]);
await document.querySelector("#save-object-review").callbacks.click();
return {dino: items[0].dino_approved, timings: window.reviewSaveTimings.length};
""", DINO_ITEM)

    post = actual["log"]["posts"][0]
    assert post["decisions"] == [{"object_index": 0, "status": "accepted"}]
    assert post["assignments"] == [{"object_index": 0, "instrument_uid": "tool", "instance_no": 1}]
    assert actual["log"]["opened"] == ["r2"]
    assert actual["result"]["dino"] is True


def test_dino_save_failure_stays_and_double_click_posts_once() -> None:
    """保存中の再クリックを送信せず、保存失敗時は現在画像に留まりエラーを示す。"""
    actual = _run_ui(DINO_SETUP + """
let release;
request = async (url, method, payload) => {
  if (method !== "POST") return latest;
  log.posts.push(payload);
  await new Promise(resolve => { release = resolve; });
  throw new Error("simulated save failure");
};
viewedObjects = new Set([0]);
const first = document.querySelector("#save-object-review").callbacks.click();
while (!release) await new Promise(resolve => setImmediate(resolve));
const busy = savingReview;
const busyMessage = document.querySelector("#object-message").textContent;
await document.querySelector("#save-object-review").callbacks.click();
release();
await first;
return {busy, busyMessage, saving: savingReview, objects: items[0].objects};
""", DINO_ITEM)

    assert len(actual["log"]["posts"]) == 1
    assert actual["result"]["busy"] is True and "保存中" in actual["result"]["busyMessage"]
    assert actual["result"]["saving"] is False
    assert actual["message"] == "simulated save failure"
    assert actual["log"]["opened"] == [] and actual["log"]["closes"] == 0
    assert actual["result"]["objects"] == DINO_ITEM["objects"]


def test_dino_editor_rejects_other_candidates_for_the_selected_tag() -> None:
    """一つ採用すると、同じタグの残り候補だけ自動で不採用になる。"""
    actual = _run_ui("""
const tags = [
  {instrument_uid: "tool", instance_no: 1, visible: true},
  {instrument_uid: "tool", instance_no: 2, visible: true}
];
const objects = [
  {object_index: 0, dino_status: "pending", instrument_uid: "tool", instance_no: 1},
  {object_index: 1, dino_status: "pending", instrument_uid: "tool", instance_no: 1},
  {object_index: 2, dino_status: "pending", instrument_uid: "tool", instance_no: 2}
];
const editor = new TagReview.DinoEditor({tags, objects});
editor.adopt(objects[0]);
return objects.map(object => [object.dino_status, object.instrument_uid, object.instance_no]);
""", DINO_ITEM)

    assert actual["result"] == [
        ["accepted", "tool", 1],
        ["rejected", None, None],
        ["pending", "tool", 2],
    ]


def test_yolo_session_bulk_confirmation_ignores_filter_and_pagination() -> None:
    """信頼度・表示ページによらず、選択セッションのbboxあり未確認画像を一括確認する。"""
    actual = _run_ui("""
items = Array.from({length: 51}, (_, index) => ({
  ...structuredClone(testItem), revision: `skip-${index}`, session_id: "selected",
  confidence: "high", reviewed: true, approved: false, reasons: [],
  annotation: {shapes: [{label: "tool", points: [[0, 0], [2, 2]]}]}
}));
items[0] = {...items[0], revision: "high-first-page", reviewed: false};
items[2] = {...items[2], revision: "already-reviewed", reviewed: true};
items[3] = {...items[3], revision: "empty", reviewed: false, annotation: {shapes: []}};
items[50] = {...items[50], revision: "high-second-page", reviewed: false};
items.push({...structuredClone(items[0]), revision: "low-hidden-by-filter", confidence: "low", reviewed: false});
items.push({...structuredClone(items[0]), revision: "other-session", session_id: "other"});
selectedSession = "selected";
currentFilter = "high";
galleryPage = 0;
renderGallery();
request = async (url, method, payload) => { if (method === "POST") log.posts.push(payload); return {}; };
await document.querySelector("#confirm-session-yolo").callbacks.click();
return items.map(item => [item.revision, item.reviewed]);
""", DINO_ITEM)

    assert [post["revision"] for post in actual["log"]["posts"]] == [
        "high-first-page", "high-second-page", "low-hidden-by-filter",
    ]
    reviewed = dict(actual["result"])
    assert reviewed["high-first-page"] is True
    assert reviewed["high-second-page"] is True
    assert reviewed["low-hidden-by-filter"] is True
    assert reviewed["already-reviewed"] is True
    assert reviewed["empty"] is False
    assert reviewed["other-session"] is False
    assert "3件" in actual["notice"]
    assert actual["noticeClass"] == "success"


def test_yolo_session_bulk_confirmation_reports_partial_failures() -> None:
    """画像単位の失敗を隠さず、成功分だけ確認済みに更新する。"""
    actual = _run_ui("""
items = [
  {...structuredClone(testItem), revision: "ok", session_id: "selected", reviewed: false, approved: false,
   reasons: [], annotation: {shapes: [{label: "tool", points: [[0, 0], [2, 2]]}]}},
  {...structuredClone(testItem), revision: "failed", session_id: "selected", reviewed: false, approved: false,
   reasons: [], annotation: {shapes: [{label: "tool", points: [[0, 0], [2, 2]]}]}},
];
selectedSession = "selected";
renderGallery();
request = async (url, method, payload) => {
  if (method === "POST" && payload.revision === "failed") throw new Error("failed request");
  if (method === "POST") log.posts.push(payload);
  return {};
};
await document.querySelector("#confirm-session-yolo").callbacks.click();
return items.map(item => item.reviewed);
""", DINO_ITEM)

    assert actual["log"]["posts"] == [{"revision": "ok", "reviewed": True}]
    assert actual["result"] == [True, False]
    assert "1件成功、1件失敗" in actual["notice"]
    assert "failed request" in actual["notice"]
    assert actual["noticeClass"] == "error"


def test_yolo_empty_annotation_is_not_saved() -> None:
    """採用bboxがない空画像はYOLO注釈として保存しない。"""
    actual = _run_ui("""
items = [structuredClone(testItem)];
selectedItem = structuredClone(testItem);
tagEditor = new TagReview.Editor(selectedItem);
request = async (url, method, payload) => { if (method === "POST") log.posts.push(payload); return {}; };
await document.querySelector("#save-detail").callbacks.click();
""", DINO_ITEM)

    assert actual["log"]["posts"] == []
    assert "採用bboxがない" in actual["message"]


def test_yolo_manual_correction_without_candidates_is_saved() -> None:
    """候補がない画像でも人手でbboxを描いた場合は保存する。"""
    actual = _run_ui("""
items = [structuredClone(testItem)];
selectedItem = structuredClone(testItem);
tagEditor = new TagReview.Editor(selectedItem);
tagEditor.draw([[1, 1], [4, 4]]);
selectedItem.annotation = tagEditor.annotation(selectedItem.annotation);
tagEdits = true;
loadItems = async () => true;
request = async (url, method, payload) => {
  if (method === "POST") { log.urls.push(url); log.posts.push(payload); }
  return {};
};
log.urls = [];
await document.querySelector("#save-detail").callbacks.click();
return selectedItem.annotation.shapes.length;
""", {
        **DINO_ITEM,
        "tags": [{"instrument_uid": "tool", "instance_no": 1, "visible": True}],
        "annotation": {"shapes": []},
    })

    assert actual["result"] == 1
    assert actual["log"]["urls"] == ["/api/annotation", "/api/reviewed"]
    assert actual["log"]["posts"][0]["annotation"]["shapes"][0]["description"] == '{"manual":true}'


def test_yolo_draw_mode_hides_existing_and_outlines_new_box() -> None:
    """描画モードは既存bboxを描かず、今回の描画bboxを黄色＋濃色縁取りで示す。"""
    item = {**DINO_ITEM, "tags": [{"instrument_uid": "tool", "instance_no": 1, "visible": True}],
            "annotation": {"shapes": [{"label": "tool", "points": [[1, 1], [5, 5]], "shape_type": "rectangle", "group_id": None, "flags": {}}]}}
    actual = _run_ui("""
selectedItem = structuredClone(testItem);
tagEditor = new TagReview.Editor(selectedItem);
const fake = {naturalWidth: 8, naturalHeight: 8};
tagEditor.mode = "draw";
drawImage(detailCanvas, selectedItem, fake, true);
const hidden = log.strokes.length;
dragStart = {x: 2, y: 2}; previewPoint = {x: 6, y: 6};
drawImage(detailCanvas, selectedItem, fake, true);
const preview = log.strokes.slice(hidden);
log.strokes.length = 0;
dragStart = null; previewPoint = null;
tagEditor.selectedTag = 0;
tagEditor.draw([[2, 2], [6, 6]]);
drawnShapes.add(tagEditor.entries[tagEditor.entries.length - 1].shape);
tagEditor.mode = "draw";
drawImage(detailCanvas, selectedItem, fake, true);
const drawn = log.strokes.slice();
log.strokes.length = 0;
tagEditor.mode = "confirm";
drawImage(detailCanvas, selectedItem, fake, true);
return {hidden, preview, drawn, confirmCount: log.strokes.length, entries: tagEditor.entries.length};
""", item)
    result = actual["result"]

    assert result["hidden"] == 0
    assert [stroke["style"] for stroke in result["preview"]] == ["#111111", "#ffd400"]
    assert result["preview"][0]["width"] > result["preview"][1]["width"]
    assert [stroke["style"] for stroke in result["drawn"]] == ["#111111", "#ffd400"]
    # 表示切替だけでは既存bboxを失わず、確認モードで再表示される。
    assert result["entries"] == result["confirmCount"] == len(item["annotation"]["shapes"]) + 1


@pytest.mark.parametrize(
    "states,expected,disabled",
    [
        ([(True, False, False), (False, False, False)], "公開対象 1件／未確認 1件", False),
        ([(True, True, False), (False, False, False)], "公開対象 0件／未確認 1件（公開済み 1件）", True),
        ([(True, False, True), (True, False, False)], "公開対象 1件／未確認 0件（公開済み画像の変更版 1件は保留）", False),
    ],
)
def test_publish_counts_show_targets_and_unchecked(states: list[tuple[bool, bool, bool]], expected: str, *, disabled: bool) -> None:
    """公開前に対象・未確認件数を示し、対象0件では公開ボタンを無効にする。"""
    body = "items = " + json.dumps([
        {**DINO_ITEM, "revision": f"r{index}", "reviewed": reviewed, "approved": False, "published": published, "published_other_version": other}
        for index, (reviewed, published, other) in enumerate(states)
    ]) + "; selectedSession = 'session'; renderSummary();"
    actual = _run_ui(body, DINO_ITEM)

    assert actual["counts"] == expected
    assert actual["publishDisabled"] is disabled
