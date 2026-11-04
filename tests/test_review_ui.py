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


@pytest.mark.parametrize("confidence,filled", [("low", 1), ("medium", 2), ("high", 3)])
@pytest.mark.parametrize(
    "approved,statuses,label,dino_approved",
    [
        (False, [], "対象なし", False),
        (True, [], "対象なし", False),
        (False, ["pending"], "未判定", False),
        (True, ["pending"], "未判定", False),
        (True, ["accepted", "pending"], "判定途中", False),
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
        "reviewed": True,
        "objects": [{"dino_status": status} for status in statuses],
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
