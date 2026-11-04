"""タグ単位の対応付けと保存契約を実際のJavaScript・Pythonで検証する。

責務:
    一対一対応、旧形式の読み込み、手描きと未採用候補の分離を検証する。
処理の流れ:
    1. 同一UIDの複数個体と複数候補を用意する。
    2. 配信されるJavaScriptの編集処理をNode.jsで実行する。
    3. 保存結果と不正入力に対するサーバー側検証を照合する。
関連ファイル:
    interface/tag_review.js、infrastructure/tag_annotations.py。
設計意図:
    DOMと推論環境に依存せず、画面と保存側の対応規則を直接検証する。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from sam3_pipeline.domain.capture import CaptureError
from sam3_pipeline.infrastructure.tag_annotations import validate_tag_review

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "sam3_pipeline" / "interface" / "tag_review.js"
TAGS = [{"instrument_uid": "tool", "instance_no": index, "visible": True} for index in (1, 2)]
RUN_EDITOR = """
const vm = require("node:vm"), fs = require("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const sandbox = vm.createContext({structuredClone});
vm.runInContext(input.script, sandbox);
sandbox.item = input.item;
vm.runInContext("const editor = new TagReview.Editor(item);", sandbox);
const result = vm.runInContext(input.action, sandbox);
process.stdout.write(JSON.stringify(result));
"""


def _shape(index: int, instance: int | None = None) -> dict[str, Any]:
    """個体番号の有無を切り替えた旧形式の候補を返す。

    Args:
        index: 候補番号。
        instance: 旧形式へ記録する個体番号。

    Returns:
        小さな矩形を持つ候補。
    """
    return {"label": "tool" if instance is not None else "__unresolved__", "points": [[index, index], [10 + index, 10 + index]], "group_id": index, "shape_type": "rectangle", "flags": {}, "description": json.dumps({"instance_no": instance})}


def _edit(action: str, shapes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """配信する編集ロジックを隔離環境で実行し、JSON結果を返す。

    Args:
        action: 編集操作とJSONで返す結果を表すJavaScript式。
        shapes: 省略時は対応済みと未対応の候補を使う。

    Returns:
        実際のJavaScriptが生成した結果。

    Raises:
        subprocess.CalledProcessError: 編集処理の実行に失敗した場合。
    """
    # 既存Node.jsを使い、ブラウザーやモデルの依存をテストへ追加しない。
    node = shutil.which("node")
    if node is None:
        pytest.skip("タグレビューのJavaScript検証にはNode.jsが必要である")
    item = {"tags": TAGS, "annotation": {"shapes": shapes if shapes is not None else [_shape(0, 1), _shape(1)]}}
    result = subprocess.run([node, "-e", RUN_EDITOR], input=json.dumps({"script": SCRIPT.read_text(encoding="utf-8"), "item": item, "action": action}), text=True, encoding="utf-8", capture_output=True, check=True, timeout=10)
    return json.loads(result.stdout)


def test_tag_review_click_selects_matching_individual_and_overlaps() -> None:
    """同じUIDの個体を区別し、重なった矩形を全て選択候補として返す。"""
    # 2個体へ別々の既存矩形を対応付ける。
    result = _edit(
        """(() => {
      editor.selectBox(1);
      return {tag: editor.selectedTag, hits: editor.hitBoxes({x: 5, y: 5}), colors: [TagReview.color(0), TagReview.color(1)]};
    })()""",
        [_shape(0, 1), _shape(1, 2)],
    )

    assert result["tag"] == 1
    assert result["hits"] == [0, 1]
    assert result["colors"][0] != result["colors"][1]


def test_tag_review_adoption_replaces_one_box_and_keeps_candidate() -> None:
    """採用し直すと旧矩形を候補へ戻し、個体と新矩形だけを対応付ける。"""
    result = _edit("""(() => {
      const needsReplacement = editor.needsReplacement(1);
      editor.adopt(1);
      return {needsReplacement, annotation: editor.annotation(item.annotation)};
    })()""")

    annotation = result["annotation"]
    validate_tag_review(annotation, TAGS)
    assert result["needsReplacement"]
    assert len(annotation["shapes"]) == 1
    assert annotation["shapes"][0]["group_id"] == 1
    assert annotation["shapes"][0]["tag_assignment"] == {"instrument_uid": "tool", "instance_no": 1}
    assert annotation["review_candidates"][0]["tag_assignment"] is None


def test_tag_review_drawing_is_yolo_only_and_survives_reload() -> None:
    """手描きは候補番号を流用せず、保存後も個体対応と旧候補を保持する。"""
    result = _edit("""(() => {
      editor.selectedTag = 1;
      editor.draw([[30, 30], [40, 40]]);
      const annotation = editor.annotation(item.annotation);
      const reloaded = new TagReview.Editor({...item, annotation});
      return {annotation, mode: editor.mode, assigned: reloaded.assigned(1), original: item.annotation,
        label: TagReview.boxName(editor.entries[2].shape, 2), reloadedLabel: TagReview.boxName(reloaded.entries[1].shape, 1)};
    })()""")

    annotation = result["annotation"]
    validate_tag_review(annotation, TAGS)
    assert len(annotation["shapes"]) == len(TAGS)
    assert annotation["shapes"][1]["tag_assignment"]["instance_no"] == TAGS[1]["instance_no"]
    assert annotation["shapes"][1]["group_id"] is None
    assert json.loads(annotation["shapes"][1]["description"]) == {"manual": True}
    assert result["mode"] == "confirm"
    assert result["assigned"] == [1]
    assert result["label"] == result["reloadedLabel"]
    assert "tag_review_version" not in result["original"]


def test_tag_review_ambiguous_legacy_uid_is_not_guessed() -> None:
    """個体番号のない同種器具は勝手に先頭タグへ対応付けない。"""
    shape = _shape(0)
    shape["label"] = "tool"

    result = _edit("editor.annotation(item.annotation)", [shape])

    assert not result["shapes"]
    assert len(result["review_candidates"]) == 1


def test_tag_review_reassigning_box_clears_previous_tag_and_box() -> None:
    """使用中のbboxを付け替えると旧タグ・対象タグの旧bboxの両方を解除する。"""
    result = _edit(
        """(() => {
      editor.selectedTag = 1;
      const needsReplacement = editor.needsReplacement(0);
      editor.adopt(0);
      return {needsReplacement, first: editor.assigned(0), second: editor.assigned(1),
        annotation: editor.annotation(item.annotation)};
    })()""",
        [_shape(0, 1), _shape(1, 2)],
    )

    validate_tag_review(result["annotation"], TAGS)
    assert result["needsReplacement"]
    assert result["first"] == []
    assert result["second"] == [0]
    assert result["annotation"]["review_candidates"][0]["group_id"] == 1


@pytest.mark.parametrize("damage", ["duplicate", "unknown", "label", "boolean", "candidate", "version"])
def test_tag_review_invalid_assignments_are_rejected(damage: str) -> None:
    """APIを直接呼んでも個体重複やタグの偽装は保存できない。

    Args:
        damage: 不正に変更する項目。
    """
    annotation = {"tag_review_version": 1, "shapes": [_shape(0, 1)], "review_candidates": []}
    shape = annotation["shapes"][0]
    shape["tag_assignment"] = {"instrument_uid": "tool", "instance_no": 1}
    # 新形式の各不正入力を同じ検証境界へ渡す。
    if damage == "duplicate":
        annotation["shapes"].append(deepcopy(shape))
    elif damage == "unknown":
        shape["tag_assignment"]["instance_no"] = 3
    elif damage == "label":
        shape["label"] = "other"
    elif damage == "boolean":
        shape["tag_assignment"]["instance_no"] = True
    elif damage == "candidate":
        annotation["review_candidates"].append(deepcopy(shape))
    else:
        annotation["tag_review_version"] = True

    with pytest.raises(CaptureError):
        validate_tag_review(annotation, TAGS)


def test_dino_editor_manual_selection_keeps_one_candidate_per_tag() -> None:
    """DINOの手動選択は同種個体を区別し、付け替え時に旧候補を不採用へ戻す。"""
    result = _edit("""(() => {
      const item = {tags: [
        {instrument_uid: 'tool', instance_no: 1, visible: true},
        {instrument_uid: 'tool', instance_no: 2, visible: true}],
        objects: [0, 1, 2].map(object_index => ({object_index, instrument_uid: null, instance_no: null, dino_status: 'pending'}))};
      const editor = new TagReview.DinoEditor(item);
      editor.adopt(item.objects[0]);
      const replace = editor.needsReplacement(item.objects[1]);
      editor.adopt(item.objects[1]);
      editor.selectedTag = 1;
      editor.adopt(item.objects[2]);
      editor.select(item.objects[1]);
      return {replace, selectedTag: editor.selectedTag, objects: item.objects};
    })()""")

    assert result["replace"]
    assert result["selectedTag"] == 0
    assert [item["dino_status"] for item in result["objects"]] == ["rejected", "accepted", "accepted"]
    assert result["objects"][0]["instrument_uid"] is None
    assert [item["instance_no"] for item in result["objects"]] == [None, 1, TAGS[1]["instance_no"]]
