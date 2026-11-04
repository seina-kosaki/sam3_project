"""タグ単位のYOLO注釈契約を検証する。

責務:
    採用矩形と撮影タグの一対一対応を検証し、DINOの推論原本は変更しない。
処理の流れ:
    1. 新しいタグレビュー形式の版を確認する。
    2. 採用矩形の器具UID・個体番号を撮影タグと照合する。
    3. 比較用の未採用候補に対応情報が混入していないことを確認する。
関連ファイル:
    annotation_store.py、review_server.py、interface/tag_review.js。
設計意図:
    保存と承認で同じ検証を使い、旧形式の注釈は変更せずに維持する。
"""

from __future__ import annotations

from typing import Any

from sam3_pipeline.domain.annotation import UNRESOLVED_LABEL
from sam3_pipeline.domain.capture import CaptureError

TAG_REVIEW_VERSION = 1


def validate_tag_review(annotation: dict[str, Any], tags: list[dict[str, Any]]) -> None:
    """採用矩形のタグ対応と未採用候補の分離を検証する。

    Args:
        annotation: 保存または承認対象のLabelme注釈。
        tags: 正本に記録された撮影タグ。

    Raises:
        CaptureError: 版、タグ対応、重複または未採用候補が不正な場合。
    """
    # 既存承認のhashと公開形式を壊さず、新形式へ明示移行した注釈だけを厳密検証する。
    if "tag_review_version" not in annotation:
        return
    version = annotation["tag_review_version"]
    if type(version) is not int or version != TAG_REVIEW_VERSION:
        raise CaptureError("タグレビュー形式の版が不正である")
    allowed = {(tag["instrument_uid"], tag["instance_no"]) for tag in tags if tag["visible"]}
    assigned: set[tuple[str, int]] = set()
    shapes = annotation.get("shapes")
    candidates = annotation.get("review_candidates")
    if not isinstance(shapes, list) or not isinstance(candidates, list):
        raise CaptureError("採用bboxと未採用候補の一覧が必要である")
    # 器具UIDだけでは同種器具の複数個体を区別できないため、個体番号との組で比較する。
    for shape in shapes:
        identity = shape.get("tag_assignment") if isinstance(shape, dict) else None
        if not isinstance(identity, dict):
            raise CaptureError("採用bboxには器具タグの対応が必要である")
        uid, instance = identity.get("instrument_uid"), identity.get("instance_no")
        if not isinstance(uid, str) or type(instance) is not int:
            raise CaptureError("器具UIDと個体番号が不正である")
        key = (uid, instance)
        if key not in allowed or shape.get("label") != uid:
            raise CaptureError("採用bboxが現在の可視タグと一致しない")
        if key in assigned:
            raise CaptureError("一つのタグに複数のbboxは採用できない")
        assigned.add(key)
    # 比較用候補は学習用shapesから隔離し、古い対応を残して誤表示させない。
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("label") != UNRESOLVED_LABEL or candidate.get("tag_assignment") is not None:
            raise CaptureError("未採用候補にタグ対応を保存できない")
