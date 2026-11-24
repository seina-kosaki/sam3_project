"""撮影セッションのYOLO bbox確認とDINO物体採否を読み取り専用で集計する。"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date as calendar_date
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import yaml
from PIL import Image, ImageDraw, ImageFont

from sam3_pipeline.domain.capture import CaptureError
from sam3_pipeline.infrastructure.capture_source import SqliteCaptureSource

LOGGER = logging.getLogger(__name__)
CONFIG_DIRECTORY = Path(__file__).parent / "configs"
ANALYZE_CONFIG = CONFIG_DIRECTORY / "analyze.yaml"
CAPTURE_CONFIG = CONFIG_DIRECTORY / "capture.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_DATASET_DB = PROJECT_ROOT.parent / "master" / "dataset.sqlite"
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
RUN_PATTERN = re.compile(r"run_(\d{4,})\Z")
METADATA_NAME = "metadata.sqlite3"
READ_ONLY_QUERY = "?mode=ro"
DEFAULT_SQLITE_TIMEOUT = 10.0
UNCLASSIFIED = "器具未確定"
TABLE_ROWS_PER_PAGE = 24
MAX_FAILURE_EXAMPLES = 3
RECTANGLE_POINT_COUNT = 2
BBOX_COORDINATE_COUNT = 4
MIN_SESSION_DATABASE_PARTS = 3
SESSION_PATH_PARTS = 2
TABLE_COLUMN_WIDTHS = (300, 120, 120, 120, 130, 160, 120, 120, 130, 150, 140, 210)
TABLE_CELL_PADDING = 12
TABLE_LINE_SPACING = 4
CHART_LABEL_WIDTH = 300
CHART_METRIC_WIDTH = 170
CHART_WIDTH = 760
CHART_RATE_WIDTH = 180
TABLE_COLUMNS = (
    "器具種別",
    "撮影器具数",
    "bbox成功数",
    "bbox失敗数",
    "bbox成功率",
    "YOLO未確認器具数",
    "mask成功数",
    "mask失敗数",
    "mask成功率",
    "DINO pending器具数",
    "未採用bbox数",
    "器具あたり平均未採用bbox数",
)


@dataclass
class Counts:
    """集計対象のbbox・mask状態を保持する。"""

    captured_instruments: int = 0
    bbox_success: int = 0
    bbox_failure: int = 0
    yolo_unconfirmed: int = 0
    mask_success: int = 0
    mask_failure: int = 0
    mask_pending: int = 0
    unaccepted_bbox_count: int = 0

    def row(self, uid: str, display_name: str | None = None) -> dict[str, str | int]:
        """日本語の表・CSVへ出力する行を作る。"""
        return {
            "器具種別": display_name or uid,
            "撮影器具数": self.captured_instruments,
            "bbox成功数": self.bbox_success,
            "bbox失敗数": self.bbox_failure,
            "bbox成功率": _format_percentage(self.bbox_success, self.captured_instruments),
            "YOLO未確認器具数": self.yolo_unconfirmed,
            "mask成功数": self.mask_success,
            "mask失敗数": self.mask_failure,
            "mask成功率": _format_percentage(self.mask_success, self.captured_instruments),
            "DINO pending器具数": self.mask_pending,
            "未採用bbox数": self.unaccepted_bbox_count,
            "器具あたり平均未採用bbox数": _format_average(
                self.unaccepted_bbox_count,
                self.captured_instruments,
            ),
        }


@dataclass
class FailureExample:
    """失敗画像の出力に必要な読み取り専用情報を保持する。"""

    date: str
    session_id: str
    image_id: str
    image_path: Path
    boxes: list[list[float]] = field(default_factory=list)
    initial_boxes: list[list[float]] = field(default_factory=list)
    mask_paths: list[str] = field(default_factory=list)
    no_detection: bool = False


@dataclass
class ImageContext:
    """画像一枚の集計に必要な読み取り専用情報をまとめる。"""

    date: str
    session_id: str
    image_id: str
    image_path: Path
    expected: dict[str, set[int]]
    review: dict[str, Any]
    counts: dict[str, Counts]
    failures: dict[str, dict[str, list[FailureExample]]]
    mask_paths: dict[int, str]
    captured_units_registered: bool = False


@dataclass
class BBoxState:
    """画像内の初期bbox、承認shape、差分集計を保持する。"""

    originals: dict[int, dict[str, Any]]
    annotations: dict[int, dict[str, Any]]
    additions: list[dict[str, Any]]
    review_candidates: list[dict[str, Any]]
    initial_boxes_by_uid: dict[str, list[list[float]]] = field(default_factory=lambda: defaultdict(list))
    boxes_by_uid: dict[str, list[list[float]]] = field(default_factory=lambda: defaultdict(list))
    accepted_units: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    unaccepted_by_uid: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class AnalysisContext:
    """セッション横断で共有する解析元・全体集計・失敗例を保持する。"""

    source: SqliteCaptureSource
    timeout: float
    counts: dict[str, Counts]
    failures: dict[str, dict[str, list[FailureExample]]]


@dataclass
class SessionContext:
    """選択した一セッションの所在を保持する。"""

    date: str
    session_id: str
    database: Path


def _format_percentage(successes: int, denominator: int) -> str:
    """分母ゼロを算出不能として表示する。"""
    if denominator == 0:
        return "算出不能"
    return f"{successes / denominator:.1%}"


def _format_average(total: int, denominator: int) -> str:
    """器具単位あたりの平均個数を小数一桁で表示する。"""
    if denominator == 0:
        return "算出不能"
    return f"{total / denominator:.1f}"


def _read_yaml(path: Path) -> dict[str, Any]:
    """YAMLのルートがmappingであることを確認して読み込む。"""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise CaptureError(f"{path.name} のルートは設定項目のmappingである必要があります")
    return loaded


def _load_settings() -> tuple[dict[str, Any], dict[str, Any], Path]:
    """解析設定と撮影設定から入力ルート・対象条件を読み込む。"""
    settings = _read_yaml(ANALYZE_CONFIG)
    capture_settings = json.loads(CAPTURE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(capture_settings, dict) or not capture_settings.get("capture_root"):
        raise CaptureError("capture.json に capture_root の設定が必要です")
    capture_root = (CAPTURE_CONFIG.parent / str(capture_settings["capture_root"])).resolve()
    targets = _validate_targets(settings.get("targets"))
    settings["targets"] = targets
    return settings, capture_settings, capture_root


def _validate_targets(value: Any) -> list[dict[str, str]]:
    """空指定・session ID単独・不正日付を解析前に拒否する。"""
    if not isinstance(value, list) or not value:
        raise CaptureError("analyze.yaml の targets に日付条件を1件以上指定してください")
    targets: list[dict[str, str]] = []
    for index, target in enumerate(value):
        if not isinstance(target, dict):
            raise CaptureError(f"analyze.yaml targets[{index}] はmappingである必要があります")
        date = target.get("date")
        session_id = target.get("session_id")
        if not isinstance(date, str) or DATE_PATTERN.fullmatch(date) is None:
            raise CaptureError(f"analyze.yaml targets[{index}].date はYYYY-MM-DD形式で指定してください")
        try:
            calendar_date.fromisoformat(date)
        except ValueError as error:
            raise CaptureError(f"analyze.yaml targets[{index}].date は実在する日付にしてください") from error
        selected: dict[str, str] = {"date": date}
        if session_id is not None:
            safe_session = isinstance(session_id, str) and SESSION_ID_PATTERN.fullmatch(session_id) is not None
            path_safe = safe_session and session_id not in {".", ".."}
            if not path_safe:
                raise CaptureError(f"analyze.yaml targets[{index}].session_id は安全なセッションIDにしてください")
            selected["session_id"] = session_id
        targets.append(selected)
    return targets


def _select_databases(capture_root: Path, targets: list[dict[str, str]]) -> list[tuple[str, str, Path]]:
    """複数条件の和集合から重複しない日付・セッションDBを選ぶ。"""
    root = capture_root.resolve()
    selected: dict[Path, tuple[str, str, Path]] = {}
    for target in targets:
        date = target["date"]
        date_directory = (root / date).resolve()
        if not date_directory.is_relative_to(root) or not date_directory.is_dir():
            continue
        requested_session = target.get("session_id")
        sessions = [date_directory / requested_session] if requested_session else sorted(date_directory.iterdir())
        for session_directory in sessions:
            database = (session_directory / METADATA_NAME).resolve()
            inside_root = database.is_relative_to(root)
            session_exists = session_directory.is_dir()
            has_database = database.is_file()
            relative_parts = database.relative_to(root).parts if inside_root else ()
            is_expected_session = len(relative_parts) >= MIN_SESSION_DATABASE_PARTS and relative_parts[:2] == (date, session_directory.name)
            if inside_root and session_exists and has_database and is_expected_session:
                selected[database] = (date, session_directory.name, database)
    if not selected:
        raise CaptureError("指定した日付・セッションに解析可能なmetadata.sqlite3がありません")
    return [selected[key] for key in sorted(selected)]


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    """任意のレビュー・推論テーブルの存在を確認する。"""
    row = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _load_instrument_names(database_path: Path) -> dict[str, str]:
    """master DBからUIDと日本語器具名の対応を読み取り専用で取得する。"""
    if not database_path.is_file():
        raise CaptureError(f"器具名DBが見つかりません: {database_path}")
    try:
        with closing(sqlite3.connect(database_path.resolve().as_uri() + READ_ONLY_QUERY, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            if not _table_exists(connection, "samples"):
                raise CaptureError("器具名DBに samples テーブルがありません")
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(samples)").fetchall()
            }
            if not {"id", "name_ja"}.issubset(columns):
                raise CaptureError("器具名DBの samples に id と name_ja が必要です")
            rows = connection.execute("SELECT id, name_ja FROM samples").fetchall()
    except sqlite3.Error as error:
        raise CaptureError(f"器具名DBを読み取れません: {database_path}") from error
    names: dict[str, str] = {}
    for row in rows:
        uid = str(row["id"])
        name = row["name_ja"]
        if isinstance(name, str) and name.strip():
            names[uid] = name.strip()
    return names


def _current_revision(connection: sqlite3.Connection, session_id: str, image_id: str) -> Any:
    """画像に紐づく現行レビュー版を取得する。"""
    if not _table_exists(connection, "sam3_review_revisions"):
        return None
    return connection.execute(
        """SELECT revision, prediction_id, status, approval_id, approval_json,
        progress_json, annotation_json FROM sam3_review_revisions
        WHERE session_id=? AND image_id=? AND is_current=1""",
        (session_id, image_id),
    ).fetchone()


def _review_annotation(connection: sqlite3.Connection, revision: Any) -> dict[str, Any]:
    """現行レビュー版の注釈を読み、承認イベントがあれば最新内容を使う。"""
    annotation = _parse_annotation_json(revision["annotation_json"])
    is_approved_revision = revision["status"] in {"approved", "exported"}
    if is_approved_revision and _table_exists(connection, "sam3_review_events"):
        event = connection.execute(
            """SELECT annotation_json FROM sam3_review_events
            WHERE revision=? AND status IN ('approved', 'exported')
            ORDER BY event_id DESC LIMIT 1""",
            (revision["revision"],),
        ).fetchone()
        if event is not None:
            annotation = _parse_annotation_json(event["annotation_json"])
    return annotation


def _parse_annotation_json(raw: str) -> dict[str, Any]:
    """注釈JSONがobjectであることを検証する。"""
    annotation = json.loads(raw)
    if not isinstance(annotation, dict):
        raise CaptureError("YOLO annotation_json のルートはobjectである必要があります")
    return annotation


def _review_objects(
    connection: sqlite3.Connection,
    session_id: str,
    image_id: str,
    revision: Any,
) -> list[dict[str, Any]]:
    """現行推論のbbox・mask参照と物体別DINO状態を取得する。"""
    tables_exist = _table_exists(connection, "sam3_prediction_objects") and _table_exists(connection, "sam3_object_reviews")
    if not tables_exist:
        return []
    rows = connection.execute(
        """SELECT objects.object_index, objects.bbox_json, objects.mask_path,
        reviews.instrument_uid, reviews.instance_no, reviews.dino_status
        FROM sam3_prediction_objects AS objects
        LEFT JOIN sam3_object_reviews AS reviews
            ON reviews.object_index=objects.object_index AND reviews.revision=?
        WHERE objects.session_id=? AND objects.image_id=? AND objects.prediction_id=?
        ORDER BY objects.object_index""",
        (revision["revision"], session_id, image_id, revision["prediction_id"]),
    )
    return [dict(row) for row in rows]


def _image_review(connection: sqlite3.Connection, session_id: str, image_id: str) -> dict[str, Any] | None:
    """画像の現行レビュー・注釈・推論候補を読み取り専用で取得する。"""
    revision = _current_revision(connection, session_id, image_id)
    if revision is None:
        return None
    progress = json.loads(revision["progress_json"] or "{}")
    approval = json.loads(revision["approval_json"] or "{}")
    confirmed = progress.get("reviewed") is True
    is_approved_yolo = (
        revision["status"] in {"approved", "exported"}
        and revision["approval_id"] is not None
        and approval.get("scope") != "dino"
    )
    return {
        "confirmed": confirmed or is_approved_yolo,
        "annotation": _review_annotation(connection, revision),
        "revision": revision["revision"],
        "objects": _review_objects(connection, session_id, image_id, revision),
    }


def _latest_event_mask_paths(connection: sqlite3.Connection, review: dict[str, Any]) -> dict[int, str]:
    """現行レビュー版で承認されたmaskファイル参照を取得する。"""
    if not _table_exists(connection, "sam3_approved_masks"):
        return {}
    rows = connection.execute(
        "SELECT object_index, mask_path FROM sam3_approved_masks WHERE revision=?",
        (review["revision"],),
    ).fetchall()
    return {int(row["object_index"]): str(row["mask_path"]) for row in rows}


def _shape_uid(shape: dict[str, Any], original_uid: str | None = None) -> str:
    """YOLO shapeまたは初期物体から集計用器具UIDを決める。"""
    assignment = shape.get("tag_assignment")
    assigned_uid = assignment.get("instrument_uid") if isinstance(assignment, dict) else None
    label = shape.get("label")
    if isinstance(assigned_uid, str) and assigned_uid:
        return assigned_uid
    if isinstance(label, str) and label and label != "__unresolved__":
        return label
    return original_uid or UNCLASSIFIED


def _shape_box(shape: dict[str, Any]) -> list[float] | None:
    """Labelme矩形の二点表現をxyxy座標へそろえる。"""
    points = shape.get("points")
    if not isinstance(points, list) or len(points) != RECTANGLE_POINT_COUNT:
        return None
    if not all(isinstance(point, list) and len(point) == RECTANGLE_POINT_COUNT for point in points):
        return None
    try:
        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _bbox_values(raw: str) -> list[float]:
    """JSONの初期bboxを数値のxyxy座標へ変換する。"""
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != BBOX_COORDINATE_COUNT:
        raise CaptureError("初期推論のbbox_jsonは4座標の配列である必要があります")
    return [float(value) for value in values]


def _annotation_shapes(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """承認注釈に含まれるshape配列を検証する。"""
    shapes = annotation.get("shapes")
    if not isinstance(shapes, list):
        raise CaptureError("YOLO annotation_json の shapes が配列ではありません")
    if not all(isinstance(shape, dict) for shape in shapes):
        raise CaptureError("YOLO annotation_json の shapes に不正な要素があります")
    return shapes


def _unaccepted_new_shapes(annotation: dict[str, Any], original_indices: set[int]) -> list[dict[str, Any]]:
    """未採用候補のうち初期物体に対応しない追加bboxを取り出す。"""
    candidates = annotation.get("review_candidates", [])
    if not isinstance(candidates, list):
        raise CaptureError("YOLO annotation_json の review_candidates が配列ではありません")
    if not all(isinstance(shape, dict) for shape in candidates):
        raise CaptureError("YOLO annotation_json の review_candidates に不正な要素があります")
    additions = []
    for shape in candidates:
        group_id = shape.get("group_id")
        is_existing_object = isinstance(group_id, int) and not isinstance(group_id, bool) and group_id in original_indices
        if not is_existing_object:
            additions.append(shape)
    return additions


def _match_bbox_shapes(
    context: ImageContext,
) -> BBoxState:
    """初期物体indexに対応する注釈shapeと追加shapeを分ける。"""
    annotation = context.review.get("annotation", {"shapes": [], "review_candidates": []})
    shapes = _annotation_shapes(annotation)
    original_by_index = {int(item["object_index"]): item for item in context.review["objects"]}
    review_candidates = _unaccepted_new_shapes(annotation, set(original_by_index))
    additions: list[dict[str, Any]] = []
    shape_by_index: dict[int, dict[str, Any]] = {}
    for shape in shapes:
        group_id = shape.get("group_id")
        is_index = isinstance(group_id, int) and not isinstance(group_id, bool)
        if is_index and group_id in original_by_index:
            shape_by_index[group_id] = shape
        else:
            additions.append(shape)
    return BBoxState(original_by_index, shape_by_index, additions, review_candidates)


def _count_bbox_for_image(context: ImageContext) -> None:
    """撮影器具単位でbboxとmaskの成否、未採用bbox数を加算する。"""
    _register_captured_instruments(context)
    state = _match_bbox_shapes(context)
    review_state = context.review.get("confirmed", True)
    if not isinstance(review_state, bool):
        raise CaptureError(f"YOLO確認状態が不正です: {context.image_id}")
    review_is_confirmed = review_state
    _collect_inferred_boxes(context, state, review_is_confirmed)
    if review_is_confirmed:
        _add_manual_boxes_to_overlay(context, state)
    _count_bbox_units(context, state)
    _refresh_bbox_failure_examples(context, state)
    _count_dino_objects(context, state)


def _collect_inferred_boxes(
    context: ImageContext,
    state: BBoxState,
    review_is_confirmed: bool,
) -> None:
    """推論bboxの採用状態を判定し、before/after表示用の座標を集める。"""
    annotation_shapes_by_index = state.annotations if review_is_confirmed else {}
    for index, original in state.originals.items():
        shape = annotation_shapes_by_index.get(index)
        uid = _resolve_object_uid(context, original, shape)
        instance_no = _resolve_instance_no(context, uid, original, shape)
        original_box = _bbox_values(original["bbox_json"])
        state.initial_boxes_by_uid[uid].append(original_box)
        shape_box = _shape_box(shape) if shape is not None else None
        is_accepted = (
            review_is_confirmed
            and shape_box == original_box
            and instance_no is not None
            and _shape_unit(context, uid, shape, instance_no) == (uid, instance_no)
        )
        if is_accepted and instance_no is not None:
            state.accepted_units[uid].add(instance_no)
        else:
            state.unaccepted_by_uid[uid] += 1
        if shape_box is not None:
            state.boxes_by_uid[uid].append(shape_box)


def _add_manual_boxes_to_overlay(context: ImageContext, state: BBoxState) -> None:
    """承認済みの手描きbboxを失敗例の修正後表示へ加える。"""
    for shape in state.additions:
        uid = _resolve_shape_uid(context, shape)
        shape_box = _shape_box(shape)
        if shape_box is not None:
            state.boxes_by_uid[uid].append(shape_box)


def _count_bbox_units(context: ImageContext, state: BBoxState) -> None:
    """各撮影器具に正しい未修正推論bboxがあるかを集計する。"""
    for uid, expected_instances in context.expected.items():
        success_count = len(state.accepted_units.get(uid, set()) & expected_instances)
        failure_count = len(expected_instances) - success_count
        instrument_counts = context.counts.setdefault(uid, Counts())
        overall_counts = context.counts.setdefault("全体", Counts())
        instrument_counts.bbox_success += success_count
        instrument_counts.bbox_failure += failure_count
        instrument_counts.unaccepted_bbox_count += state.unaccepted_by_uid.get(uid, 0)
        overall_counts.bbox_success += success_count
        overall_counts.bbox_failure += failure_count
        overall_counts.unaccepted_bbox_count += state.unaccepted_by_uid.get(uid, 0)
        if failure_count:
            example = FailureExample(
                context.date,
                context.session_id,
                context.image_id,
                context.image_path,
                boxes=list(state.boxes_by_uid.get(uid, [])),
                initial_boxes=list(state.initial_boxes_by_uid.get(uid, [])),
                no_detection=not state.initial_boxes_by_uid.get(uid) and not state.boxes_by_uid.get(uid),
            )
            _add_failure(context.failures, "bbox", uid, example)


def _register_captured_instruments(context: ImageContext) -> None:
    """可視タグをbbox・mask共通の撮影器具分母へ一度だけ加える。"""
    if context.captured_units_registered:
        return
    if not context.expected:
        raise CaptureError(f"撮影画像に可視器具タグがありません: {context.image_id}")
    for uid, instances in context.expected.items():
        if not uid or not instances:
            raise CaptureError(f"撮影器具のUIDまたは個体番号が不正です: {context.image_id}")
    total = sum(len(instances) for instances in context.expected.values())
    context.counts.setdefault("全体", Counts()).captured_instruments += total
    for uid, instances in context.expected.items():
        context.counts.setdefault(uid, Counts()).captured_instruments += len(instances)
    context.captured_units_registered = True


def _resolve_object_uid(
    context: ImageContext,
    item: dict[str, Any],
    shape: dict[str, Any] | None,
) -> str:
    """物体記録・対応shapeから画像内の既知器具UIDを特定する。"""
    uid = item.get("instrument_uid")
    if isinstance(uid, str) and uid in context.expected:
        return uid
    if isinstance(uid, str) and uid and uid not in {UNCLASSIFIED, "__unresolved__"}:
        raise CaptureError(f"物体UIDが画像の撮影器具タグと一致しません: {context.image_id}")
    return _resolve_shape_uid(context, shape)


def _resolve_shape_uid(context: ImageContext, shape: dict[str, Any] | None) -> str:
    """shapeのタグ割当またはラベルから撮影器具UIDを特定する。"""
    if shape is not None:
        uid = _shape_uid(shape)
        if uid in context.expected:
            return uid
        if uid not in {UNCLASSIFIED, "__unresolved__"}:
            raise CaptureError(f"bboxの器具UIDが撮影器具タグと一致しません: {context.image_id}")
    if len(context.expected) == 1:
        return next(iter(context.expected))
    raise CaptureError(f"bboxを撮影器具UIDへ対応付けられません: {context.image_id}")


def _resolve_instance_no(
    context: ImageContext,
    uid: str,
    item: dict[str, Any],
    shape: dict[str, Any] | None,
) -> int | None:
    """物体またはshapeから可視タグの個体番号を解決する。"""
    candidates = context.expected[uid]
    original_instance = item.get("instance_no")
    if isinstance(original_instance, int) and not isinstance(original_instance, bool):
        return original_instance if original_instance in candidates else None
    assignment = shape.get("tag_assignment") if shape is not None else None
    assigned_instance = assignment.get("instance_no") if isinstance(assignment, dict) else None
    if isinstance(assigned_instance, int) and not isinstance(assigned_instance, bool):
        return assigned_instance if assigned_instance in candidates else None
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _shape_unit(
    context: ImageContext,
    uid: str,
    shape: dict[str, Any] | None,
    fallback_instance: int,
) -> tuple[str, int] | None:
    """shapeのタグ割当が推論物体と同じ撮影器具を指すか確認する。"""
    if shape is None:
        return None
    assignment = shape.get("tag_assignment")
    if isinstance(assignment, dict):
        assigned_uid = assignment.get("instrument_uid")
        assigned_instance = assignment.get("instance_no")
        if (
            assigned_uid == uid
            and isinstance(assigned_instance, int)
            and not isinstance(assigned_instance, bool)
            and assigned_instance in context.expected[uid]
        ):
            return uid, assigned_instance
        return None
    if shape.get("label") == uid and fallback_instance in context.expected[uid]:
        return uid, fallback_instance
    return None


def _refresh_bbox_failure_examples(context: ImageContext, state: BBoxState) -> None:
    """画像例に器具別の確定bboxを反映し、無検出表示を正しく決める。"""
    for uid, examples in context.failures["bbox"].items():
        for example in examples:
            is_current_image = (example.date, example.session_id, example.image_id) == (
                context.date,
                context.session_id,
                context.image_id,
            )
            if is_current_image:
                example.initial_boxes = list(state.initial_boxes_by_uid.get(uid, []))
                example.boxes = list(state.boxes_by_uid.get(uid, []))
                example.no_detection = not example.initial_boxes and not example.boxes


def _count_dino_objects(context: ImageContext, bbox_state: BBoxState | None = None) -> None:
    """撮影器具ごとに採用maskの有無を成功・失敗として集計する。"""
    accepted_units: dict[str, set[int]] = defaultdict(set)
    pending_units: dict[str, set[int]] = defaultdict(set)
    pending_unassigned_by_uid: dict[str, int] = defaultdict(int)
    failure_masks_by_uid: dict[str, list[str]] = defaultdict(list)
    objects = context.review.get("objects", [])
    for item in objects:
        object_index = int(item["object_index"])
        shape = bbox_state.annotations.get(object_index) if bbox_state is not None else None
        uid = _resolve_object_uid(context, item, shape)
        status = item.get("dino_status")
        if status is None:
            status = "pending"
        if status not in {"accepted", "rejected", "pending"}:
            raise CaptureError(f"DINO採否状態が不正です: {context.image_id}")
        instance_no = _resolve_instance_no(context, uid, item, shape)
        if status == "accepted":
            if instance_no is None:
                raise CaptureError(f"採用maskの個体番号を特定できません: {context.image_id}")
            accepted_units[uid].add(instance_no)
        elif status == "pending":
            if instance_no is None:
                pending_unassigned_by_uid[uid] += 1
            else:
                pending_units[uid].add(instance_no)
            failure_masks_by_uid[uid].append(
                context.mask_paths.get(object_index, str(item.get("mask_path", "")))
            )
        else:
            failure_masks_by_uid[uid].append(
                context.mask_paths.get(object_index, str(item.get("mask_path", "")))
            )

    for uid, expected_instances in context.expected.items():
        successful_units = accepted_units.get(uid, set()) & expected_instances
        failure_count = len(expected_instances) - len(successful_units)
        pending_count = len(pending_units.get(uid, set()) & expected_instances)
        pending_count += pending_unassigned_by_uid.get(uid, 0)
        pending_count = min(pending_count, failure_count)
        instrument_counts = context.counts.setdefault(uid, Counts())
        overall_counts = context.counts.setdefault("全体", Counts())
        instrument_counts.mask_success += len(successful_units)
        instrument_counts.mask_failure += failure_count
        instrument_counts.mask_pending += pending_count
        overall_counts.mask_success += len(successful_units)
        overall_counts.mask_failure += failure_count
        overall_counts.mask_pending += pending_count
        if failure_count:
            _add_failure(
                context.failures,
                "mask",
                uid,
                FailureExample(
                    context.date,
                    context.session_id,
                    context.image_id,
                    context.image_path,
                    mask_paths=list(dict.fromkeys(failure_masks_by_uid.get(uid, []))),
                ),
            )


def _add_failure(
    failures: dict[str, dict[str, list[FailureExample]]],
    metric: str,
    uid: str,
    example: FailureExample,
) -> None:
    """同じ器具・評価区分・画像の失敗を1例にまとめる。"""
    examples = failures.setdefault(metric, {}).setdefault(uid, [])
    for existing in examples:
        if (existing.date, existing.session_id, existing.image_id) == (example.date, example.session_id, example.image_id):
            existing.boxes.extend(box for box in example.boxes if box not in existing.boxes)
            existing.initial_boxes.extend(box for box in example.initial_boxes if box not in existing.initial_boxes)
            existing.mask_paths.extend(path for path in example.mask_paths if path not in existing.mask_paths)
            existing.no_detection = existing.no_detection or example.no_detection
            return
    examples.append(example)


def _evaluate(capture_root: Path, targets: list[dict[str, str]], timeout: float) -> tuple[dict[str, Counts], dict[str, dict[str, list[FailureExample]]]]:
    """選択済みセッションの画像・レビューを読み取り、全件を集計する。"""
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    analysis = AnalysisContext(SqliteCaptureSource(capture_root, timeout), timeout, counts, failures)
    for date, session_id, database in _select_databases(capture_root, targets):
        session = SessionContext(date, session_id, database)
        _evaluate_session(session, analysis)
    if counts["全体"].captured_instruments == 0:
        raise CaptureError("解析対象に可視タグ付きの撮影器具がありません")
    return counts, failures


def _evaluate_session(session: SessionContext, analysis: AnalysisContext) -> None:
    """一つのセッションDBを読み取り専用で開き、対象画像を評価する。"""
    try:
        with closing(sqlite3.connect(session.database.as_uri() + READ_ONLY_QUERY, uri=True, timeout=analysis.timeout)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            images = connection.execute(
                "SELECT session_id, image_id, path FROM images WHERE tag_status='tagged' ORDER BY image_id"
            ).fetchall()
            for image in images:
                _evaluate_image(connection, session, analysis, image)
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError) as error:
        raise CaptureError(f"解析データを読み込めません: {session.database}: {error}") from error


def _evaluate_image(
    connection: sqlite3.Connection,
    session: SessionContext,
    analysis: AnalysisContext,
    image: sqlite3.Row,
) -> None:
    """撮影画像を現行SQLと照合し、bboxまたはDINO採否を集計する。"""
    relative_image = PurePosixPath(image["path"])
    image_belongs_to_session = (
        image["session_id"] == session.session_id
        and len(relative_image.parts) >= SESSION_PATH_PARTS
        and relative_image.parts[:2] == (session.date, session.session_id)
    )
    if not image_belongs_to_session:
        raise CaptureError(f"撮影画像が選択した日付・セッションと一致しません: {image['image_id']}")
    sample = analysis.source.current_sample(image["session_id"], image["image_id"], image["path"])
    if sample is None:
        return
    expected: dict[str, set[int]] = defaultdict(set)
    for tag in sample.tags:
        if tag.visible:
            if not tag.instrument_uid or not isinstance(tag.instance_no, int) or isinstance(tag.instance_no, bool):
                raise CaptureError(f"撮影器具のUIDまたは個体番号が不正です: {image['image_id']}")
            if tag.instance_no in expected[tag.instrument_uid]:
                raise CaptureError(f"同一画像で撮影器具タグが重複しています: {image['image_id']}")
            expected[tag.instrument_uid].add(tag.instance_no)
    review = _image_review(connection, session.session_id, image["image_id"])
    mask_paths = _latest_event_mask_paths(connection, review) if review is not None else {}
    context = ImageContext(
        session.date,
        session.session_id,
        image["image_id"],
        sample.image_path,
        expected,
        review or {},
        analysis.counts,
        analysis.failures,
        mask_paths,
    )
    if review is None or not review["confirmed"]:
        context.review = review or {
            "confirmed": False,
            "annotation": {"shapes": [], "review_candidates": []},
            "objects": [],
        }
        _record_unconfirmed_image(context)
    _count_bbox_for_image(context)


def _record_unconfirmed_image(
    context: ImageContext,
) -> None:
    """YOLO未確認の撮影器具数を別途数える。"""
    total = sum(len(instances) for instances in context.expected.values())
    context.counts.setdefault("全体", Counts()).yolo_unconfirmed += total
    for uid, instances in context.expected.items():
        context.counts.setdefault(uid, Counts()).yolo_unconfirmed += len(instances)


def _count_mask_only(context: ImageContext) -> None:
    """既存呼び出し向けに撮影器具単位でmask状態を集計する。"""
    _register_captured_instruments(context)
    _count_dino_objects(context)


def _safe_output_root(configured_output_root: Any, project_root: Path) -> Path:
    """設定された解析出力先をプロジェクトルート配下に解決する。"""
    if not isinstance(configured_output_root, str) or not configured_output_root.strip():
        raise CaptureError("analyze.yaml の output_root に出力先を指定してください")

    root = project_root.resolve()
    configured_path = Path(configured_output_root)
    if configured_path.is_absolute():
        unresolved_destination = configured_path
    else:
        unresolved_destination = root / configured_path
    destination = unresolved_destination.resolve()
    if destination == root or not destination.is_relative_to(root):
        raise CaptureError("解析出力先はプロジェクトルート配下に指定してください")
    return destination


def _create_run_directory(output_root: Path) -> Path:
    """新しい連番runディレクトリを競合・既存上書きなしで作成する。"""
    if output_root.exists() and output_root.is_symlink():
        raise CaptureError("解析出力先にシンボリックリンクは使用できません")
    output_root.mkdir(parents=True, exist_ok=True)
    run_numbers = []
    for path in output_root.iterdir():
        match = RUN_PATTERN.fullmatch(path.name)
        if match and path.is_dir():
            run_numbers.append(int(match.group(1)))
    number = max(run_numbers, default=0) + 1
    run_directory = output_root / f"run_{number:04d}"
    if run_directory.resolve().parent != output_root.resolve():
        raise CaptureError("解析run出力先が安全な配下ではありません")
    run_directory.mkdir(exist_ok=False)
    return run_directory


def _write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    """日本語列名を維持してCSVを保存する。"""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Windows標準の日本語フォントを優先し、利用不能時はPillow既定へ戻す。"""
    font_paths = (
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/YuGothR.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
    )
    for font_path in font_paths:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    """描画に使うフォントで文字列の幅を測定する。"""
    bounds = draw.textbbox((0, 0), text, font=font)
    return int(bounds[2] - bounds[0])


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """文字列を省略せず、指定幅に収まる描画行へ分ける。"""
    lines: list[str] = []
    current_line = ""
    for character in text:
        if character == "\n":
            lines.append(current_line)
            current_line = ""
            continue
        candidate = current_line + character
        line_is_too_wide = bool(current_line) and _text_width(draw, candidate, font) > max_width
        if line_is_too_wide:
            lines.append(current_line)
            current_line = character
        else:
            current_line = candidate
    if current_line or not lines:
        lines.append(current_line)
    return lines


def _text_line_height(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    """複数行のセルに必要なフォント高を測る。"""
    bounds = draw.textbbox((0, 0), "国Ag", font=font)
    return int(bounds[3] - bounds[1])


def _table_row_height(
    draw: ImageDraw.ImageDraw,
    values: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    """セル内の折返し行数に応じて表の行高を決める。"""
    line_height = _text_line_height(draw, font)
    max_lines = 1
    for column_number, text in enumerate(values):
        available_width = TABLE_COLUMN_WIDTHS[column_number] - TABLE_CELL_PADDING * 2
        wrapped_lines = _wrap_text(draw, text, font, available_width)
        max_lines = max(max_lines, len(wrapped_lines))
    vertical_padding = TABLE_CELL_PADDING * 2
    return max(44, max_lines * (line_height + TABLE_LINE_SPACING) + vertical_padding)


def _write_table_pages(path_prefix: Path, rows: list[dict[str, str | int]]) -> None:
    """表を1ページ24行までに分割し、折返し済みセルをPNGへ保存する。"""
    if not rows:
        rows = [Counts().row("該当なし")]
    pages = [rows[start : start + TABLE_ROWS_PER_PAGE] for start in range(0, len(rows), TABLE_ROWS_PER_PAGE)]
    for page_number, page_rows in enumerate(pages, start=1):
        width = sum(TABLE_COLUMN_WIDTHS)
        title_font = _font(28)
        cell_font = _font(17)
        title = "全体サマリ表" if path_prefix.stem == "summary_table" else "器具種別サマリ表"
        header_row: dict[str, str | int] = {column: column for column in TABLE_COLUMNS}
        table_rows = [header_row, *page_rows]
        measure_image = Image.new("RGB", (width, 1), "white")
        measure_draw = ImageDraw.Draw(measure_image)
        row_heights = [
            _table_row_height(measure_draw, [str(row[column]) for column in TABLE_COLUMNS], cell_font)
            for row in table_rows
        ]
        height = 72 + sum(row_heights)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), title, fill="#17324d", font=title_font)
        top = 60
        for row_number, values in enumerate(table_rows):
            row_values = [str(values[column]) for column in TABLE_COLUMNS]
            row_height = row_heights[row_number]
            fill = "#dce8f1" if row_number == 0 else ("#f5f8fa" if row_number % 2 else "white")
            draw.rectangle((0, top, width, top + row_height), fill=fill, outline="#aab7c4")
            left = 0
            for column_number, text in enumerate(row_values):
                column_width = TABLE_COLUMN_WIDTHS[column_number]
                available_width = column_width - TABLE_CELL_PADDING * 2
                lines = _wrap_text(draw, text, cell_font, available_width)
                line_height = _text_line_height(draw, cell_font) + TABLE_LINE_SPACING
                text_height = len(lines) * line_height - TABLE_LINE_SPACING
                text_top = top + max(TABLE_CELL_PADDING, (row_height - text_height) // 2)
                for line_number, line in enumerate(lines):
                    draw.text(
                        (left + TABLE_CELL_PADDING, text_top + line_number * line_height),
                        line,
                        fill="#182532",
                        font=cell_font,
                    )
                left += column_width
            top += row_height
        image.save(path_prefix.with_name(f"{path_prefix.stem}_{page_number:02d}.png"))


def _write_chart(path: Path, rows: list[dict[str, str | int]], title: str) -> None:
    """bbox・mask成功率を日本語ラベルの横棒グラフで保存する。"""
    chart_rows = rows or [Counts().row("該当なし")]
    width = 24 + CHART_LABEL_WIDTH + 20 + CHART_METRIC_WIDTH + CHART_WIDTH + CHART_RATE_WIDTH + 24
    label_font = _font(18)
    measure_image = Image.new("RGB", (width, 1), "white")
    measure_draw = ImageDraw.Draw(measure_image)
    line_height = _text_line_height(measure_draw, label_font) + 4
    wrapped_labels = [
        _wrap_text(measure_draw, str(row["器具種別"]), label_font, CHART_LABEL_WIDTH)
        for row in chart_rows
    ]
    row_heights = [max(len(lines) * line_height + 12, line_height * 2 + 12) for lines in wrapped_labels]
    height = 92 + sum(row_heights)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(28)
    draw.text((24, 18), title, fill="#17324d", font=title_font)
    label_left = 24
    metric_left = label_left + CHART_LABEL_WIDTH + 20
    chart_left = metric_left + CHART_METRIC_WIDTH
    rate_left = chart_left + CHART_WIDTH
    top = 72
    for row_number, row in enumerate(chart_rows):
        row_height = row_heights[row_number]
        uid_lines = wrapped_labels[row_number]
        uid_text_height = len(uid_lines) * line_height - 4
        uid_top = top + (row_height - uid_text_height) // 2
        for line_number, line in enumerate(uid_lines):
            draw.text((label_left, uid_top + line_number * line_height), line, fill="#182532", font=label_font)
        for metric_number, rate_key, color, label in (
            (0, "bbox成功率", "#4d88b8", "bbox成功率"),
            (1, "mask成功率", "#43a078", "mask成功率"),
        ):
            metric_top = top + 6 + metric_number * line_height
            draw.text((metric_left, metric_top), label, fill="#182532", font=label_font)
            rate_text = str(row[rate_key])
            if rate_text == "算出不能":
                draw.text((rate_left + 12, metric_top), rate_text, fill="#555555", font=label_font)
                continue
            rate = float(rate_text.rstrip("%")) / 100
            bar_width = int(CHART_WIDTH * rate)
            draw.rectangle((chart_left, metric_top + 3, chart_left + bar_width, metric_top + 19), fill=color)
            draw.text((rate_left + 12, metric_top), rate_text, fill="#182532", font=label_font)
        top += row_height
    image.save(path)


def _write_average_chart(path: Path, rows: list[dict[str, str | int]], title: str) -> None:
    """器具あたり平均未採用bbox数を横棒グラフで保存する。"""
    chart_rows = rows or [Counts().row("該当なし")]
    label_font = _font(18)
    measure_image = Image.new("RGB", (1, 1), "white")
    measure_draw = ImageDraw.Draw(measure_image)
    line_height = _text_line_height(measure_draw, label_font) + 4
    wrapped_labels = [
        _wrap_text(measure_draw, str(row["器具種別"]), label_font, CHART_LABEL_WIDTH)
        for row in chart_rows
    ]
    row_heights = [max(len(lines) * line_height + 12, line_height + 12) for lines in wrapped_labels]
    numeric_averages = [
        float(row["器具あたり平均未採用bbox数"])
        for row in chart_rows
        if row["器具あたり平均未採用bbox数"] != "算出不能"
    ]
    value_labels = [
        "算出不能"
        if row["器具あたり平均未採用bbox数"] == "算出不能"
        else f"{row['器具あたり平均未採用bbox数']}個"
        for row in chart_rows
    ]
    value_width = max(CHART_RATE_WIDTH, max(_text_width(measure_draw, value, label_font) for value in value_labels) + 24)
    width = 24 + CHART_LABEL_WIDTH + 20 + CHART_WIDTH + value_width + 24
    maximum_average = max(numeric_averages, default=0.0)
    height = 92 + sum(row_heights)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill="#17324d", font=_font(28))
    label_left = 24
    chart_left = label_left + CHART_LABEL_WIDTH + 20
    value_left = chart_left + CHART_WIDTH
    top = 72
    for row_number, row in enumerate(chart_rows):
        row_height = row_heights[row_number]
        label_lines = wrapped_labels[row_number]
        label_top = top + (row_height - len(label_lines) * line_height) // 2
        for line_number, line in enumerate(label_lines):
            draw.text((label_left, label_top + line_number * line_height), line, fill="#182532", font=label_font)
        average_text = str(row["器具あたり平均未採用bbox数"])
        metric_top = top + (row_height - line_height) // 2
        if average_text == "算出不能":
            draw.text((value_left + 12, metric_top), average_text, fill="#555555", font=label_font)
        else:
            average = float(average_text)
            bar_width = int(CHART_WIDTH * average / maximum_average) if maximum_average else 0
            draw.rectangle((chart_left, metric_top + 3, chart_left + bar_width, metric_top + 19), fill="#d66a4a")
            draw.text((value_left + 12, metric_top), f"{average_text}個", fill="#182532", font=label_font)
        top += row_height
    image.save(path)


def _safe_uid(uid: str) -> str:
    """器具UIDを出力ディレクトリ名に使える形式へ限定する。"""
    encoded = quote(uid, safe="._-")
    if encoded in {"", ".", ".."}:
        return f"uid_{encoded or 'unknown'}"
    return encoded


def _font_text(draw: ImageDraw.ImageDraw, text: str, position: tuple[int, int], fill: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> None:
    """画像内の日本語説明を描画する。"""
    draw.text(position, text, fill=fill, font=font)


def _add_failure_caption(image: Image.Image, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    """画像上部に幅・高さを確保した説明欄を追加し、日本語文字を切らずに表示する。"""
    draw = ImageDraw.Draw(image)
    caption_width = _text_width(draw, text, font)
    caption_height = _text_line_height(draw, font) + 20
    canvas_width = max(image.width, caption_width + 24)
    canvas = Image.new("RGB", (canvas_width, image.height + caption_height), "white")
    canvas.paste(image, (0, caption_height))
    _font_text(ImageDraw.Draw(canvas), text, (12, 10), "#ff2d2d", font)
    return canvas


def _add_bbox_legend(
    image: Image.Image,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    no_detection: bool,
) -> Image.Image:
    """修正前後の色と、必要な場合は検出なしを画像上部に表示する。"""
    labels = [
        ("修正前（初期推論）", "#2864dc"),
        ("修正後（承認後）", "#ff2d2d"),
    ]
    if no_detection:
        labels.append(("検出なし", "#ff2d2d"))

    text_draw = ImageDraw.Draw(image)
    line_height = _text_line_height(text_draw, font)
    longest_label_width = max(_text_width(text_draw, label, font) for label, _ in labels)
    header_height = line_height * len(labels) + 20
    canvas_width = max(image.width, longest_label_width + 56)
    canvas = Image.new("RGB", (canvas_width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, image.width // 300)
    for row_number, (label, color) in enumerate(labels):
        row_top = 10 + row_number * line_height
        swatch_y = row_top + line_height // 2
        draw.line((12, swatch_y, 38, swatch_y), fill=color, width=line_width)
        _font_text(draw, label, (48, row_top), color, font)
    return canvas


def _resolve_mask_path(capture_root: Path, stored_path: str) -> Path | None:
    """DBのmask参照をcapture_root内に限定して実ファイルへ解決する。"""
    if not stored_path or stored_path.startswith("memory:"):
        return None
    relative = PurePosixPath(stored_path)
    candidate = (capture_root / Path(*relative.parts)).resolve()
    if relative.is_absolute() or ".." in relative.parts or "\\" in stored_path or not candidate.is_relative_to(capture_root.resolve()):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _save_failure_example(example: FailureExample, destination: Path, metric: str, capture_root: Path) -> None:
    """元画像のコピーにbboxまたは利用可能なmaskを重ね、別ファイルとして保存する。"""
    with Image.open(example.image_path) as source:
        image = source.convert("RGB")
    font = _font(max(18, image.width // 45))
    if metric == "bbox":
        draw = ImageDraw.Draw(image)
        line_width = max(2, image.width // 300)
        for box in example.initial_boxes:
            draw.rectangle(tuple(box), outline="#2864dc", width=line_width)
        for box in example.boxes:
            draw.rectangle(tuple(box), outline="#ff2d2d", width=line_width)
        image = _add_bbox_legend(image, font, example.no_detection)
    else:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        found_mask = False
        for stored_path in example.mask_paths:
            mask_path = _resolve_mask_path(capture_root, stored_path)
            if mask_path is None:
                continue
            with Image.open(mask_path) as mask_image:
                mask = mask_image.convert("L").resize(image.size)
            overlay_draw.bitmap((0, 0), mask, fill=(255, 40, 35, 115))
            found_mask = True
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        if not found_mask:
            image = _add_failure_caption(image, "mask画像なし", font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _write_failure_examples(
    run_directory: Path,
    failure_examples: dict[str, dict[str, list[FailureExample]]],
    capture_root: Path,
    limit: int,
) -> None:
    """器具・評価区分ごとに無作為抽出した失敗画像を上限まで出力する。"""
    for metric, instruments in failure_examples.items():
        for uid, examples in instruments.items():
            unique = {(item.date, item.session_id, item.image_id): item for item in examples}
            candidates = list(unique.values())
            selected = random.sample(candidates, min(limit, len(candidates)))
            for example in selected:
                filename_parts = (example.date, example.session_id, example.image_id)
                filename = "_".join(_safe_uid(part) for part in filename_parts) + ".png"
                destination = run_directory / "failure_examples" / metric / _safe_uid(uid) / filename
                _save_failure_example(example, destination, metric, capture_root)


def _instrument_display_names(instrument_uids: list[str], names: dict[str, str]) -> dict[str, str]:
    """日本語器具名を使い、同名UIDは表示時だけ識別できるようにする。"""
    labels = {uid: names.get(uid, uid) for uid in instrument_uids}
    label_counts: dict[str, int] = defaultdict(int)
    for label in labels.values():
        label_counts[label] += 1
    for uid, label in labels.items():
        if label_counts[label] > 1:
            labels[uid] = f"{label}（{uid}）"
    return labels


def _write_results(
    run_directory: Path,
    capture_root: Path,
    counts: dict[str, Counts],
    failures: dict[str, dict[str, list[FailureExample]]],
    settings: dict[str, Any],
    instrument_names: dict[str, str] | None = None,
) -> None:
    """CSV・表画像・グラフ・失敗例と実行設定をrunへ保存する。"""
    overall_rows = [counts.get("全体", Counts()).row("全体")]
    instrument_uids = sorted(uid for uid in counts if uid != "全体")
    display_names = _instrument_display_names(instrument_uids, instrument_names or {})
    instrument_rows = [counts[uid].row(uid, display_names[uid]) for uid in instrument_uids]
    _write_csv(run_directory / "summary.csv", overall_rows)
    _write_csv(run_directory / "instrument_summary.csv", instrument_rows)
    _write_table_pages(run_directory / "summary_table", overall_rows)
    _write_table_pages(run_directory / "instrument_table", instrument_rows)
    _write_chart(run_directory / "success_rate_overall.png", overall_rows, "全体の成功率")
    _write_chart(run_directory / "success_rate_by_instrument.png", instrument_rows, "器具種別ごとの成功率")
    _write_average_chart(
        run_directory / "unaccepted_bbox_average_overall.png",
        overall_rows,
        "全体の器具あたり平均未採用bbox数",
    )
    _write_average_chart(
        run_directory / "unaccepted_bbox_average_by_instrument.png",
        instrument_rows,
        "器具種別ごとの器具あたり平均未採用bbox数",
    )
    _write_failure_examples(run_directory, failures, capture_root, MAX_FAILURE_EXAMPLES)
    (run_directory / "analyze.yaml").write_text(
        yaml.safe_dump(settings, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    """設定を読み取り専用解析へ渡し、新しいrun出力を作る。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings, capture_settings, capture_root = _load_settings()
        output_root = _safe_output_root(settings.get("output_root"), PROJECT_ROOT)
        instrument_names = _load_instrument_names(MASTER_DATASET_DB)
        timeout = float(capture_settings.get("sqlite_timeout", DEFAULT_SQLITE_TIMEOUT))
        _select_databases(capture_root, settings["targets"])
        counts, failures = _evaluate(capture_root, settings["targets"], timeout)
        run_directory = _create_run_directory(output_root)
        _write_results(run_directory, capture_root, counts, failures, settings, instrument_names)
        LOGGER.info("解析結果を保存しました: %s", run_directory)
    except (CaptureError, OSError, sqlite3.Error, ValueError, TypeError, KeyError, yaml.YAMLError):
        LOGGER.exception("解析を中止しました")
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
