"""解析対象選択とbbox・mask集計のfocused tests。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from analyze import (
    CHART_LABEL_WIDTH,
    CHART_METRIC_WIDTH,
    CHART_WIDTH,
    TABLE_CELL_PADDING,
    TABLE_COLUMN_WIDTHS,
    TABLE_COLUMNS,
    Counts,
    FailureExample,
    ImageContext,
    _count_bbox_for_image,
    _count_mask_only,
    _create_run_directory,
    _evaluate,
    _font,
    _instrument_display_names,
    _load_instrument_names,
    _safe_output_root,
    _save_failure_example,
    _select_databases,
    _table_row_height,
    _text_width,
    _validate_targets,
    _wrap_text,
    _write_average_chart,
    _write_chart,
    _write_results,
    _write_table_pages,
)
from sam3_pipeline.domain.capture import CaptureError

EXPECTED_SUCCESS = 1
EXPECTED_BBOX_FAILURES = 2
EXPECTED_MASK_FAILURES = 1
EXPECTED_PENDING = 1
EXPECTED_NONE = 0
EXPECTED_SINGLE_FAILURE = 1
MINIMUM_TABLE_ROW_HEIGHT = 44
MINIMUM_MULTILINE_TABLE_HEIGHT = 150
NO_DETECTION_IMAGE_SIZE = 20
CAPTURE_DATE = "2026-11-24"
SESSION_ID = "session_a"
IMAGE_ID = "image_a"


def _create_review_database(root: Path) -> Path:
    """撮影入力・YOLO承認・DINO採否を持つ最小のセッションDBを作る。"""
    image_path = root / CAPTURE_DATE / SESSION_ID / "raw" / "photo.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), "white").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    database = image_path.parents[1] / "metadata.sqlite3"
    annotation = {
        "shapes": [
            {
                "group_id": 0,
                "label": "tool_a",
                "points": [[2, 3], [20, 25]],
                "tag_assignment": {"instrument_uid": "tool_a", "instance_no": 1},
            }
        ]
    }
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE declarations(declaration_id TEXT, items_json TEXT);
            CREATE TABLE images(session_id TEXT, image_id TEXT, declaration_id TEXT, arrangement_no INTEGER,
                path TEXT, sha256 TEXT, width INTEGER, height INTEGER, tag_revision INTEGER, tag_status TEXT);
            CREATE TABLE hints(image_id TEXT, instrument_uid TEXT, instance_no INTEGER, visible INTEGER,
                hint_source TEXT, center_json TEXT, end_a_json TEXT, end_b_json TEXT, occluded INTEGER);
            CREATE TABLE sam3_review_revisions(revision TEXT, session_id TEXT, image_id TEXT, prediction_id TEXT,
                is_current INTEGER, status TEXT, approval_id TEXT, approval_json TEXT, progress_json TEXT,
                annotation_json TEXT);
            CREATE TABLE sam3_prediction_objects(session_id TEXT, image_id TEXT, prediction_id TEXT,
                object_index INTEGER, bbox_json TEXT, mask_path TEXT);
            CREATE TABLE sam3_object_reviews(revision TEXT, object_index INTEGER, instrument_uid TEXT,
                instance_no INTEGER, dino_status TEXT);
            CREATE TABLE sam3_review_events(event_id INTEGER, revision TEXT, status TEXT, annotation_json TEXT);
            """
        )
        connection.execute(
            "INSERT INTO declarations VALUES (?, ?)",
            ("decl_a", json.dumps([{"instrument_uid": "tool_a", "instrument_name": "工具A"}])),
        )
        connection.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                SESSION_ID,
                IMAGE_ID,
                "decl_a",
                1,
                f"{CAPTURE_DATE}/{SESSION_ID}/raw/photo.png",
                digest,
                40,
                30,
                1,
                "tagged",
            ),
        )
        connection.execute(
            "INSERT INTO hints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (IMAGE_ID, "tool_a", 1, 1, "manual", json.dumps({"x": 0.5, "y": 0.5}), None, None, 0),
        )
        connection.execute(
            "INSERT INTO sam3_review_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("revision_a", SESSION_ID, IMAGE_ID, "prediction_a", 1, "approved", "approval_a", '{"scope":"yolo"}', "{}", json.dumps(annotation)),
        )
        connection.execute(
            "INSERT INTO sam3_prediction_objects VALUES (?, ?, ?, ?, ?, ?)",
            (SESSION_ID, IMAGE_ID, "prediction_a", 0, "[2,3,20,25]", "memory:0"),
        )
        connection.execute(
            "INSERT INTO sam3_object_reviews VALUES (?, ?, ?, ?, ?)",
            ("revision_a", 0, "tool_a", 1, "accepted"),
        )
        connection.execute(
            "INSERT INTO sam3_review_events VALUES (?, ?, ?, ?)",
            (1, "revision_a", "approved", json.dumps(annotation)),
        )
    return database


def _context(
    tmp_path: Path,
    review: dict[str, Any],
    counts: dict[str, Counts],
    failures: dict[str, dict[str, list[FailureExample]]],
    expected: dict[str, int] | None = None,
) -> ImageContext:
    """集計関数へテスト用の一枚分コンテキストを渡す。"""
    expected_units = {
        uid: set(range(1, count + 1))
        for uid, count in (expected or {}).items()
    }
    return ImageContext(
        "2026-11-24",
        "session_a",
        "image_a",
        tmp_path / "source.png",
        expected_units,
        review,
        counts,
        failures,
        {},
    )


def test_validate_targets_requires_dates_and_nonempty_conditions() -> None:
    """日付なし・対象空を全件解析へ拡張しない。"""
    with pytest.raises(CaptureError, match="targets"):
        _validate_targets([])
    with pytest.raises(CaptureError, match="date"):
        _validate_targets([{"session_id": "session_a"}])
    with pytest.raises(CaptureError, match="date"):
        _validate_targets([{"date": "2026-99-99"}])


def test_validate_targets_accepts_date_and_session_filters() -> None:
    """日付のみと日付＋session IDの条件を保持する。"""
    assert _validate_targets([{"date": "2026-11-24"}, {"date": "2026-11-25", "session_id": "session_a"}]) == [
        {"date": "2026-11-24"},
        {"date": "2026-11-25", "session_id": "session_a"},
    ]


def test_select_databases_unions_conditions_without_duplicates(tmp_path: Path) -> None:
    """重複条件を一つのセッションDBとして選択する。"""
    database = tmp_path / "2026-11-24" / "session_a" / "metadata.sqlite3"
    database.parent.mkdir(parents=True)
    database.touch()

    selected = _select_databases(
        tmp_path,
        [{"date": "2026-11-24"}, {"date": "2026-11-24", "session_id": "session_a"}],
    )

    assert selected == [("2026-11-24", "session_a", database.resolve())]


def test_safe_output_root_resolves_relative_path_from_project_root(tmp_path: Path) -> None:
    """相対output_rootをプロジェクトルート基準で解決する。"""
    output_root = _safe_output_root("results_analyze", tmp_path)

    assert output_root == (tmp_path / "results_analyze").resolve()


def test_safe_output_root_rejects_paths_outside_project_root(tmp_path: Path) -> None:
    """親移動やシンボリックリンクによるルート外出力を拒否する。"""
    with pytest.raises(CaptureError, match="プロジェクトルート"):
        _safe_output_root("../outside", tmp_path)

    outside_root = tmp_path.parent / f"{tmp_path.name}_outside"
    try:
        (tmp_path / "linked").symlink_to(outside_root, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("この環境ではディレクトリsymlinkを作成できません")

    with pytest.raises(CaptureError, match="プロジェクトルート"):
        _safe_output_root("linked/results_analyze", tmp_path)


def test_create_run_directory_keeps_sequential_names(tmp_path: Path) -> None:
    """runディレクトリを連番で作成する。"""
    output_root = _safe_output_root("results_analyze", tmp_path)

    first_run = _create_run_directory(output_root)
    second_run = _create_run_directory(output_root)

    assert first_run == output_root / "run_0001"
    assert second_run == output_root / "run_0002"


def test_evaluate_reads_selected_session_without_modifying_database(tmp_path: Path) -> None:
    """撮影画像と承認済みSQLを読み取り、既存DBを変更しない。"""
    database = _create_review_database(tmp_path)
    database_snapshot = database.read_bytes()

    counts, failures = _evaluate(tmp_path, [{"date": CAPTURE_DATE}], timeout=1.0)

    assert counts["全体"].bbox_success == EXPECTED_SUCCESS
    assert counts["全体"].mask_success == EXPECTED_SUCCESS
    assert counts["全体"].bbox_failure == EXPECTED_NONE
    assert failures["bbox"] == {}
    assert database.read_bytes() == database_snapshot

    output_root = _safe_output_root("results_analyze", tmp_path)
    run_directory = _create_run_directory(output_root)
    _write_results(
        run_directory,
        tmp_path,
        counts,
        failures,
        {"targets": [{"date": CAPTURE_DATE}]},
        {"tool_a": "器具A"},
    )
    assert (run_directory / "summary.csv").is_file()
    assert (run_directory / "instrument_summary.csv").is_file()
    assert (run_directory / "summary_table_01.png").is_file()
    assert (run_directory / "instrument_table_01.png").is_file()
    assert (run_directory / "success_rate_overall.png").is_file()
    assert (run_directory / "success_rate_by_instrument.png").is_file()
    assert (run_directory / "unaccepted_bbox_average_overall.png").is_file()
    assert (run_directory / "unaccepted_bbox_average_by_instrument.png").is_file()
    assert (run_directory / "analyze.yaml").is_file()
    assert "器具A" in (run_directory / "instrument_summary.csv").read_text(encoding="utf-8-sig")


def test_count_bbox_detects_changed_deleted_and_added_shapes(tmp_path: Path) -> None:
    """group_idで初期物体と注釈を対応させ、追加・削除も失敗とする。"""
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    review: dict[str, Any] = {
        "annotation": {
            "shapes": [
                {"group_id": 0, "label": "tool_a", "points": [[1, 2], [8, 9]]},
                {"group_id": None, "label": "tool_a", "points": [[20, 20], [30, 30]]},
            ],
            "review_candidates": [
                {"group_id": None, "label": "tool_a", "points": [[31, 31], [38, 38]]},
            ],
        },
        "objects": [
            {"object_index": 0, "bbox_json": json.dumps([1, 2, 8, 9]), "mask_path": "memory:0", "instrument_uid": "tool_a", "instance_no": 1, "dino_status": "accepted"},
            {"object_index": 1, "bbox_json": json.dumps([10, 10, 15, 15]), "mask_path": "memory:1", "instrument_uid": "tool_a", "instance_no": 2, "dino_status": "rejected"},
            {"object_index": 2, "bbox_json": json.dumps([40, 40, 45, 45]), "mask_path": "memory:2", "instrument_uid": "tool_a", "instance_no": 3, "dino_status": "pending"},
        ],
    }

    _count_bbox_for_image(_context(tmp_path, review, counts, failures, {"tool_a": 3}))

    assert counts["全体"].bbox_success == EXPECTED_SUCCESS
    assert counts["全体"].bbox_failure == EXPECTED_BBOX_FAILURES
    assert counts["全体"].captured_instruments == 3
    assert counts["全体"].unaccepted_bbox_count == 2
    assert counts["全体"].mask_success == EXPECTED_SUCCESS
    assert counts["全体"].mask_failure == 2
    assert counts["全体"].mask_pending == EXPECTED_PENDING
    assert len(failures["bbox"]["tool_a"]) == 1
    assert len(failures["mask"]["tool_a"]) == 1


def test_count_bbox_adds_failure_when_target_has_no_detection(tmp_path: Path) -> None:
    """YOLO確認済みでも対象器具の注釈がなければ検出失敗を追加する。"""
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    review: dict[str, Any] = {"annotation": {"shapes": []}, "objects": []}

    _count_bbox_for_image(_context(tmp_path, review, counts, failures, {"tool_a": 1}))

    assert counts["全体"].captured_instruments == 1
    assert counts["全体"].bbox_failure == EXPECTED_SINGLE_FAILURE
    assert counts["全体"].mask_failure == EXPECTED_SINGLE_FAILURE
    assert failures["bbox"]["tool_a"][0].no_detection


def test_bbox_failure_overlay_compares_initial_and_approved_boxes(tmp_path: Path) -> None:
    """修正前後を異なる色で描き、未承認候補は描かない。"""
    image_path = tmp_path / "source.png"
    Image.new("RGB", (50, 50), "white").save(image_path)
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    review: dict[str, Any] = {
        "annotation": {
            "shapes": [
                {"group_id": 0, "label": "tool_a", "points": [[15, 15], [28, 28]]},
                {"group_id": None, "label": "tool_a", "points": [[33, 4], [45, 16]]},
            ],
            "review_candidates": [
                {"group_id": None, "label": "tool_a", "points": [[2, 30], [10, 40]]},
            ],
        },
        "objects": [
            {"object_index": 0, "bbox_json": "[2,2,12,12]", "mask_path": "memory:0", "instrument_uid": "tool_a", "instance_no": 1, "dino_status": "accepted"},
            {"object_index": 1, "bbox_json": "[30,30,40,40]", "mask_path": "memory:1", "instrument_uid": "tool_a", "instance_no": 2, "dino_status": "accepted"},
        ],
    }

    _count_bbox_for_image(_context(tmp_path, review, counts, failures, {"tool_a": 2}))
    example = failures["bbox"]["tool_a"][0]
    destination = tmp_path / "failure.png"

    _save_failure_example(example, destination, "bbox", tmp_path)

    with Image.open(destination) as rendered:
        image_offset = rendered.height - 50
        assert rendered.getpixel((2, image_offset + 2)) == (40, 100, 220)
        assert rendered.getpixel((30, image_offset + 30)) == (40, 100, 220)
        assert rendered.getpixel((15, image_offset + 15)) == (255, 45, 45)
        assert rendered.getpixel((33, image_offset + 4)) == (255, 45, 45)
        assert rendered.getpixel((2, image_offset + 30)) == (255, 255, 255)
    assert example.boxes == [[15.0, 15.0, 28.0, 28.0], [33.0, 4.0, 45.0, 16.0]]
    assert example.initial_boxes == [[2.0, 2.0, 12.0, 12.0], [30.0, 30.0, 40.0, 40.0]]


def test_deleted_bbox_failure_overlay_shows_initial_box_only(tmp_path: Path) -> None:
    """削除されたbboxは修正前としてのみ描画し、検出なしとは表示しない。"""
    source_image_size = NO_DETECTION_IMAGE_SIZE
    image_path = tmp_path / "source.png"
    Image.new("RGB", (source_image_size, source_image_size), "white").save(image_path)
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    review: dict[str, Any] = {
        "annotation": {"shapes": []},
        "objects": [
            {"object_index": 0, "bbox_json": "[2,2,12,12]", "mask_path": "memory:0", "instrument_uid": "tool_a", "dino_status": "accepted"},
        ],
    }

    _count_bbox_for_image(_context(tmp_path, review, counts, failures, {"tool_a": 1}))
    example = failures["bbox"]["tool_a"][0]
    destination = tmp_path / "failure.png"
    _save_failure_example(example, destination, "bbox", tmp_path)

    with Image.open(destination) as rendered:
        image_offset = rendered.height - source_image_size
        assert rendered.width > source_image_size
        assert rendered.getpixel((2, image_offset + 2)) == (40, 100, 220)
        assert rendered.getpixel((2, image_offset + 13)) == (255, 255, 255)
    assert example.initial_boxes == [[2.0, 2.0, 12.0, 12.0]]
    assert example.boxes == []
    assert not example.no_detection


def test_bbox_failure_without_any_boxes_keeps_no_detection_caption(tmp_path: Path) -> None:
    """初期・承認後のどちらにもbboxがないときは検出なし表示を残す。"""
    image_path = tmp_path / "source.png"
    Image.new("RGB", (NO_DETECTION_IMAGE_SIZE, NO_DETECTION_IMAGE_SIZE), "white").save(image_path)
    example = FailureExample(CAPTURE_DATE, SESSION_ID, IMAGE_ID, image_path, no_detection=True)
    destination = tmp_path / "failure.png"
    without_caption = tmp_path / "without_caption.png"

    _save_failure_example(example, destination, "bbox", tmp_path)
    _save_failure_example(
        FailureExample(CAPTURE_DATE, SESSION_ID, IMAGE_ID, image_path),
        without_caption,
        "bbox",
        tmp_path,
    )

    with Image.open(destination) as rendered:
        assert rendered.height > NO_DETECTION_IMAGE_SIZE
        caption_height = rendered.height
    with Image.open(without_caption) as rendered_without_caption:
        assert caption_height > rendered_without_caption.height
    assert example.no_detection


def test_generated_table_and_chart_wrap_long_ids_without_text_overlap(tmp_path: Path) -> None:
    """表・グラフの日本語見出しと長いUIDを省略せず、別領域に描画する。"""
    long_uid = "器具UID-" + "tool_0123456789-" * 5
    row = Counts(bbox_failure=1, mask_failure=1).row(long_uid)
    table_path_prefix = tmp_path / "instrument_table"
    chart_path = tmp_path / "chart.png"
    measure_image = Image.new("RGB", (2000, 1), "white")
    measure_draw = ImageDraw.Draw(measure_image)
    table_font = _font(17)
    chart_font = _font(18)

    uid_lines = _wrap_text(measure_draw, long_uid, table_font, TABLE_COLUMN_WIDTHS[0] - TABLE_CELL_PADDING * 2)
    assert "".join(uid_lines) == long_uid
    for line in uid_lines:
        assert _text_width(measure_draw, line, table_font) <= TABLE_COLUMN_WIDTHS[0] - TABLE_CELL_PADDING * 2
    for column_number, column in enumerate(TABLE_COLUMNS):
        available_width = TABLE_COLUMN_WIDTHS[column_number] - TABLE_CELL_PADDING * 2
        header_lines = _wrap_text(measure_draw, column, table_font, available_width)
        assert "".join(header_lines) == column
        for line in header_lines:
            assert _text_width(measure_draw, line, table_font) <= available_width
    assert _table_row_height(
        measure_draw,
        [long_uid] + ["0"] * (len(TABLE_COLUMN_WIDTHS) - 1),
        table_font,
    ) > MINIMUM_TABLE_ROW_HEIGHT

    _write_table_pages(table_path_prefix, [row])
    average_chart_path = tmp_path / "average_chart.png"
    _write_chart(chart_path, [row], "器具種別ごとの成功率")
    _write_average_chart(average_chart_path, [row], "器具あたり平均未採用bbox数")

    with Image.open(tmp_path / "instrument_table_01.png") as table_image:
        assert table_image.width == sum(TABLE_COLUMN_WIDTHS)
        assert table_image.height > MINIMUM_MULTILINE_TABLE_HEIGHT
    with Image.open(chart_path) as chart_image:
        assert chart_image.width > CHART_LABEL_WIDTH + CHART_METRIC_WIDTH + CHART_WIDTH
    with Image.open(average_chart_path) as chart_image:
        assert chart_image.width > CHART_LABEL_WIDTH + CHART_WIDTH
    chart_uid_lines = _wrap_text(measure_draw, long_uid, chart_font, CHART_LABEL_WIDTH)
    assert "".join(chart_uid_lines) == long_uid
    for line in chart_uid_lines:
        assert _text_width(measure_draw, line, chart_font) <= CHART_LABEL_WIDTH
    bbox_label_width = _text_width(measure_draw, "bbox成功率", chart_font)
    mask_label_width = _text_width(measure_draw, "mask成功率", chart_font)
    assert max(bbox_label_width, mask_label_width) <= CHART_METRIC_WIDTH
    label_left = 24
    metric_left = label_left + CHART_LABEL_WIDTH + 20
    chart_left = metric_left + CHART_METRIC_WIDTH
    rate_left = chart_left + CHART_WIDTH
    assert label_left + CHART_LABEL_WIDTH <= metric_left
    assert metric_left + CHART_METRIC_WIDTH <= chart_left
    assert chart_left + CHART_WIDTH <= rate_left


def test_instrument_rate_and_overdetection_use_capture_count() -> None:
    """成功率と未採用bbox平均は撮影器具数を分母にする。"""
    row = Counts(
        captured_instruments=10,
        bbox_success=8,
        mask_success=7,
        unaccepted_bbox_count=30,
    ).row("tool_a")

    assert row["bbox成功率"] == "80.0%"
    assert row["mask成功率"] == "70.0%"
    assert row["器具あたり平均未採用bbox数"] == "3.0"


def test_overdetection_counts_every_unaccepted_model_box(tmp_path: Path) -> None:
    """1成功＋2失敗なら2個、3失敗なら3個の未採用bboxとする。"""
    for accepted_count, expected_unaccepted in ((1, 2), (0, 3)):
        counts: dict[str, Counts] = {"全体": Counts()}
        failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
        shapes = []
        if accepted_count:
            shapes.append({"group_id": 0, "label": "tool_a", "points": [[1, 1], [6, 6]]})
        review: dict[str, Any] = {
            "annotation": {"shapes": shapes, "review_candidates": []},
            "objects": [
                {
                    "object_index": index,
                    "bbox_json": json.dumps([1 + index, 1, 6 + index, 6]),
                    "mask_path": f"memory:{index}",
                    "instrument_uid": "tool_a",
                    "instance_no": 1,
                    "dino_status": "pending",
                }
                for index in range(3)
            ],
        }

        _count_bbox_for_image(_context(tmp_path, review, counts, failures, {"tool_a": 1}))

        assert counts["全体"].captured_instruments == 1
        assert counts["全体"].bbox_success == accepted_count
        assert counts["全体"].bbox_failure == 1 - accepted_count
        assert counts["全体"].unaccepted_bbox_count == expected_unaccepted


def test_human_added_bbox_fails_capture_unit_without_overdetection(tmp_path: Path) -> None:
    """人手追加bboxだけでは成功せず、モデル未採用bbox数にも加えない。"""
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    review: dict[str, Any] = {
        "annotation": {
            "shapes": [{"group_id": None, "label": "tool_a", "points": [[2, 2], [8, 8]]}],
            "review_candidates": [],
        },
        "objects": [],
    }

    _count_bbox_for_image(_context(tmp_path, review, counts, failures, {"tool_a": 1}))

    assert counts["全体"].bbox_failure == 1
    assert counts["全体"].unaccepted_bbox_count == 0
    assert failures["bbox"]["tool_a"][0].boxes == [[2.0, 2.0, 8.0, 8.0]]


def test_missing_object_uid_uses_single_visible_instrument(tmp_path: Path) -> None:
    """画像内の可視器具が一意なら、UIDのない物体もその器具へ対応する。"""
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    review: dict[str, Any] = {
        "annotation": {
            "shapes": [{"group_id": 0, "label": "tool_a", "points": [[1, 1], [6, 6]]}],
            "review_candidates": [],
        },
        "objects": [
            {
                "object_index": 0,
                "bbox_json": "[1,1,6,6]",
                "mask_path": "memory:0",
                "instrument_uid": None,
                "instance_no": None,
                "dino_status": "accepted",
            }
        ],
    }

    _count_bbox_for_image(_context(tmp_path, review, counts, failures, {"tool_a": 1}))

    assert counts["全体"].bbox_success == 1
    assert counts["tool_a"].captured_instruments == 1
    assert "器具未確定" not in counts


def test_missing_object_uid_with_multiple_instruments_fails_explicitly(tmp_path: Path) -> None:
    """複数器具から物体UIDを特定できない場合、未確定行に混ぜず停止する。"""
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    review: dict[str, Any] = {
        "annotation": {"shapes": [], "review_candidates": []},
        "objects": [
            {
                "object_index": 0,
                "bbox_json": "[1,1,6,6]",
                "mask_path": "memory:0",
                "instrument_uid": None,
                "instance_no": None,
                "dino_status": "pending",
            }
        ],
    }

    with pytest.raises(CaptureError, match="対応付けられません"):
        _count_bbox_for_image(
            _context(
                tmp_path,
                review,
                counts,
                failures,
                {"tool_a": 1, "tool_b": 1},
            )
        )


def test_instrument_names_are_loaded_read_only_and_duplicate_names_keep_uid(tmp_path: Path) -> None:
    """master DBのname_jaを使い、同名器具はUID付きで区別する。"""
    database = tmp_path / "dataset.sqlite"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("CREATE TABLE samples (id TEXT PRIMARY KEY, name_ja TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO samples VALUES (?, ?)",
            [("tool_a", "器具"), ("tool_b", "器具")],
        )
    before = database.read_bytes()

    names = _load_instrument_names(database)
    display_names = _instrument_display_names(["tool_a", "tool_b", "tool_c"], names)

    assert names == {"tool_a": "器具", "tool_b": "器具"}
    assert display_names == {
        "tool_a": "器具（tool_a）",
        "tool_b": "器具（tool_b）",
        "tool_c": "tool_c",
    }
    assert database.read_bytes() == before


def test_count_mask_counts_pending_as_instrument_failure(tmp_path: Path) -> None:
    """pendingは撮影器具数の分母に含み、mask失敗に数える。"""
    counts: dict[str, Counts] = {"全体": Counts()}
    failures: dict[str, dict[str, list[FailureExample]]] = {"bbox": {}, "mask": {}}
    review: dict[str, Any] = {
        "objects": [
            {"object_index": 0, "instrument_uid": "tool_a", "instance_no": 1, "dino_status": "pending", "mask_path": "memory:0"},
        ]
    }

    _count_mask_only(_context(tmp_path, review, counts, failures, {"tool_a": 1}))

    assert counts["全体"].mask_success == EXPECTED_NONE
    assert counts["全体"].mask_failure == EXPECTED_MASK_FAILURES
    assert counts["全体"].mask_pending == EXPECTED_PENDING
    assert len(failures["mask"]["tool_a"]) == 1


def test_failure_example_renders_available_mask_without_changing_original(tmp_path: Path) -> None:
    """利用可能なmaskを別画像へ重ね、元撮影画像を変更しない。"""
    image_path = tmp_path / "source.png"
    original_image = Image.new("RGB", (20, 20), (200, 200, 200))
    original_image.save(image_path)
    original_bytes = image_path.read_bytes()
    mask_path = tmp_path / "masks" / "mask.png"
    mask_path.parent.mkdir()
    Image.new("L", (20, 20), 255).save(mask_path)
    destination = tmp_path / "failure.png"
    example = FailureExample(
        CAPTURE_DATE,
        SESSION_ID,
        IMAGE_ID,
        image_path,
        mask_paths=[mask_path.relative_to(tmp_path).as_posix()],
    )

    _save_failure_example(example, destination, "mask", tmp_path)

    with Image.open(destination) as rendered:
        assert rendered.getpixel((10, 10)) != (200, 200, 200)
    assert image_path.read_bytes() == original_bytes
