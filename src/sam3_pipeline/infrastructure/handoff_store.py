"""セッションDB内のDINO・YOLO専用テーブルを承認状態と同期する。

責務:
    推論原本と編集済み注釈の公開スナップショットを保持し、下流の処理済み状態は管理しない。
処理の流れ:
    1. 専用テーブルと撮影情報変更時の失効トリガーを作成する。
    2. 推論・レビュー・承認台帳を同一トランザクションで照合する。
    3. 独立した公開可否と出典を各テーブルへ反映する。
関連ファイル:
    object_review_store.py、annotation_ledger.py、prediction_store.py。
設計意図:
    公開テーブルを別の編集可能な正本にせず、保存済み推論・確認履歴からのみ更新する。

Note:
    呼び出し元のSQLiteトランザクション内で実行する。ファイルhashの再検証は取得側でも必要である。
"""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_revisions (
    session_id TEXT NOT NULL, image_id TEXT NOT NULL, tag_revision INTEGER NOT NULL,
    revision TEXT NOT NULL, status TEXT NOT NULL, approval_id TEXT, reviewer TEXT,
    approved_at TEXT, manifest_path TEXT, annotation_path TEXT NOT NULL,
    annotation_sha256 TEXT NOT NULL, exported_at TEXT, updated_at TEXT NOT NULL,
    PRIMARY KEY(session_id, image_id, revision)
);
CREATE TABLE IF NOT EXISTS sam3_dino_results (
    revision TEXT NOT NULL, object_index INTEGER NOT NULL,
    session_id TEXT NOT NULL, image_id TEXT NOT NULL, prediction_id TEXT NOT NULL,
    tag_revision INTEGER NOT NULL, source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL, depth_path TEXT,
    width INTEGER NOT NULL, height INTEGER NOT NULL,
    prompt TEXT, min_score REAL, model_signature TEXT, model_metadata_json TEXT NOT NULL,
    provenance TEXT NOT NULL, sam3_score REAL NOT NULL, bbox_json TEXT NOT NULL,
    mask_path TEXT NOT NULL, mask_sha256 TEXT NOT NULL,
    instrument_uid TEXT, instance_no INTEGER, review_status TEXT NOT NULL,
    reviewer TEXT, reviewed_at TEXT, review_event_id INTEGER, approval_id TEXT,
    is_dino_ready INTEGER NOT NULL CHECK(is_dino_ready IN (0, 1)),
    PRIMARY KEY(revision, object_index)
);
CREATE TABLE IF NOT EXISTS sam3_yolo_results (
    revision TEXT PRIMARY KEY, session_id TEXT NOT NULL, image_id TEXT NOT NULL,
    tag_revision INTEGER NOT NULL, source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL,
    source_json TEXT NOT NULL, annotation_json TEXT NOT NULL, annotation_sha256 TEXT, annotation_path TEXT,
    approval_json TEXT,
    status TEXT NOT NULL, approval_id TEXT, reviewer TEXT, approved_at TEXT,
    allow_empty INTEGER NOT NULL, exception_note TEXT NOT NULL,
    manifest_path TEXT, exported_at TEXT,
    is_yolo_ready INTEGER NOT NULL CHECK(is_yolo_ready IN (0, 1))
);
"""

DINO_SELECT = """
SELECT r.revision, o.object_index, r.session_id, r.image_id, r.prediction_id,
    r.tag_revision, p.source_path, p.source_sha256, i.depth_path, p.width, p.height,
    json_extract(p.model_metadata_json, '$.prompt'), json_extract(p.model_metadata_json, '$.min_score'),
    p.model_signature, p.model_metadata_json, p.provenance, o.sam3_score, o.bbox_json,
    COALESCE(m.mask_path, o.mask_path), COALESCE(m.mask_sha256, o.mask_sha256),
    v.instrument_uid, v.instance_no, v.dino_status,
    v.reviewer, v.reviewed_at,
    (SELECT MAX(e.event_id) FROM sam3_object_review_events e
     WHERE e.revision=r.revision AND e.object_index=o.object_index),
    r.approval_id,
    CASE WHEN r.is_current=1 AND r.status IN ('approved', 'exported') AND r.approval_id IS NOT NULL
        AND (o.mask_path NOT LIKE 'memory:%' OR m.mask_path IS NOT NULL)
        AND v.dino_status='accepted' AND v.instrument_uid IS NOT NULL AND v.instance_no IS NOT NULL
        AND i.tag_status='tagged' AND i.tag_revision=r.tag_revision
        AND i.sha256=p.source_sha256 AND i.path=p.source_path
        AND i.width=p.width AND i.height=p.height THEN 1 ELSE 0 END
FROM sam3_review_revisions r
JOIN sam3_predictions p USING(session_id, image_id, prediction_id)
JOIN sam3_prediction_objects o USING(session_id, image_id, prediction_id)
JOIN sam3_object_reviews v ON v.revision=r.revision AND v.object_index=o.object_index
LEFT JOIN sam3_approved_masks m ON m.revision=r.revision AND m.object_index=o.object_index
LEFT JOIN images i ON i.session_id=r.session_id AND i.image_id=r.image_id
"""

YOLO_SELECT = """
SELECT r.revision, r.session_id, r.image_id, r.tag_revision, r.source_path, r.source_sha256,
    json_extract(r.metadata_json, '$.source'), r.annotation_json, a.annotation_sha256, a.annotation_path, r.approval_json,
    CASE WHEN a.status='exported' AND a.approval_id=r.approval_id AND r.status='approved'
        THEN 'exported' ELSE r.status END, r.approval_id,
    json_extract(r.approval_json, '$.reviewer'), json_extract(r.approval_json, '$.approved_at'),
    COALESCE(json_extract(r.approval_json, '$.allow_empty'), 0),
    COALESCE(json_extract(r.approval_json, '$.exception_note'), ''), a.manifest_path, a.exported_at,
    CASE WHEN r.is_current=1 AND r.status IN ('approved', 'exported') AND r.approval_id IS NOT NULL
        AND a.status='exported' AND a.approval_id=r.approval_id AND a.tag_revision=r.tag_revision
        AND a.annotation_sha256=json_extract(r.approval_json, '$.labelme_sha256')
        AND a.manifest_path IS NOT NULL AND a.exported_at IS NOT NULL
        AND i.tag_status='tagged' AND i.tag_revision=r.tag_revision
        AND i.sha256=r.source_sha256 AND i.path=r.source_path
        AND i.width=p.width AND i.height=p.height THEN 1 ELSE 0 END
FROM sam3_review_revisions r
JOIN sam3_predictions p USING(session_id, image_id, prediction_id)
LEFT JOIN annotation_revisions a USING(session_id, image_id, revision)
LEFT JOIN images i ON i.session_id=r.session_id AND i.image_id=r.image_id
"""

IMAGE_INVALIDATION_TRIGGERS = (
    """CREATE TRIGGER IF NOT EXISTS sam3_handoff_image_update AFTER UPDATE ON images
        WHEN OLD.tag_revision IS NOT NEW.tag_revision OR OLD.tag_status IS NOT NEW.tag_status
            OR OLD.sha256 IS NOT NEW.sha256 OR OLD.path IS NOT NEW.path
            OR OLD.width IS NOT NEW.width OR OLD.height IS NOT NEW.height
            OR OLD.session_id IS NOT NEW.session_id OR OLD.image_id IS NOT NEW.image_id
        BEGIN
            UPDATE sam3_dino_results SET is_dino_ready=0
                WHERE session_id=OLD.session_id AND image_id=OLD.image_id;
            UPDATE sam3_yolo_results SET is_yolo_ready=0
                WHERE session_id=OLD.session_id AND image_id=OLD.image_id;
        END""",
    """CREATE TRIGGER IF NOT EXISTS sam3_handoff_image_delete AFTER DELETE ON images
        BEGIN
            UPDATE sam3_dino_results SET is_dino_ready=0
                WHERE session_id=OLD.session_id AND image_id=OLD.image_id;
            UPDATE sam3_yolo_results SET is_yolo_ready=0
                WHERE session_id=OLD.session_id AND image_id=OLD.image_id;
        END""",
)


def initialize_handoff(connection: sqlite3.Connection) -> None:
    """公開テーブルと撮影情報変更時の失効処理を冪等に追加する。

    Args:
        connection: 呼び出し元のトランザクション内にあるSQLite接続。
    """
    # executescriptの暗黙commitを避け、承認履歴と公開状態の原子性を維持する。
    for statement in SCHEMA.split(";"):
        if statement.strip():
            connection.execute(statement)
    # 撮影APIから直接タグが変わった場合も、次のSAM3起動を待たずに公開を停止する。
    for statement in IMAGE_INVALIDATION_TRIGGERS:
        connection.execute(statement)


def refresh_handoff(connection: sqlite3.Connection) -> None:
    """保存済みの原本・承認状態から専用テーブルを同一トランザクションで更新する。

    Args:
        connection: 行名アクセスが有効なSQLite接続。
    """
    # 台帳のみを使う呼び出しでは、未作成の推論・レビューテーブルに依存させない。
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='sam3_review_revisions'").fetchone():
        return
    for table, query in (("sam3_dino_results", DINO_SELECT), ("sam3_yolo_results", YOLO_SELECT)):
        columns = [row["name"] for row in connection.execute(f"PRAGMA table_info({table})")]
        keys = ("revision", "object_index") if table == "sam3_dino_results" else ("revision",)
        updates = ", ".join(f"{name}=excluded.{name}" for name in columns if name not in keys)
        # 履歴を削除せず、旧版や取消の行も公開不可として保持する。
        connection.execute(f"INSERT INTO {table} {query} WHERE 1 ON CONFLICT({', '.join(keys)}) DO UPDATE SET {updates}")
