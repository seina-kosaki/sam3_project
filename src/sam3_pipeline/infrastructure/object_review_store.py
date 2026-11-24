"""推論原本とは別のSQL領域で物体確認・画像承認・公開条件を管理する。

責務:
    セッションごとのレビュー・採否履歴・公開状態を担い、推論原本は編集しない。
処理の流れ:
    1. 画像またはレビュー版から所属DBを特定する。
    2. 現行性と原本を検証し、レビュー・画像承認・物体採否を記録する。
    3. 同一トランザクションでDINO・YOLO専用テーブルを同期する。
関連ファイル:
    prediction_store.py、annotation_store.py、handoff_store.py、session_database.py。
設計意図:
    採否履歴を維持し、YOLO用の編集済みbboxをDINOの推論原本へ混入させない。

Note:
    公開時に撮影SQLのタグ版と原本情報を照合する。ファイルhashは取得側でも再検証する。
"""

from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sam3_pipeline.domain.annotation import Candidate, PredictionStore
from sam3_pipeline.domain.capture import CaptureError, CaptureSample
from sam3_pipeline.infrastructure.annotation_store import content_hash, digest_file
from sam3_pipeline.infrastructure.approved_masks import capture_root, mask_destination, publish_masks
from sam3_pipeline.infrastructure.handoff_store import initialize_handoff, refresh_handoff
from sam3_pipeline.infrastructure.matching import ArrayMask
from sam3_pipeline.infrastructure.session_database import (
    capture_date_from_source_path,
    metadata_databases,
    review_database,
    session_database,
)

REVIEW_SCHEMA_VERSION = 5
SUPPORTED_REVIEW_VERSIONS = frozenset({1, 2, 3, 4, REVIEW_SCHEMA_VERSION})
DINO_STATUSES = frozenset({"pending", "accepted", "rejected"})
APPROVED_STATUSES = frozenset({"approved", "exported"})
OVERLAY_COLOR = (0, 155, 110, 0)
OVERLAY_ALPHA = 110
SCHEMA = """
CREATE TABLE IF NOT EXISTS sam3_review_schema_migrations (
    version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sam3_review_revisions (
    revision TEXT PRIMARY KEY, session_id TEXT NOT NULL, image_id TEXT NOT NULL,
    prediction_id TEXT NOT NULL, tag_revision INTEGER NOT NULL,
    source_sha256 TEXT NOT NULL, source_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL, annotation_json TEXT NOT NULL,
    progress_json TEXT NOT NULL DEFAULT '{}',
    is_current INTEGER NOT NULL CHECK(is_current IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'inferred', approval_id TEXT, approval_json TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id, image_id, prediction_id)
        REFERENCES sam3_predictions(session_id, image_id, prediction_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_sam3_current_review
    ON sam3_review_revisions(session_id, image_id) WHERE is_current=1;
CREATE TABLE IF NOT EXISTS sam3_object_reviews (
    revision TEXT NOT NULL REFERENCES sam3_review_revisions(revision),
    object_index INTEGER NOT NULL, instrument_uid TEXT, instance_no INTEGER,
    dino_status TEXT NOT NULL DEFAULT 'pending' CHECK(dino_status IN ('pending', 'accepted', 'rejected')),
    reviewer TEXT, reviewed_at TEXT,
    PRIMARY KEY(revision, object_index)
);
CREATE TABLE IF NOT EXISTS sam3_object_review_events (
    event_id INTEGER PRIMARY KEY, revision TEXT NOT NULL, object_index INTEGER NOT NULL,
    dino_status TEXT NOT NULL CHECK(dino_status IN ('pending', 'accepted', 'rejected')),
    reviewer TEXT NOT NULL, reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(revision, object_index) REFERENCES sam3_object_reviews(revision, object_index)
);
CREATE TABLE IF NOT EXISTS sam3_review_events (
    event_id INTEGER PRIMARY KEY, revision TEXT NOT NULL REFERENCES sam3_review_revisions(revision),
    status TEXT NOT NULL, annotation_json TEXT NOT NULL, approval_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sam3_object_assignment_events (
    event_id INTEGER PRIMARY KEY, revision TEXT NOT NULL, object_index INTEGER NOT NULL,
    previous_uid TEXT, previous_instance_no INTEGER, instrument_uid TEXT, instance_no INTEGER,
    reviewer TEXT NOT NULL, reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(revision, object_index) REFERENCES sam3_object_reviews(revision, object_index)
);
CREATE TABLE IF NOT EXISTS sam3_approved_masks (
    revision TEXT NOT NULL, object_index INTEGER NOT NULL,
    mask_path TEXT NOT NULL, mask_sha256 TEXT NOT NULL,
    PRIMARY KEY(revision, object_index),
    FOREIGN KEY(revision, object_index) REFERENCES sam3_object_reviews(revision, object_index)
);
DROP VIEW IF EXISTS sam3_dino_objects;
CREATE VIEW sam3_dino_objects AS
SELECT revisions.session_id, revisions.image_id, revisions.revision,
    revisions.prediction_id, objects.object_index, reviews.instrument_uid,
    reviews.instance_no, predictions.source_path AS image_path,
    images.depth_path, predictions.source_sha256, predictions.width, predictions.height,
    objects.bbox_json, COALESCE(masks.mask_path, objects.mask_path) AS mask_path,
    COALESCE(masks.mask_sha256, objects.mask_sha256) AS mask_sha256, objects.sam3_score,
    predictions.model_signature, predictions.model_metadata_json, predictions.provenance,
    revisions.approval_id, reviews.reviewer, reviews.reviewed_at,
    (SELECT MAX(events.event_id) FROM sam3_object_review_events AS events
     WHERE events.revision=reviews.revision AND events.object_index=reviews.object_index) AS review_event_id
FROM sam3_review_revisions AS revisions
JOIN sam3_predictions AS predictions
    USING(session_id, image_id, prediction_id)
JOIN sam3_prediction_objects AS objects
    USING(session_id, image_id, prediction_id)
JOIN sam3_object_reviews AS reviews
    ON reviews.revision=revisions.revision AND reviews.object_index=objects.object_index
JOIN images ON images.session_id=revisions.session_id AND images.image_id=revisions.image_id
LEFT JOIN sam3_approved_masks AS masks
    ON masks.revision=reviews.revision AND masks.object_index=reviews.object_index
WHERE revisions.is_current=1 AND revisions.status IN ('approved', 'exported')
    AND (objects.mask_path NOT LIKE 'memory:%' OR masks.mask_path IS NOT NULL)
    AND revisions.approval_id IS NOT NULL AND reviews.dino_status='accepted'
    AND reviews.instrument_uid IS NOT NULL AND reviews.instance_no IS NOT NULL
    AND images.tag_status='tagged' AND images.tag_revision=revisions.tag_revision
    AND images.sha256=predictions.source_sha256 AND images.path=predictions.source_path
    AND images.width=predictions.width AND images.height=predictions.height;
"""


class SqliteObjectReviewStore:
    """レビュー注釈と物体採否をSQLへ保持し、原本からのみDINO向けに公開する。

    Args:
        database_path: 共有ルート、またはSAM3推論テーブルと撮影SQLを持つセッションDB。
        timeout: SQLiteのロック待ち秒数。
        predictions: PNGと推論原本を検証する保存ポート。

    Note:
        旧承認の同期でも物体採否の初期値はpendingである。採否を高信頼度から自動決定しない。
    """

    def __init__(self, database_path: Path, timeout: float, predictions: PredictionStore) -> None:
        """設定済み接続先と推論原本検証ポートを束縛する。"""
        self.database_path = database_path.resolve()
        self.timeout = timeout
        self.predictions = predictions
        self.database_paths: list[Path] | None = None

    def initialize(self, databases: list[Path] | None = None) -> None:
        """撮影・推論テーブルを変更せずレビューschemaを冪等に追加する。

        Raises:
            CaptureError: DBが存在しない、未対応版、または追加に失敗した場合。
        """
        # セッションごとのschemaを維持し、ルート直下に集約DBを生成しない。
        if databases is not None:
            self.database_paths = databases
            for database in databases:
                SqliteObjectReviewStore(database, self.timeout, self.predictions).initialize()
            return
        if self.database_path.is_dir():
            for database in metadata_databases(self.database_path):
                SqliteObjectReviewStore(database, self.timeout, self.predictions).initialize()
            return
        with self._connection(refresh=False) as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='sam3_review_schema_migrations'").fetchone()
            if exists and any(row[0] not in SUPPORTED_REVIEW_VERSIONS for row in connection.execute("SELECT version FROM sam3_review_schema_migrations")):
                raise CaptureError("Unsupported SAM3 review schema version")
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute("INSERT OR IGNORE INTO sam3_review_schema_migrations(version) VALUES (?)", (REVIEW_SCHEMA_VERSION,))
            initialize_handoff(connection)
            refresh_handoff(connection)

    def sync_revision(self, sample: CaptureSample, revision: str, metadata: dict[str, Any], annotation: dict[str, Any]) -> None:
        """画像・推論・器具対応を固定した現行レビュー版を登録する。

        Args:
            sample: 原本を照合済みの撮影スナップショット。
            revision: 最新レビュー版。
            metadata: prediction_keyと推論候補の器具対応を含む草案。
            annotation: YOLO向け編集可能注釈の現在値。

        Raises:
            CaptureError: 推論が未保存、原本が不整合、または保存に失敗した場合。
        """
        prediction_id = str(metadata["prediction_key"])
        if revision != content_hash(metadata):
            raise CaptureError("Review metadata differs from its revision")
        # 一覧復元にmaskを要求せず、推論情報だけでレビューを同期する。
        with self._connection(sample=sample, refresh=False) as connection:
            header = connection.execute("SELECT object_count FROM sam3_predictions WHERE session_id=? AND image_id=? AND prediction_id=?", (sample.session_id, sample.image_id, prediction_id)).fetchone()
            candidate_indices = {row[0] for row in connection.execute("SELECT object_index FROM sam3_prediction_objects WHERE session_id=? AND image_id=? AND prediction_id=?", (sample.session_id, sample.image_id, prediction_id))}
        if header is None or header["object_count"] != len(candidate_indices):
            raise CaptureError("Prediction must be generated before object review")
        assignments = {item["candidate_index"]: item for item in metadata["matching"]["assignments"]}
        allowed = {(tag.instrument_uid, tag.instance_no) for tag in sample.tags if tag.visible}
        identities = [(item["instrument_uid"], item["instance_no"]) for item in assignments.values()]
        if len(assignments) != len(metadata["matching"]["assignments"]) or len(set(identities)) != len(identities) or any(identity not in allowed for identity in identities):
            raise CaptureError("Invalid instrument assignment")
        if not set(assignments).issubset(candidate_indices):
            raise CaptureError("Assignment references an unknown prediction object")
        with self._connection(sample=sample) as connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT is_current, annotation_json FROM sam3_review_revisions WHERE revision=?", (revision,)).fetchone()
            connection.execute("UPDATE sam3_review_revisions SET is_current=0 WHERE session_id=? AND image_id=? AND revision<>?", (sample.session_id, sample.image_id, revision))
            connection.execute(
                """INSERT INTO sam3_review_revisions(revision, session_id, image_id, prediction_id,
                    tag_revision, source_sha256, source_path, metadata_json, annotation_json, is_current)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(revision) DO UPDATE SET is_current=1""",
                (revision, sample.session_id, sample.image_id, prediction_id, sample.tag_revision, sample.sha256, sample.source_path, self._json(metadata), self._json(annotation)),
            )
            if old is not None and not old["is_current"]:
                # 過去の版への復帰でも、当時の確認を現在の確認として自動再利用しない。
                self._reset(connection, revision, "revision_changed")
                connection.execute("UPDATE sam3_review_revisions SET status='inferred', approval_id=NULL, approval_json=NULL WHERE revision=?", (revision,))
            if old is not None and old["annotation_json"] != self._json(annotation):
                # 外部で変更された注釈も再承認させ、古いJSON承認をSQLの正本へ上書きしない。
                connection.execute(
                    """UPDATE sam3_review_revisions SET annotation_json=?, status='inferred',
                    approval_id=NULL, approval_json=NULL WHERE revision=?""",
                    (self._json(annotation), revision),
                )
            for index in candidate_indices:
                assignment = assignments.get(index, {})
                connection.execute("INSERT OR IGNORE INTO sam3_object_reviews(revision, object_index, instrument_uid, instance_no) VALUES (?, ?, ?, ?)", (revision, index, assignment.get("instrument_uid"), assignment.get("instance_no")))

    def record_annotation(
        self,
        revision: str,
        annotation: dict[str, Any],
        status: str,
        approval: dict[str, Any] | None = None,
        *,
        decisions: dict[int, str] | None = None,
        sample: CaptureSample | None = None,
        overwrite_masks: bool = False,
        assignments: dict[int, tuple[str, int] | None] | None = None,
        decision_reviewer: str | None = None,
    ) -> None:
        """YOLO注釈と画像承認を同期し、取消時に物体採否も未確認へ戻す。

        Args:
            revision: 対象のレビュー版。
            annotation: YOLO用の編集可能注釈。
            status: 画像の確認・承認・取消・公開状態。
            approval: 明示承認の記録。承認以外はNoneである。
            decisions: 今回確定する物体の採否。画像承認と同じトランザクションで保存する。
            sample: 一括確定時の撮影SQL・原本照合に用いる画像。
            overwrite_masks: 既存承認用maskの上書きに利用者が同意したか。
            assignments: DINO候補ごとの手動対応。採否・承認と同時に確定する。
            decision_reviewer: DINO確認の担当者。既存のYOLO承認者とは独立して記録する。

        Raises:
            CaptureError: 対象版が存在しないか、記録に失敗した場合。
        """
        # PNG破損を含む原本不整合を、画像承認の書き込み前に検出する。
        candidates: tuple[Candidate, ...] = ()
        write_masks = False
        if assignments is not None and decisions is None:
            raise CaptureError("手動対応には物体の採否が必要である")
        if decisions is not None and (sample is None or status != "approved" or approval is None or not str(approval.get("reviewer", "")).strip()):
            raise CaptureError("物体判定の一括保存には画像と担当者の承認が必要である")
        if sample is not None and status == "approved":
            # 再推論が必要でもDBの書込ロックを取る前に完了させる。
            with self._connection(sample=sample, refresh=False) as connection:
                prediction_id = self._current(connection, sample, revision)["prediction_id"]
                missing = connection.execute(
                    """SELECT 1 FROM sam3_object_reviews v
                    LEFT JOIN sam3_approved_masks m USING(revision, object_index)
                    WHERE v.revision=? AND v.dino_status='accepted' AND m.mask_path IS NULL LIMIT 1""",
                    (revision,),
                ).fetchone()
            write_masks = decisions is not None or missing is not None
            if write_masks:
                loaded = self.predictions.load(sample, prediction_id)
                if loaded is None:
                    raise CaptureError("確認用maskがない。runでSAM3推論を実行してください")
                candidates = loaded
        # ファイル復元の終了処理をSQL commitより外側に置き、commit失敗も復元対象にする。
        with ExitStack() as assets, self._connection(revision=revision) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM sam3_review_revisions WHERE revision=?", (revision,)).fetchone()
            if row is None:
                raise CaptureError("Object review revision is missing")
            approved = status in APPROVED_STATUSES
            if approved and (approval is None or not approval.get("approval_id")):
                raise CaptureError("Approval evidence is required")
            # 全件検証後に更新し、途中の物体だけが公開される状態を作らない。
            if decisions is not None and sample is not None and approval is not None:
                reviewer = decision_reviewer.strip() if decision_reviewer is not None else approval["reviewer"]
                if not reviewer:
                    raise CaptureError("DINO確認担当者を入力してください")
                self._current(connection, sample, revision)
                objects = connection.execute("SELECT * FROM sam3_object_reviews WHERE revision=?", (revision,)).fetchall()
                object_indices = {item["object_index"] for item in objects}
                if not decisions or not set(decisions).issubset(object_indices):
                    raise CaptureError("保存対象の候補を指定してください")
                if assignments is not None and not set(assignments).issubset(set(decisions)):
                    raise CaptureError("タグ対応は採否を指定した候補だけに指定してください")
                allowed = {(tag.instrument_uid, tag.instance_no) for tag in sample.tags if tag.visible}
                decisions = dict(decisions)
                accepted_by_tag: dict[tuple[str, int], int] = {}
                for item in objects:
                    index = item["object_index"]
                    if decisions.get(index) != "accepted":
                        continue
                    identity = (item["instrument_uid"], item["instance_no"]) if item["instrument_uid"] is not None and item["instance_no"] is not None else None
                    if assignments is not None and index in assignments:
                        identity = assignments[index]
                    if identity is None:
                        raise CaptureError("器具未確定の物体はDINOに使えない")
                    if identity not in allowed:
                        raise CaptureError("現在の可視タグにない器具対応である")
                    if identity in accepted_by_tag:
                        raise CaptureError("一つのタグに複数のDINO候補は採用できない")
                    accepted_by_tag[identity] = index
                for item in objects:
                    index = item["object_index"]
                    if index in decisions or item["dino_status"] == "rejected":
                        continue
                    identity = (item["instrument_uid"], item["instance_no"]) if item["instrument_uid"] is not None and item["instance_no"] is not None else None
                    if identity in accepted_by_tag:
                        decisions[index] = "rejected"
                accepted: set[tuple[str, int]] = set()
                for item in objects:
                    index = item["object_index"]
                    decision = decisions.get(index, item["dino_status"])
                    identity = (item["instrument_uid"], item["instance_no"]) if item["instrument_uid"] is not None and item["instance_no"] is not None else None
                    if assignments is not None and index in assignments:
                        identity = assignments[index]
                    if index in decisions and decision not in {"accepted", "rejected"}:
                        raise CaptureError("保存対象の候補は「使う／使わない」を選択してください")
                    if identity is not None and identity not in allowed and (index in decisions or decision == "accepted"):
                        raise CaptureError("現在の可視タグにない器具対応である")
                    if decision == "accepted" and identity is None:
                        raise CaptureError("器具未確定の物体はDINOに使えない")
                    if decision == "accepted" and identity is not None:
                        if identity in accepted:
                            raise CaptureError("一つのタグに複数のDINO候補は採用できない")
                        accepted.add(identity)
                if not accepted:
                    raise CaptureError("採用されたmaskがないため、DINO保存対象がありません")
                # 対応だけが先に残らないよう、検証後に採否・承認・maskと同じトランザクションへ記録する。
                if assignments is not None:
                    for item in objects:
                        index = item["object_index"]
                        if index not in assignments:
                            continue
                        identity = assignments[index]
                        uid, instance = identity if identity is not None else (None, None)
                        if (uid, instance) != (item["instrument_uid"], item["instance_no"]):
                            connection.execute(
                                """INSERT INTO sam3_object_assignment_events
                                (revision, object_index, previous_uid, previous_instance_no, instrument_uid, instance_no, reviewer)
                                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (revision, item["object_index"], item["instrument_uid"], item["instance_no"], uid, instance, reviewer),
                            )
                            connection.execute("UPDATE sam3_object_reviews SET instrument_uid=?, instance_no=? WHERE revision=? AND object_index=?", (uid, instance, revision, item["object_index"]))
                for index, decision in decisions.items():
                    connection.execute("INSERT INTO sam3_object_review_events(revision, object_index, dino_status, reviewer) VALUES (?, ?, ?, ?)", (revision, index, decision, reviewer))
                    connection.execute("UPDATE sam3_object_reviews SET dino_status=?, reviewer=?, reviewed_at=CURRENT_TIMESTAMP WHERE revision=? AND object_index=?", (decision, reviewer, revision, index))
            if write_masks and sample is not None:
                self._current(connection, sample, revision)
                capture_date = capture_date_from_source_path(sample.source_path, sample.session_id)
                database = session_database(
                    self.database_path,
                    sample.session_id,
                    self.database_paths,
                    capture_date,
                )
                root = capture_root(sample)
                originals = connection.execute(
                    """SELECT o.object_index, o.mask_path, o.mask_sha256
                    FROM sam3_prediction_objects o
                    JOIN sam3_object_reviews v ON v.object_index=o.object_index AND v.revision=?
                    WHERE o.session_id=? AND o.image_id=? AND o.prediction_id=? AND v.dino_status='accepted'""",
                    (revision, sample.session_id, sample.image_id, row["prediction_id"]),
                ).fetchall()
                targets = {mask_destination(database, sample, original["object_index"]) for original in originals}
                remove = self._mask_cleanup(connection, sample, database, root, targets)
                connection.execute(
                    """DELETE FROM sam3_approved_masks WHERE revision IN
                    (SELECT revision FROM sam3_review_revisions WHERE session_id=? AND image_id=?)""",
                    (sample.session_id, sample.image_id),
                )
                files = []
                by_index = {candidate.index: candidate for candidate in candidates}
                for original in originals:
                    target = mask_destination(database, sample, original["object_index"])
                    relative = target.relative_to(root).as_posix()
                    # 同じ画像名を別画像が持つ場合、Yesでも他画像の承認原本を壊さない。
                    owner = connection.execute(
                        """SELECT 1 FROM sam3_approved_masks m
                        JOIN sam3_review_revisions r USING(revision)
                        WHERE m.mask_path=? AND (r.session_id<>? OR r.image_id<>?) LIMIT 1""",
                        (relative, sample.session_id, sample.image_id),
                    ).fetchone()
                    if owner is not None:
                        raise CaptureError("同名のmaskを別画像が使用している。元画像名を一意にしてください")
                    candidate = by_index.get(original["object_index"])
                    if candidate is None or not isinstance(candidate.mask, ArrayMask):
                        raise CaptureError("確認用maskが不足している。画像を開き直してください")
                    files.append((candidate.mask.pixels, target))
                    connection.execute(
                        """INSERT INTO sam3_approved_masks VALUES (?, ?, ?, ?)
                        ON CONFLICT(revision, object_index) DO UPDATE SET
                        mask_path=excluded.mask_path, mask_sha256=excluded.mask_sha256""",
                        (revision, original["object_index"], relative, original["mask_sha256"]),
                    )
                assets.enter_context(publish_masks(files, overwrite=overwrite_masks, remove=remove))
                # 正式保存したPNGのhashを公開参照へ記録する。
                for _, target in files:
                    connection.execute("UPDATE sam3_approved_masks SET mask_sha256=? WHERE revision=? AND mask_path=?", (digest_file(target), revision, target.relative_to(root).as_posix()))
            if status == "revoked":
                self._reset(connection, revision, "approval_revoked")
            connection.execute(
                """UPDATE sam3_review_revisions SET annotation_json=?, status=?, approval_id=?,
                approval_json=?, updated_at=CURRENT_TIMESTAMP WHERE revision=?""",
                (self._json(annotation), status, approval["approval_id"] if approved and approval else None, self._json(approval) if approved and approval else None, revision),
            )
            connection.execute("INSERT INTO sam3_review_events(revision, status, annotation_json, approval_json) VALUES (?, ?, ?, ?)", (revision, status, self._json(annotation), self._json(approval) if approval else None))

    @staticmethod
    def _mask_cleanup(connection: sqlite3.Connection, sample: CaptureSample, database: Path, root: Path, targets: set[Path]) -> list[Path]:
        directory = (database.parent / "masks").resolve()
        identity = (sample.session_id, sample.image_id)
        owners: dict[Path, set[tuple[str, str]]] = {}
        for item in connection.execute(
            """SELECT m.mask_path, r.session_id, r.image_id
            FROM sam3_approved_masks m JOIN sam3_review_revisions r USING(revision)"""
        ):
            path = (root / item["mask_path"]).resolve()
            owners.setdefault(path, set()).add((item["session_id"], item["image_id"]))
        owned = {path for path, identities in owners.items() if identity in identities}
        if any(path.parent != directory or path.suffix.lower() != ".png" for path in owned | targets):
            raise CaptureError("承認用maskの台帳または保存先がセッションのmasks外である")

        # ファイル名ではなく台帳で所有を確定する。未登録の旧候補は推測で削除しない。
        expected = set(targets)
        for item in connection.execute(
            """SELECT DISTINCT p.object_index, r.source_path
            FROM sam3_review_revisions r JOIN sam3_prediction_objects p
            USING(session_id, image_id, prediction_id)
            WHERE r.session_id=? AND r.image_id=?""",
            identity,
        ):
            expected.add((directory / f"{Path(item['source_path']).stem}_{item['object_index']:03d}.png").resolve())
        affected = owned | targets
        if any(path.exists() and path not in owned for path in expected):
            raise CaptureError("既存maskの画像所有を台帳で確認できない。保存を中止しました")
        if any(owners.get(path, set()) - {identity} for path in affected):
            raise CaptureError("同名のmaskを別画像が使用している。元画像名を一意にしてください")

        protected = set()
        for item in connection.execute("SELECT mask_path FROM sam3_prediction_objects WHERE mask_path NOT LIKE 'memory:%'"):
            protected.add((root / item["mask_path"]).resolve())
        for item in connection.execute("SELECT path, depth_path FROM images"):
            protected.update((root / value).resolve() for value in item if value)
        for item in connection.execute("SELECT source_path FROM sam3_predictions"):
            protected.add((root / item["source_path"]).resolve())
        if affected & protected:
            raise CaptureError("承認用maskが推論原本または撮影原本と共有されている。保存を中止しました")
        return sorted(owned - targets)

    def objects(self, sample: CaptureSample, revision: str) -> list[dict[str, Any]]:
        """検証済み推論原本と現在の物体別判定を返す。

        Args:
            sample: 現行性を検証済みの撮影スナップショット。
            revision: 最新のレビュー版。

        Returns:
            元bbox、maskパス、器具対応、採否を持つ物体列。

        Raises:
            CaptureError: 旧版、原本不整合、または読込に失敗した場合。
        """
        with self._connection(sample=sample) as connection:
            row = self._current(connection, sample, revision)
            prediction_id = row["prediction_id"]
            rows = connection.execute(
                """SELECT objects.*, reviews.instrument_uid, reviews.instance_no,
                reviews.dino_status, reviews.reviewer, reviews.reviewed_at FROM sam3_prediction_objects AS objects
                JOIN sam3_object_reviews AS reviews ON reviews.object_index=objects.object_index
                WHERE reviews.revision=? AND objects.session_id=? AND objects.image_id=? AND objects.prediction_id=?
                ORDER BY objects.object_index""",
                (revision, sample.session_id, sample.image_id, prediction_id),
            ).fetchall()
        capture_date = capture_date_from_source_path(sample.source_path, sample.session_id)
        database = session_database(
            self.database_path,
            sample.session_id,
            self.database_paths,
            capture_date,
        )
        return [{**dict(item), "bbox": json.loads(item["bbox_json"]), "saved_mask_exists": mask_destination(database, sample, item["object_index"]).exists()} for item in rows]

    def decide(self, sample: CaptureSample, revision: str, object_index: int, *, status: str, reviewer: str) -> None:
        """元bbox・maskを確認した担当者の物体別採否を履歴付きで記録する。

        Args:
            sample: 現行性を検証済みの撮影スナップショット。
            revision: 最新レビュー版。
            object_index: 推論時に固定した物体番号。
            status: pending、accepted、rejectedのいずれか。
            reviewer: 確認担当者。

        Raises:
            CaptureError: 旧版、不明物体、不正採否、器具対応未確定、または原本不整合の場合。
        """
        if status not in DINO_STATUSES or not reviewer.strip() or isinstance(object_index, bool) or not isinstance(object_index, int):
            raise CaptureError("A valid object status, index and reviewer are required")
        objects = self.objects(sample, revision)
        selected = next((item for item in objects if item["object_index"] == object_index), None)
        if selected is None or (status == "accepted" and (selected["instrument_uid"] is None or selected["instance_no"] is None)):
            raise CaptureError("Unknown object or unresolved instrument assignment")
        if self.predictions.load(sample, selected["prediction_id"]) is None:
            raise CaptureError("確認用maskがない。runでSAM3推論を実行してください")
        with self._connection(sample=sample) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._current(connection, sample, revision)
            # 旧物体単位APIでも一括保存と同じ一対一の採用条件を守る。
            current = connection.execute("SELECT instrument_uid, instance_no FROM sam3_object_reviews WHERE revision=? AND object_index=?", (revision, object_index)).fetchone()
            if status == "accepted":
                if current is None or current["instrument_uid"] is None or current["instance_no"] is None:
                    raise CaptureError("器具未確定の物体はDINOに使えない")
                duplicate = connection.execute(
                    """SELECT 1 FROM sam3_object_reviews
                    WHERE revision=? AND object_index<>? AND dino_status='accepted' AND instrument_uid=? AND instance_no=?""",
                    (revision, object_index, current["instrument_uid"], current["instance_no"]),
                ).fetchone()
                if duplicate:
                    raise CaptureError("一つのタグに複数のDINO候補は採用できない")
            connection.execute("INSERT INTO sam3_object_review_events(revision, object_index, dino_status, reviewer) VALUES (?, ?, ?, ?)", (revision, object_index, status, reviewer.strip()))
            connection.execute("UPDATE sam3_object_reviews SET dino_status=?, reviewer=?, reviewed_at=CURRENT_TIMESTAMP WHERE revision=? AND object_index=?", (status, reviewer.strip(), revision, object_index))

    def reject_all(self, sample: CaptureSample, revision: str, *, reviewer: str) -> None:
        """画像承認やmask保存をせず、現行候補すべての不採用を一括記録する。

        Args:
            sample: 現行性を検証済みの撮影スナップショット。
            revision: 最新レビュー版。
            reviewer: 確認担当者。

        Raises:
            CaptureError: 担当者、レビュー版、または候補一覧が不正な場合。
        """
        reviewer_name = reviewer.strip()
        if not reviewer_name:
            raise CaptureError("A reviewer is required")
        with self._connection(sample=sample) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._current(connection, sample, revision)
            objects = connection.execute(
                "SELECT object_index FROM sam3_object_reviews WHERE revision=? ORDER BY object_index",
                (revision,),
            ).fetchall()
            if not objects:
                raise CaptureError("保存対象の候補を指定してください")
            for item in objects:
                connection.execute(
                    "INSERT INTO sam3_object_review_events(revision, object_index, dino_status, reviewer) VALUES (?, ?, 'rejected', ?)",
                    (revision, item["object_index"], reviewer_name),
                )
                connection.execute(
                    "UPDATE sam3_object_reviews SET dino_status='rejected', reviewer=?, reviewed_at=CURRENT_TIMESTAMP WHERE revision=? AND object_index=?",
                    (reviewer_name, revision, item["object_index"]),
                )

    def mask_png(self, sample: CaptureSample, revision: str, object_index: int) -> bytes:
        """原本maskを透明背景で可視化し、補正せずに重ね表示へ渡す。

        Args:
            sample: 原本を照合済みの現行撮影データ。
            revision: 現行レビュー版。
            object_index: 固定した推論候補番号。

        Returns:
            物体領域だけに色を付けたRGBA PNG。

        Raises:
            CaptureError: 対象が旧版、不明物体、または原本不整合の場合。
        """
        objects = self.objects(sample, revision)
        selected = next((item for item in objects if item["object_index"] == object_index), None)
        if selected is None:
            raise CaptureError("Unknown object")
        candidates = self.predictions.load(sample, selected["prediction_id"])
        candidate = next((item for item in candidates or () if item.index == object_index), None)
        if candidate is None or not isinstance(candidate.mask, ArrayMask):
            raise CaptureError("Missing original mask")
        image = Image.new("RGBA", (sample.width, sample.height), OVERLAY_COLOR)
        image.putalpha(Image.fromarray(candidate.mask.pixels.astype(np.uint8) * OVERLAY_ALPHA))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def progress(self) -> dict[str, dict[str, Any]]:
        """保存済みレビュー版の一覧確認状態を返す。

        Returns:
            レビュー版をキーとする確認状態。

        Raises:
            CaptureError: SQLまたはJSONの読み取りに失敗した場合。
        """
        # 共有ルートのときだけ各セッションへ分配し、版ごとの状態をまとめて返す。
        if self.database_paths is not None:
            scoped_states: dict[str, dict[str, Any]] = {}
            for database in self.database_paths:
                scoped_states.update(SqliteObjectReviewStore(database, self.timeout, self.predictions).progress())
            return scoped_states
        if self.database_path.is_dir():
            states: dict[str, dict[str, Any]] = {}
            for database in metadata_databases(self.database_path):
                states.update(SqliteObjectReviewStore(database, self.timeout, self.predictions).progress())
            return states
        with self._connection() as connection:
            # 起動後に撮影APIが追加した未処理セッションには、まだレビュー状態がない。
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='sam3_review_revisions'").fetchone():
                return {}
            return {row["revision"]: json.loads(row["progress_json"]) for row in connection.execute("SELECT revision, progress_json FROM sam3_review_revisions")}

    def set_progress(self, revision: str, state: dict[str, Any]) -> None:
        """一覧確認状態をSQLへ記録する。

        Args:
            revision: 現行レビュー版。
            state: 確認済み・空画像確認・例外理由。

        Raises:
            CaptureError: 現行レビューがないか、記録に失敗した場合。
        """
        with self._connection(revision=revision) as connection:
            cursor = connection.execute("UPDATE sam3_review_revisions SET progress_json=? WHERE revision=? AND is_current=1", (self._json(state), revision))
            if cursor.rowcount != 1:
                raise CaptureError("Object review is no longer current")

    def approved(self, revision: str, *, approval_id: str | None = None, scope: str | None = None) -> bool:
        """現行レビュー版の有効な画像承認を返す。

        Args:
            revision: 確認するレビュー版。
            approval_id: 指定時に一致を要求する承認ID。
            scope: yolo指定時はDINO専用承認を除外する。

        Returns:
            現行かつ画像承認済みの場合はTrue。

        Raises:
            CaptureError: SQL読み取りに失敗した場合。
        """
        with self._connection(revision=revision) as connection:
            row = connection.execute("SELECT status, approval_id, approval_json FROM sam3_review_revisions WHERE revision=? AND is_current=1", (revision,)).fetchone()
            approval_scope = json.loads(row["approval_json"] or "{}").get("scope") if row is not None else None
            return (
                row is not None
                and row["status"] in APPROVED_STATUSES
                and row["approval_id"] is not None
                and (approval_id is None or row["approval_id"] == approval_id)
                and (scope != "yolo" or approval_scope != "dino")
                and (scope != "dino" or approval_scope == "dino")
            )

    def object_review_state(self, revision: str) -> str:
        """DINO物体の判定進捗をYOLO承認と独立して返す。"""
        with self._connection(revision=revision) as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS total,
                SUM(CASE WHEN dino_status != 'pending' THEN 1 ELSE 0 END) AS decided
                FROM sam3_object_reviews WHERE revision=?""",
                (revision,),
            ).fetchone()
        total = row["total"]
        decided = row["decided"] or 0
        if total == 0 or decided == 0:
            return "未確認"
        if decided < total:
            return "判定途中"
        return "判定済み"

    def catalog_state(
        self,
        session_id: str,
        image_id: str,
        revision: str,
        source_path: str | None = None,
    ) -> dict[str, Any] | None:
        """選択一覧用に、指定セッション内の登録済みレビュー状態だけを返す。"""
        if self.database_paths is not None and source_path is None:
            raise CaptureError("対象DBを限定したレビュー確認にはsource_pathが必要である")
        capture_date = capture_date_from_source_path(source_path, session_id) if source_path is not None else None
        database = session_database(self.database_path, session_id, self.database_paths, capture_date)
        store = SqliteObjectReviewStore(database, self.timeout, self.predictions)
        with store._connection(refresh=False) as connection:
            has_review_schema = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sam3_review_revisions'"
            ).fetchone()
            if has_review_schema is None:
                return None
            row = connection.execute(
                """SELECT status, approval_id, approval_json, progress_json, is_current
                FROM sam3_review_revisions WHERE revision=? AND session_id=? AND image_id=?""",
                (revision, session_id, image_id),
            ).fetchone()
            if row is None:
                return None
            approval_scope = json.loads(row["approval_json"] or "{}").get("scope")
            is_approved = row["is_current"] == 1 and row["status"] in APPROVED_STATUSES and row["approval_id"] is not None
            object_counts = connection.execute(
                """SELECT COUNT(*) AS total,
                SUM(CASE WHEN dino_status != 'pending' THEN 1 ELSE 0 END) AS decided,
                SUM(CASE WHEN dino_status = 'accepted' THEN 1 ELSE 0 END) AS accepted
                FROM sam3_object_reviews WHERE revision=?""",
                (revision,),
            ).fetchone()
        total = object_counts["total"]
        decided = object_counts["decided"] or 0
        accepted = object_counts["accepted"] or 0
        if total == 0 or decided == 0:
            dino_state = "未確認"
        elif decided < total:
            dino_state = "判定途中"
        else:
            dino_state = "判定済み"
        return {
            "progress": json.loads(row["progress_json"]),
            "yolo_approved": is_approved and approval_scope != "dino",
            "dino_approved": is_approved and (approval_scope == "dino" or accepted > 0),
            "dino_state": dino_state,
        }

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)

    @staticmethod
    def _current(connection: sqlite3.Connection, sample: CaptureSample, revision: str) -> sqlite3.Row:
        """現行レビュー版と所属セッションの撮影SQLの両方を照合する。"""
        row = connection.execute("SELECT * FROM sam3_review_revisions WHERE revision=? AND is_current=1", (revision,)).fetchone()
        if not isinstance(row, sqlite3.Row) or (row["session_id"], row["image_id"], row["tag_revision"], row["source_sha256"], row["source_path"]) != (sample.session_id, sample.image_id, sample.tag_revision, sample.sha256, sample.source_path):
            raise CaptureError("Object review is no longer current")
        capture = connection.execute("SELECT * FROM images WHERE session_id=? AND image_id=?", (sample.session_id, sample.image_id)).fetchone()
        if capture is None or (capture["tag_status"], capture["tag_revision"], capture["sha256"], capture["path"], capture["width"], capture["height"]) != (
            "tagged",
            sample.tag_revision,
            sample.sha256,
            sample.source_path,
            sample.width,
            sample.height,
        ):
            raise CaptureError("Capture source is no longer current")
        return row

    @staticmethod
    def _reset(connection: sqlite3.Connection, revision: str, reason: str) -> None:
        """自動失効の理由も履歴に残し、旧採用を未確認へ戻す。"""
        connection.execute(
            """INSERT INTO sam3_object_review_events(revision, object_index, dino_status, reviewer)
            SELECT revision, object_index, 'pending', ? FROM sam3_object_reviews WHERE revision=? AND dino_status<>'pending'""",
            (reason, revision),
        )
        connection.execute("UPDATE sam3_object_reviews SET dino_status='pending', reviewer=NULL, reviewed_at=NULL WHERE revision=?", (revision,))

    @contextmanager
    def _connection(self, *, sample: CaptureSample | None = None, revision: str | None = None, refresh: bool = True) -> Iterator[sqlite3.Connection]:
        """SQLiteのロック待ちを制限し、不整合や保存失敗をドメイン例外へ変換する。

        Args:
            sample: セッションIDでDBを特定する画像。
            revision: 画像を持たない操作でDBを特定するレビュー版。
            refresh: 更新トランザクションで公開テーブルも同期するか。

        Yields:
            行名アクセスと外部キー制約が有効なセッションDB接続。

        Raises:
            CaptureError: 保存先が欠損、またはSQLite操作に失敗した場合。
        """
        # 画像を持つ要求はセッションID、承認・取消などの要求は保存済みレビュー版で解決する。
        database = self.database_path
        if sample is not None:
            capture_date = capture_date_from_source_path(sample.source_path, sample.session_id)
            database = session_database(database, sample.session_id, self.database_paths, capture_date)
        elif revision is not None:
            database = review_database(database, revision, self.timeout, self.database_paths)
        if not database.is_file():
            raise CaptureError(f"セッションDBが存在しない: {database}")
        try:
            with closing(sqlite3.connect(database, timeout=self.timeout)) as connection, connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys=ON")
                yield connection
                # 承認と専用テーブルの公開可否を同一commitで更新する。
                if refresh and connection.in_transaction:
                    refresh_handoff(connection)
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as error:
            raise CaptureError(f"物体レビュー保存先が不正である: {database}: {error}") from error
