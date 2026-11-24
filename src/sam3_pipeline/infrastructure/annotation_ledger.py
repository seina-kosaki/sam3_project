"""各セッションDBへ画像承認と公開履歴を保存する。

責務:
    承認台帳とYOLO公開状態の更新を担い、推論や注釈の編集は行わない。
処理の流れ:
    1. セッションIDから既存の撮影DBを特定する。
    2. 承認・取消・公開をトランザクションで記録する。
    3. DINO・YOLO専用テーブルの公開状態を同期する。
関連ファイル:
    annotation_store.py、session_database.py、handoff_store.py。
設計意図:
    従来のmanifest契約を維持しながら、撮影と結果の正本をセッション内に揃える。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sam3_pipeline.domain.capture import CaptureError
from sam3_pipeline.infrastructure.handoff_store import initialize_handoff, refresh_handoff
from sam3_pipeline.infrastructure.session_database import capture_date_from_source_path, session_database

SQLITE_TIMEOUT_SECONDS = 10.0


class SqliteAnnotationLedger:
    """画像承認のライフサイクルを所属セッションの撮影DBへ記録する。"""

    def __init__(
        self,
        capture_root: Path,
        timeout: float = SQLITE_TIMEOUT_SECONDS,
        database_paths: list[Path] | None = None,
    ) -> None:
        """共有データルートと有限のSQLiteロック待ちを保持する。

        Args:
            capture_root: 日付・セッション配下に撮影DBを持つ共有ルート。
            timeout: SQLiteのロック待ち秒数。
            database_paths: 指定時に接続可能なセッションDBを限定する一覧。
        """
        self.capture_root = capture_root.resolve()
        self.timeout = timeout
        self.database_paths = database_paths

    def upsert_annotation_revision(
        self,
        *,
        session_id: str,
        image_id: str,
        tag_revision: int,
        revision: str,
        status: str,
        annotation_path: str,
        annotation_sha256: str,
        source_path: str | None = None,
        approval_id: str | None = None,
        reviewer: str | None = None,
        approved_at: str | None = None,
    ) -> None:
        """画像レビュー・承認状態と注釈の所在を記録する。

        Args:
            session_id: 所属セッションID。
            image_id: 撮影画像ID。
            tag_revision: 撮影時のタグ版。
            revision: レビュー版。
            status: 確認・承認・取消などの状態。
            annotation_path: 作業ルート相対の注釈パス。
            annotation_sha256: 注釈ファイルのハッシュ。
            source_path: 撮影ルート相対の画像パス。指定時は日付もDB解決に使う。
            approval_id: 明示承認のID。
            reviewer: 確認担当者。
            approved_at: 承認日時。

        Raises:
            CaptureError: 対象セッションが欠損、または保存に失敗した場合。
        """
        with self._connect(session_id, source_path) as connection:
            if status == "inferred":
                connection.execute(
                    """UPDATE annotation_revisions SET status='revoked', updated_at=?
                    WHERE session_id=? AND image_id=? AND revision<>?
                    AND status IN ('inferred', 'reviewed', 'approved', 'exported')""",
                    (datetime.now(UTC).isoformat(), session_id, image_id, revision),
                )
            connection.execute(
                """INSERT INTO annotation_revisions(
                    session_id, image_id, tag_revision, revision, status, approval_id,
                    reviewer, approved_at, annotation_path, annotation_sha256, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, image_id, revision) DO UPDATE SET
                    tag_revision=excluded.tag_revision, status=excluded.status,
                    approval_id=excluded.approval_id, reviewer=excluded.reviewer,
                    approved_at=excluded.approved_at, annotation_path=excluded.annotation_path,
                    annotation_sha256=excluded.annotation_sha256, updated_at=excluded.updated_at""",
                (
                    session_id,
                    image_id,
                    tag_revision,
                    revision,
                    status,
                    approval_id,
                    reviewer,
                    approved_at,
                    annotation_path,
                    annotation_sha256,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def mark_annotation_exported(
        self,
        *,
        session_id: str,
        image_id: str,
        revision: str,
        manifest_path: str,
        source_path: str | None = None,
    ) -> None:
        """manifestの原子的な公開後にYOLOへの引き渡しを有効にする。

        Args:
            session_id: 所属セッションID。
            image_id: 公開画像ID。
            revision: 公開レビュー版。
            manifest_path: 公開済みmanifestの絶対パス。
            source_path: 撮影ルート相対の画像パス。指定時は日付もDB解決に使う。

        Raises:
            CaptureError: 承認が欠損、または保存に失敗した場合。
        """
        with self._connect(session_id, source_path) as connection:
            cursor = connection.execute(
                """UPDATE annotation_revisions
                SET status = 'exported', manifest_path = ?, exported_at = ?, updated_at = ?
                WHERE session_id = ? AND image_id = ? AND revision = ? AND status = 'approved'""",
                (
                    manifest_path,
                    datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                    session_id,
                    image_id,
                    revision,
                ),
            )
            if cursor.rowcount != 1:
                raise CaptureError(f"セッションDBに承認済み注釈がない: {session_id}/{image_id}")

    @contextmanager
    def _connect(self, session_id: str, source_path: str | None = None) -> Iterator[sqlite3.Connection]:
        """所属DBの承認履歴と公開状態を同一トランザクションで更新する。

        Args:
            session_id: 接続先のセッションID。
            source_path: 指定時に日付を含めて接続先を特定する撮影ルート相対パス。

        Yields:
            承認台帳を更新するSQLite接続。

        Raises:
            CaptureError: 対象DBが欠損、または更新に失敗した場合。
        """
        if self.database_paths is not None and source_path is None:
            raise CaptureError("対象DBを限定した台帳更新にはsource_pathが必要である")
        capture_date = capture_date_from_source_path(source_path, session_id) if source_path is not None else None
        database = session_database(self.capture_root, session_id, self.database_paths, capture_date)
        try:
            with closing(sqlite3.connect(database, timeout=self.timeout)) as connection, connection:
                connection.row_factory = sqlite3.Row
                connection.execute("BEGIN IMMEDIATE")
                initialize_handoff(connection)
                yield connection
                refresh_handoff(connection)
        except sqlite3.Error as error:
            raise CaptureError(f"セッション承認台帳の更新に失敗した: {database}: {error}") from error
