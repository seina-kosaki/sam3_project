"""各セッションの撮影SQLiteを読み取り、タグ付き画像の入力を提供する。

責務:
    セッションDBの探索・画像とタグの取得・現行性照合を担う。
    撮影DBの更新、旧中央DBの参照、推論結果の保存は行わない。
処理の流れ:
    1. 共有データルートから日付・セッション配下のDBを探索する。
    2. 読み取り専用で画像・宣言・タグを取得する。
    3. ルート基準の画像パスとハッシュを検証し、入力スナップショットを返す。
関連ファイル:
    capture_workflow.py、domain/capture.py、infrastructure/annotation_store.py。
設計意図:
    標準SQLiteで撮影APIへの実行時依存を避け、保存側と同じセッション境界で参照する。

Note:
    taggedのRGB画像だけを扱う。画像パスはDBの所在ではなく共有データルート基準である。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Any

from sam3_pipeline.domain.capture import TAGGED, VISIBLE, CaptureError, CaptureSample, InstanceTag, TagPoint

METADATA_NAME = "metadata.sqlite3"
READ_ONLY_QUERY = "?mode=ro"
MIN_CAPTURE_PATH_PARTS = 4


def _point(raw: str | None) -> TagPoint | None:
    """Parse optional points using the capture API's normalized coordinate schema."""
    if not raw:
        return None
    data = json.loads(raw)
    return TagPoint(float(data["x"]), float(data["y"]), data.get("status", VISIBLE))


class SqliteCaptureSource:
    """SQLiteセッションを撮影SQLを正本とする共通入力契約へ変換する。

    標準ライブラリSQLiteを使い、撮影サービスへの実行時依存を避ける。

    Args:
        data_root: 撮影サービスのIC_DATA_DIR。
        timeout: SQLite読み取りタイムアウト秒数。

    Note:
        taggedのRGB画像だけを選択する。DBの変更は行わない。
    """

    def __init__(self, data_root: Path, timeout: float) -> None:
        """Bind the source root and injected SQLite timeout."""
        self.root = data_root.resolve()
        self.timeout = timeout

    def iter_samples(self) -> Iterator[CaptureSample]:
        """各セッションDBの読み取りスナップショットから画像とタグを取得する。

        Yields:
            画像原本を検証したタグ付き撮影スナップショット。

        Raises:
            CaptureError: 入力ルートやセッションDBがない、または撮影データが不正な場合。
        """
        if not self.root.is_dir():
            raise CaptureError(f"Capture root does not exist: {self.root}")
        # 旧DBが残っていても、追加・更新されたセッションDBを唯一の入力とする。
        databases = sorted(self.root.glob(f"*/*/{METADATA_NAME}"))
        if not databases:
            raise CaptureError(f"No session bundles found: {self.root}")
        for database in databases:
            try:
                # 撮影中の変更を混在させず、セッション内の画像・宣言・タグを一貫して読む。
                with closing(sqlite3.connect(database.as_uri() + READ_ONLY_QUERY, uri=True, timeout=self.timeout)) as connection:
                    connection.row_factory = sqlite3.Row
                    connection.execute("BEGIN")
                    rows = connection.execute("SELECT * FROM images WHERE tag_status = ? ORDER BY image_id", (TAGGED,)).fetchall()
                    samples = [self._sample(connection, row) for row in rows]
                yield from samples
            except (sqlite3.Error, OSError, ValueError, KeyError, IndexError, TypeError, CaptureError) as error:
                raise CaptureError(f"撮影セッションDBが不正である: {database}: {error}") from error

    def catalog(self) -> list[dict[str, Any]]:
        """画像内容を読まず、選択画面に必要な撮影SQL項目だけを返す。"""
        if not self.root.is_dir():
            raise CaptureError(f"Capture root does not exist: {self.root}")
        databases = sorted(self.root.glob(f"*/*/{METADATA_NAME}"))
        if not databases:
            raise CaptureError(f"No session bundles found: {self.root}")
        entries = []
        try:
            for database in databases:
                with closing(sqlite3.connect(database.as_uri() + READ_ONLY_QUERY, uri=True, timeout=self.timeout)) as connection:
                    connection.row_factory = sqlite3.Row
                    rows = connection.execute(
                        "SELECT session_id, image_id, declaration_id, arrangement_no, path, sha256, width, height, tag_revision, tag_status FROM images WHERE tag_status = ? ORDER BY image_id",
                        (TAGGED,),
                    ).fetchall()
                for row in rows:
                    image_path = (self.root / row["path"]).resolve()
                    if not image_path.is_relative_to(self.root) or not image_path.is_file():
                        raise CaptureError(f"Missing or unsafe image path: {row['path']}")
                    entries.append(
                        {
                            "session_id": row["session_id"],
                            "image_id": row["image_id"],
                            "capture_date": database.parent.parent.name,
                            "source_path": row["path"],
                            "sha256": row["sha256"],
                            "declaration_id": row["declaration_id"],
                            "arrangement_no": row["arrangement_no"],
                            "width": row["width"],
                            "height": row["height"],
                            "tag_revision": row["tag_revision"],
                        }
                    )
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as error:
            raise CaptureError(f"撮影セッションDBが不正である: {database}: {error}") from error
        return entries

    def current_sample(self, session_id: str, image_id: str, source_path: str) -> CaptureSample | None:
        """照合対象のSQL行を読み、画像ファイルまで現行性を検証する。

        Args:
            session_id: 撮影セッションID。
            image_id: SQL上の画像ID。
            source_path: SQLに記録されたdataルート相対パス。

        Returns:
            SQL上で同じパスを持つtagged画像。削除済み等の場合はNone。

        Raises:
            CaptureError: パス、DBレコード、画像内容が不正である場合。
        """
        relative = PurePosixPath(source_path)
        if relative.is_absolute() or len(relative.parts) < MIN_CAPTURE_PATH_PARTS or ".." in relative.parts or "\\" in source_path or relative.as_posix() != source_path:
            raise CaptureError(f"Invalid capture-relative path: {source_path}")
        database = self._database_path(relative)
        if not database.is_relative_to(self.root):
            raise CaptureError(f"Unsafe metadata path: {database}")
        if not database.is_file():
            return None
        try:
            with closing(sqlite3.connect(database.as_uri() + READ_ONLY_QUERY, uri=True, timeout=self.timeout)) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute(
                    "SELECT * FROM images WHERE session_id = ? AND image_id = ?",
                    (session_id, image_id),
                ).fetchone()
                if row is None or row["tag_status"] != TAGGED or row["path"] != source_path:
                    return None
                image_path = (self.root / row["path"]).resolve()
                if not image_path.is_relative_to(self.root):
                    raise CaptureError(f"Unsafe image path: {row['path']}")
                if not image_path.is_file():
                    return None
                return self._sample(connection, row)
        except (sqlite3.Error, OSError, ValueError, KeyError, IndexError, TypeError) as error:
            raise CaptureError(f"撮影セッションDBが不正である: {database}: {error}") from error

    def is_current(self, sample: CaptureSample) -> bool:
        """現行性の一覧確認ではSQLの版情報だけを照合し、画像全体の再ハッシュを避ける。

        Args:
            sample: 過去に取得した撮影スナップショット。

        Returns:
            同じtagged画像・タグ版と元画像ファイルが現存する場合はTrue。

        Raises:
            CaptureError: 所属セッションDBが破損、またはスキーマが不正な場合。
        """
        relative = PurePosixPath(sample.source_path)
        if relative.is_absolute() or len(relative.parts) < MIN_CAPTURE_PATH_PARTS or ".." in relative.parts or "\\" in sample.source_path or relative.as_posix() != sample.source_path:
            return False
        database = self._database_path(relative)
        if not database.is_relative_to(self.root):
            return False
        image_path = (self.root / Path(*relative.parts)).resolve()
        if not database.is_file() or not image_path.is_relative_to(self.root) or not image_path.is_file():
            return False
        try:
            with closing(sqlite3.connect(database.as_uri() + READ_ONLY_QUERY, uri=True, timeout=self.timeout)) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute(
                    """SELECT declaration_id, arrangement_no, path, sha256, width, height, tag_revision, tag_status
                    FROM images WHERE session_id = ? AND image_id = ?""",
                    (sample.session_id, sample.image_id),
                ).fetchone()
            return row is not None and (
                row["tag_status"] == TAGGED
                and row["declaration_id"] == sample.declaration_id
                and row["arrangement_no"] == sample.arrangement_no
                and row["path"] == sample.source_path
                and row["sha256"] == sample.sha256
                and row["width"] == sample.width
                and row["height"] == sample.height
                and row["tag_revision"] == sample.tag_revision
            )
        except (sqlite3.Error, OSError, ValueError, KeyError, IndexError, TypeError) as error:
            raise CaptureError(f"撮影セッションDBが不正である: {database}: {error}") from error

    def _sample(self, connection: sqlite3.Connection, row: sqlite3.Row) -> CaptureSample:
        """Resolve capture-root-relative images and preserve declaration name snapshots."""
        image_path = (self.root / row["path"]).resolve()
        if not image_path.is_relative_to(self.root) or not image_path.is_file():
            raise CaptureError(f"Missing or unsafe image path: {row['path']}")
        digest = self._digest(image_path)
        if row["sha256"] and row["sha256"] != digest:
            raise CaptureError(f"Image hash mismatch: {image_path}")
        declaration = connection.execute("SELECT items_json FROM declarations WHERE declaration_id = ?", (row["declaration_id"],)).fetchone()
        if declaration is None:
            raise CaptureError(f"Missing declaration: {row['declaration_id']}")
        names = {item["instrument_uid"]: item["instrument_name"] for item in json.loads(declaration["items_json"])}
        hints = connection.execute("SELECT * FROM hints WHERE image_id = ? ORDER BY instrument_uid, instance_no", (row["image_id"],)).fetchall()
        return CaptureSample(
            session_id=row["session_id"],
            image_id=row["image_id"],
            declaration_id=row["declaration_id"],
            arrangement_no=row["arrangement_no"],
            image_path=image_path,
            sha256=digest,
            width=row["width"],
            height=row["height"],
            tag_revision=row["tag_revision"],
            tags=tuple(self._tag(hint, names) for hint in hints),
            instruments=tuple(sorted(names.items())),
            source_path=row["path"],
        )

    def _database_path(self, source_path: PurePosixPath) -> Path:
        """画像のルート相対パスから所属セッションのDBを特定する。

        Args:
            source_path: 日付とセッションを先頭に持つ検証済みの画像相対パス。

        Returns:
            所属セッションのmetadata.sqlite3の絶対パス。
        """
        return self.root.joinpath(*source_path.parts[:2], METADATA_NAME).resolve()

    @staticmethod
    def _digest(path: Path) -> str:
        """Hash the image without retaining its full contents in memory."""
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def _tag(row: sqlite3.Row, names: dict[str, Any]) -> InstanceTag:
        """Require every tagged UID to exist in the original declaration."""
        uid = row["instrument_uid"]
        if uid not in names:
            raise CaptureError(f"Undeclared instrument: {uid}")
        return InstanceTag(uid, str(names[uid]), row["instance_no"], bool(row["visible"]), row["hint_source"], _point(row["center_json"]), _point(row["end_a_json"]), _point(row["end_b_json"]), bool(row["occluded"]))
