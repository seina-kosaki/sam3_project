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
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import date as calendar_date
from pathlib import Path, PurePosixPath
from typing import Any

from sam3_pipeline.domain.capture import TAGGED, VISIBLE, CaptureError, CaptureSample, InstanceTag, TagPoint

METADATA_NAME = "metadata.sqlite3"
READ_ONLY_QUERY = "?mode=ro"
MIN_CAPTURE_PATH_PARTS = 4
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")


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

    def __init__(self, data_root: Path, timeout: float, targets: Any = None) -> None:
        """Bind the source root and injected SQLite timeout."""
        self.root = data_root.resolve()
        self.timeout = timeout
        self.targets = self._validate_targets(targets) if targets is not None else None

    @staticmethod
    def _validate_targets(value: Any) -> tuple[tuple[str, str | None], ...]:
        """Validate date and optional session conditions before database discovery."""
        if not isinstance(value, list) or not value:
            raise CaptureError("capture.json targets must contain at least one date condition")
        validated: list[tuple[str, str | None]] = []
        for index, target in enumerate(value):
            if not isinstance(target, dict) or set(target) - {"date", "session_id"}:
                raise CaptureError(f"capture.json targets[{index}] is invalid")
            capture_date = target.get("date")
            if not isinstance(capture_date, str) or DATE_PATTERN.fullmatch(capture_date) is None:
                raise CaptureError(f"capture.json targets[{index}].date must use YYYY-MM-DD")
            try:
                calendar_date.fromisoformat(capture_date)
            except ValueError as error:
                raise CaptureError(f"capture.json targets[{index}].date must be a real date") from error
            has_session_id = "session_id" in target
            session_id = target.get("session_id")
            if has_session_id and (
                not isinstance(session_id, str) or SESSION_ID_PATTERN.fullmatch(session_id) is None
            ):
                raise CaptureError(f"capture.json targets[{index}].session_id must be a safe session ID")
            validated.append((capture_date, session_id))
        return tuple(validated)

    def databases(self) -> list[Path]:
        """Return unique, safe session databases matching the optional target conditions."""
        if not self.root.is_dir():
            raise CaptureError(f"Capture root does not exist: {self.root}")
        discovered = self._discover_databases()
        if self.targets is None:
            return discovered

        selected: dict[Path, Path] = {}
        for capture_date, session_id in self.targets:
            matches = [
                database
                for database in discovered
                if database.parent.parent.name == capture_date
                and (session_id is None or database.parent.name == session_id)
            ]
            if not matches:
                scope = capture_date if session_id is None else f"{capture_date}/{session_id}"
                raise CaptureError(f"指定された対象にセッションDBがありません: {scope}")
            for database in matches:
                selected[database] = database
        if not selected:
            raise CaptureError("capture.json targets に一致するセッションDBがありません")
        return sorted(selected.values())

    def _discover_databases(self) -> list[Path]:
        """Find only real database files in safe date/session directories below the root."""
        databases: list[Path] = []
        for database in sorted(self.root.glob(f"*/*/{METADATA_NAME}")):
            date_directory = database.parent.parent
            session_directory = database.parent
            capture_date = date_directory.name
            session_id = session_directory.name
            safe_date = self._is_real_date(capture_date)
            safe_session = SESSION_ID_PATTERN.fullmatch(session_id) is not None
            has_real_directories = date_directory.is_dir() and session_directory.is_dir()
            has_no_symlinks = not date_directory.is_symlink() and not session_directory.is_symlink() and not database.is_symlink()
            resolved_database = database.resolve()
            stays_below_root = resolved_database.is_relative_to(self.root)
            is_safe_database = (
                safe_date
                and safe_session
                and has_real_directories
                and has_no_symlinks
                and stays_below_root
                and resolved_database.is_file()
            )
            if is_safe_database:
                databases.append(resolved_database)
        return databases

    @staticmethod
    def _is_real_date(value: str) -> bool:
        """Return whether a directory name is an actual ISO calendar date."""
        if DATE_PATTERN.fullmatch(value) is None:
            return False
        try:
            calendar_date.fromisoformat(value)
        except ValueError:
            return False
        return True

    def _require_databases(self) -> list[Path]:
        """Require discovered databases for operations that enumerate capture input."""
        databases = self.databases()
        if not databases:
            raise CaptureError(f"No session bundles found: {self.root}")
        return databases

    def _database_for_image(self, source_path: str, session_id: str) -> tuple[Path | None, PurePosixPath]:
        """Resolve an image only through the already discovered database allow-list."""
        relative = PurePosixPath(source_path)
        is_unsafe_path = (
            relative.is_absolute()
            or len(relative.parts) < MIN_CAPTURE_PATH_PARTS
            or ".." in relative.parts
            or "\\" in source_path
            or relative.as_posix() != source_path
        )
        if is_unsafe_path:
            raise CaptureError(f"Invalid capture-relative path: {source_path}")
        capture_date, path_session = relative.parts[:2]
        if path_session != session_id:
            return None, relative
        database = next(
            (
                candidate
                for candidate in self.databases()
                if candidate.parent.name == path_session and candidate.parent.parent.name == capture_date
            ),
            None,
        )
        if database is None:
            return None, relative
        return database, relative

    def _image_file(self, relative: PurePosixPath, database: Path) -> Path:
        """Resolve a database image path and require it to remain in its session."""
        expected_prefix = (database.parent.parent.name, database.parent.name)
        has_session_prefix = relative.parts[:2] == expected_prefix
        safe_relative = (
            not relative.is_absolute()
            and len(relative.parts) >= MIN_CAPTURE_PATH_PARTS
            and ".." not in relative.parts
            and "\\" not in relative.as_posix()
        )
        if not safe_relative or not has_session_prefix:
            raise CaptureError(f"Image path does not belong to session database: {relative}")
        image_path = (self.root / Path(*relative.parts)).resolve()
        session_root = database.parent.resolve()
        if not image_path.is_relative_to(session_root):
            raise CaptureError(f"Unsafe image path: {relative}")
        return image_path

    def iter_samples(self) -> Iterator[CaptureSample]:
        """各セッションDBの読み取りスナップショットから画像とタグを取得する。

        Yields:
            画像原本を検証したタグ付き撮影スナップショット。

        Raises:
            CaptureError: 入力ルートやセッションDBがない、または撮影データが不正な場合。
        """
        databases = self._require_databases()
        for database in databases:
            try:
                # 撮影中の変更を混在させず、セッション内の画像・宣言・タグを一貫して読む。
                with closing(sqlite3.connect(database.as_uri() + READ_ONLY_QUERY, uri=True, timeout=self.timeout)) as connection:
                    connection.row_factory = sqlite3.Row
                    connection.execute("BEGIN")
                    rows = connection.execute("SELECT * FROM images WHERE tag_status = ? ORDER BY image_id", (TAGGED,)).fetchall()
                    samples = [self._sample(connection, row, database) for row in rows]
                yield from samples
            except (sqlite3.Error, OSError, ValueError, KeyError, IndexError, TypeError, CaptureError) as error:
                raise CaptureError(f"撮影セッションDBが不正である: {database}: {error}") from error

    def catalog(self) -> list[dict[str, Any]]:
        """画像内容を読まず、選択画面に必要な撮影SQL項目だけを返す。"""
        databases = self._require_databases()
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
                    if row["session_id"] != database.parent.name:
                        raise CaptureError(f"Image session_id does not match its database: {database}")
                    relative = PurePosixPath(row["path"])
                    image_path = self._image_file(relative, database)
                    if not image_path.is_file():
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
        database, _relative = self._database_for_image(source_path, session_id)
        if database is None:
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
                image_path = self._image_file(PurePosixPath(row["path"]), database)
                if not image_path.is_file():
                    return None
                return self._sample(connection, row, database)
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
        database, relative = self._database_for_image(sample.source_path, sample.session_id)
        if database is None:
            return False
        image_path = self._image_file(relative, database)
        if not image_path.is_file():
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

    def _sample(self, connection: sqlite3.Connection, row: sqlite3.Row, database: Path) -> CaptureSample:
        """Resolve capture-root-relative images and preserve declaration name snapshots."""
        if row["session_id"] != database.parent.name:
            raise CaptureError(f"Image session_id does not match its database: {database}")
        image_path = self._image_file(PurePosixPath(row["path"]), database)
        if not image_path.is_file():
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
