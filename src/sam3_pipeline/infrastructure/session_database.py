"""撮影セッションに属する既存SQLiteの所在を解決する。

責務:
    保存先の探索・セッションとレビュー版の照合を担い、DBの新規作成は行わない。
処理の流れ:
    1. 共有ルートから日付・セッション配下のDBを列挙する。
    2. セッションIDまたは保存済みレビュー版で所有DBを特定する。
    3. 欠損・重複・ルート外参照を拒否してパスを返す。
関連ファイル:
    prediction_store.py、object_review_store.py、annotation_ledger.py。
設計意図:
    三つの保存処理で同じDB境界を使い、旧単一DBへの暗黙の退避を防ぐ。
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date as calendar_date
from pathlib import Path, PurePosixPath

from sam3_pipeline.domain.capture import CaptureError

METADATA_NAME = "metadata.sqlite3"
MIN_CAPTURE_PATH_PARTS = 4


def metadata_databases(location: Path) -> list[Path]:
    """共有ルート、または明示された単一セッションDBを検証する。

    Args:
        location: 共有ルートまたは内部ストアに直接指定するセッションDB。

    Returns:
        パス順に並べた既存DB。

    Raises:
        CaptureError: DBがない、または共有ルート外を指す場合。
    """
    # 単一ストアは従来のパス注入を維持し、CLIからは必ず共有ルートを渡す。
    if location.name == METADATA_NAME:
        paths = [location] if location.is_file() else []
    else:
        paths = sorted(location.glob(f"*/*/{METADATA_NAME}"))
        if any(not path.resolve().is_relative_to(location.resolve()) for path in paths):
            raise CaptureError(f"共有ルート外のセッションDBである: {location}")
    if not paths:
        raise CaptureError(f"セッションDBが存在しない: {location}")
    return paths


def capture_date_from_source_path(source_path: str, session_id: str) -> str:
    """Validate a capture-relative image path and return its date directory."""
    relative = PurePosixPath(source_path)
    is_unsafe_path = (
        relative.is_absolute()
        or len(relative.parts) < MIN_CAPTURE_PATH_PARTS
        or ".." in relative.parts
        or "\\" in source_path
        or relative.as_posix() != source_path
        or relative.parts[1] != session_id
    )
    if is_unsafe_path:
        raise CaptureError(f"撮影画像パスが対象セッションと一致しない: {source_path}")
    capture_date = relative.parts[0]
    try:
        parsed_date = calendar_date.fromisoformat(capture_date)
    except ValueError as error:
        raise CaptureError(f"撮影画像パスの日付が不正である: {source_path}") from error
    if parsed_date.isoformat() != capture_date:
        raise CaptureError(f"撮影画像パスの日付が不正である: {source_path}")
    return capture_date


def session_database(
    location: Path,
    session_id: str,
    database_paths: list[Path] | None = None,
    capture_date: str | None = None,
) -> Path:
    """セッションIDから一意な既存の保存先を返す。

    Args:
        location: 共有ルートまたは明示されたセッションDB。
        session_id: 撮影セッションID。
        database_paths: 指定時に探索対象を限定する既存DB一覧。
        capture_date: 指定時に日付も含めてDBを特定する。

    Returns:
        対象セッションのDBパス。

    Raises:
        CaptureError: IDが不正、対象DBが欠損、または同名セッションが複数ある場合。
    """
    if not session_id or session_id in {".", ".."} or any(character in session_id for character in "/\\:"):
        raise CaptureError(f"セッションIDが不正である: {session_id}")
    if location.name == METADATA_NAME:
        database = location.resolve()
        if not database.is_file():
            raise CaptureError(f"セッションDBが存在しない: {location}")
        if database.parent.name != session_id:
            raise CaptureError(f"明示されたセッションDBが対象と一致しない: {session_id}: {location}")
        if capture_date is not None and database.parent.parent.name != capture_date:
            raise CaptureError(f"明示されたセッションDBの日付が対象と一致しない: {capture_date}: {location}")
        if database_paths is not None:
            root = database.parent.parent.parent
            allowed_paths = _validate_database_allowlist(root, database_paths)
            if database not in allowed_paths:
                raise CaptureError(f"対象外のセッションDBである: {location}")
        return database
    if database_paths is not None:
        paths = _validate_database_allowlist(location, database_paths)
    else:
        paths = metadata_databases(location)
    matches = [
        path
        for path in paths
        if path.parent.name == session_id and (capture_date is None or path.parent.parent.name == capture_date)
    ]
    if len(matches) != 1:
        raise CaptureError(f"セッションDBを一意に特定できない: {session_id}: {location}")
    return matches[0]


def review_database(
    location: Path,
    revision: str,
    timeout: float,
    database_paths: list[Path] | None = None,
) -> Path:
    """保存済みレビュー版を所有するセッションDBを読み取り専用で探す。

    Args:
        location: 共有ルートまたは明示されたセッションDB。
        revision: レビュー版の識別子。
        timeout: SQLiteのロック待ち秒数。
        database_paths: 指定時に探索対象を限定する既存DB一覧。

    Returns:
        レビュー版を所有するDBパス。

    Raises:
        CaptureError: 版が欠損・重複、または探索中のDBが不正な場合。
    """
    if location.name == METADATA_NAME:
        database = location.resolve()
        if not database.is_file():
            raise CaptureError(f"セッションDBが存在しない: {location}")
        if database_paths is not None:
            root = database.parent.parent.parent
            allowed_paths = _validate_database_allowlist(root, database_paths)
            if database not in allowed_paths:
                raise CaptureError(f"対象外のセッションDBである: {location}")
        return database
    paths = _validate_database_allowlist(location, database_paths) if database_paths is not None else metadata_databases(location)
    matches: list[Path] = []
    # 版の保存先を別台帳へ複製せず、各セッションDBの記録から復元する。
    for path in paths:
        try:
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)) as connection:
                exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='sam3_review_revisions'").fetchone()
                if exists and connection.execute("SELECT 1 FROM sam3_review_revisions WHERE revision=?", (revision,)).fetchone():
                    matches.append(path)
        except sqlite3.Error as error:
            raise CaptureError(f"レビュー版の探索に失敗した: {path}: {error}") from error
    if len(matches) != 1:
        raise CaptureError(f"レビュー版のセッションDBを一意に特定できない: {revision}")
    return matches[0]


def _validate_database_allowlist(location: Path, database_paths: list[Path]) -> list[Path]:
    """Keep explicitly scoped database lists below their configured capture root."""
    root = location.resolve()
    existing_paths = [database for database in database_paths if database.is_file()]
    has_unsafe_path = any(
        database.name != METADATA_NAME
        or database.is_symlink()
        or database.parent.is_symlink()
        or database.parent.parent.is_symlink()
        for database in existing_paths
    )
    if has_unsafe_path:
        raise CaptureError(f"対象セッションDBのパスが安全ではない: {location}")
    paths = sorted({database.resolve() for database in existing_paths})
    if any(not path.is_relative_to(root) for path in paths):
        raise CaptureError(f"共有ルート外のセッションDBである: {location}")
    if not paths:
        raise CaptureError(f"セッションDBが存在しない: {location}")
    return paths
