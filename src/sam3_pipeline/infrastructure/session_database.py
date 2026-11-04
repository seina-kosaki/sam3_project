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
from pathlib import Path

from sam3_pipeline.domain.capture import CaptureError

METADATA_NAME = "metadata.sqlite3"


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


def session_database(location: Path, session_id: str) -> Path:
    """セッションIDから一意な既存の保存先を返す。

    Args:
        location: 共有ルートまたは明示されたセッションDB。
        session_id: 撮影セッションID。

    Returns:
        対象セッションのDBパス。

    Raises:
        CaptureError: IDが不正、対象DBが欠損、または同名セッションが複数ある場合。
    """
    if not session_id or session_id in {".", ".."} or any(character in session_id for character in "/\\:"):
        raise CaptureError(f"セッションIDが不正である: {session_id}")
    if location.name == METADATA_NAME:
        return metadata_databases(location)[0]
    matches = [path for path in metadata_databases(location) if path.parent.name == session_id]
    if len(matches) != 1:
        raise CaptureError(f"セッションDBを一意に特定できない: {session_id}: {location}")
    return matches[0]


def review_database(location: Path, revision: str, timeout: float) -> Path:
    """保存済みレビュー版を所有するセッションDBを読み取り専用で探す。

    Args:
        location: 共有ルートまたは明示されたセッションDB。
        revision: レビュー版の識別子。
        timeout: SQLiteのロック待ち秒数。

    Returns:
        レビュー版を所有するDBパス。

    Raises:
        CaptureError: 版が欠損・重複、または探索中のDBが不正な場合。
    """
    paths = metadata_databases(location)
    if location.name == METADATA_NAME:
        return paths[0]
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
