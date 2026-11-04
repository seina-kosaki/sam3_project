"""ディレクトリおよびファイル構造をツリー形式で出力・保存するモジュール.

このモジュールは、指定されたパスのディレクトリ構造を再帰的に走査し、
アスキーアートのツリー形式で表示およびテキストファイルへ保存する機能を提供します。
.gitignore 形式のパターンマッチングによる除外機能に対応しています。
"""

from collections.abc import Sequence
from pathlib import Path
from typing import List, Union

import pathspec


def print_dir_tree(
    startpath: str | Path,
    max_depth: int = 3,
    include_files: bool = True,
    ignore_patterns: Sequence[str] = (
        ".*",
        "__pycache__",
        "*.pyc",
        "venv",
        ".venv",
        "node_modules",
        "build",
        "dist",
    ),
    save_to_file: bool = False,
    output_filename: str = "dir_tree.txt",
) -> str:
    """ディレクトリおよびファイルの階層構造をツリー形式で表示・保存します。

    Args:
        startpath (Union[str, Path]): 走査対象のベースディレクトリパス。
        max_depth (int, optional): 表示する最大深度（1以上）。デフォルトは 3。
        include_files (bool, optional): ファイルも表示対象に含めるかどうか。デフォルトは True。
        ignore_patterns (Sequence[str], optional): 除外するファイル/ディレクトリのパターンリスト
            (.gitignore 形式)。デフォルトは主要な隠しファイルやキャッシュ等。
        save_to_file (bool, optional): 結果をテキストファイルに保存するかどうか。デフォルトは False。
        output_filename (str, optional): 保存するファイル名。デフォルトは 'dir_tree.txt'。

    Returns:
        str: 生成されたツリー構造の文字列テキスト。

    Raises:
        ValueError: startpath が存在しない、またはディレクトリでない場合。
    """
    base_path: Path = Path(startpath).resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"指定されたパスが存在しないかディレクトリではありません: {startpath}")

    # pathspec を用いて GitIgnore 形式のパターンマッチャーを作成
    spec: pathspec.GitIgnoreSpec = pathspec.GitIgnoreSpec.from_lines("gitwildmatch", ignore_patterns)

    output_lines: list[str] = [f"{base_path.name}/"]

    def _should_ignore(path: Path) -> bool:
        """指定されたパスが除外パターンに合致するか判定します。

        Args:
            path (Path): チェック対象の Path オブジェクト。

        Returns:
            bool: 除外対象であれば True、それ以外は False。
        """
        # ルートからの相対パスでチェック
        try:
            rel_path: Path = path.relative_to(base_path)
            # ディレクトリの場合は末尾に / を付けて比較
            rel_str: str = f"{rel_path}/" if path.is_dir() else str(rel_path)
            return spec.match_file(rel_str) or spec.match_file(path.name)
        except ValueError:
            return False

    def _build_tree(directory: Path, prefix: str = "", current_depth: int = 1) -> None:
        """再帰的にディレクトリを走査し、ツリー表現を生成します。

        Args:
            directory (Path): 走査対象のディレクトリ。
            prefix (str, optional): ツリー描画用のインデント接頭辞。デフォルトは ""。
            current_depth (int, optional): 現在の階層深さ。デフォルトは 1。
        """
        if current_depth > max_depth:
            return

        try:
            entries: list[Path] = list(directory.iterdir())
        except PermissionError:
            output_lines.append(f"{prefix}└── [アクセス拒否]")
            return

        # 除外パターンのフィルタリング
        filtered_entries: list[Path] = [e for e in entries if not _should_ignore(e)]

        # ファイルを含めない場合はディレクトリのみに絞り込み
        if not include_files:
            filtered_entries = [e for e in filtered_entries if e.is_dir()]

        # ディレクトリを優先し、名前のアルファベット順（大文字小文字を区別しない）でソート
        sorted_entries: list[Path] = sorted(
            filtered_entries,
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )

        entries_count: int = len(sorted_entries)
        for index, entry in enumerate(sorted_entries):
            is_last: bool = index == entries_count - 1
            connector: str = "└── " if is_last else "├── "
            display_name: str = f"{entry.name}/" if entry.is_dir() else entry.name

            output_lines.append(f"{prefix}{connector}{display_name}")

            if entry.is_dir():
                new_prefix: str = prefix + ("    " if is_last else "│   ")
                _build_tree(entry, new_prefix, current_depth + 1)

    _build_tree(base_path)
    tree_text: str = "\n".join(output_lines)

    print(tree_text)

    if save_to_file:
        output_file_path: Path = Path(output_filename)
        output_file_path.write_text(tree_text, encoding="utf-8")
        print(f"\n[情報] 実行結果を '{output_file_path.resolve()}' に保存しました。")

    return tree_text


if __name__ == "__main__":
    # 使用例: 現在のプロジェクトディレクトリ構造を走査
    target_directory: str = "./src"

    # カスタムの除外パターン（.gitignore のように記述可能）
    custom_ignores: list[str] = [
        ".git/",
        "__pycache__/",
        "*.pyc",
        ".venv/",
        "node_modules/",
        "*.egg-info/",
        "temp_data/",  # 特定のフォルダを潜らないように指定
        "hospitools/",
        "labeled_surgical_tools/",
    ]

    # パスの存在確認後実行（サンプル用のダミー実行）
    target_path: Path = Path(target_directory)
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "main.py").touch()
        (target_path / "utils").mkdir(exist_ok=True)
        (target_path / "utils" / "helper.py").touch()

    print_dir_tree(
        startpath=target_directory,
        max_depth=5,
        include_files=True,  # .py などのファイルも出力
        ignore_patterns=custom_ignores,  # 無視するパターンを指定
        save_to_file=True,
    )
