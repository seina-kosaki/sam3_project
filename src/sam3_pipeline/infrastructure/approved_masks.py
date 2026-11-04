"""承認用maskをセッション直下へ配置し、通常の保存失敗時に元へ戻す。

責務:
    メモリー上の推論maskの正式PNG保存と上書き確認を扱い、採否は判断しない。
処理の流れ:
    1. 撮影元に対応するセッションの出力先を検証する。
    2. 全PNGを一時領域へ準備し、既存ファイルを退避して置換する。
    3. 呼び出し元のSQL確定が失敗した場合は退避ファイルを復元する。
関連ファイル:
    object_review_store.py、prediction_store.py、handoff_store.py。
設計意図:
    承認前はPNGを保存せず、利用者が明示した場合だけ既存maskを上書きする。

Note:
    呼び出し元は同じセッションDBの書込ロックを保持する。プロセス強制終了に対する
    ファイルとSQLiteの分散トランザクションではなく、取得側でもhashを検証する。
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from sam3_pipeline.domain.capture import CaptureError, CaptureSample

MASK_DIRECTORY = "masks"
MASK_FOREGROUND = 255


def mask_destination(database: Path, sample: CaptureSample, index: int) -> Path:
    """所属セッション内の承認用maskパスを返す。

    Args:
        database: 所属セッションDB。
        sample: 撮影元画像。
        index: 固定した物体番号。

    Returns:
        画像名と物体番号を持つPNGの絶対パス。

    Raises:
        CaptureError: 保存先がセッション外を指す場合。
    """
    path = (database.parent / MASK_DIRECTORY / f"{sample.image_path.stem}_{index:03d}.png").resolve()
    if not path.is_relative_to(database.parent.resolve()):
        raise CaptureError("承認用maskの保存先がセッション外である")
    return path


def capture_root(sample: CaptureSample) -> Path:
    """撮影SQLの相対参照と絶対パスから共有ルートを復元する。

    Args:
        sample: 原本照合済みの撮影画像。

    Returns:
        共有データルート。

    Raises:
        CaptureError: 撮影元の相対参照が不正な場合。
    """
    relative = PurePosixPath(sample.source_path)
    if not sample.source_path or relative.is_absolute() or ".." in relative.parts or "\\" in sample.source_path:
        raise CaptureError("撮影元の相対パスが不正である")
    root = sample.image_path.resolve()
    for _ in relative.parts:
        root = root.parent
    if (root / Path(*relative.parts)).resolve() != sample.image_path.resolve():
        raise CaptureError("撮影元のパスを共有ルートから解決できない")
    return root


@contextmanager
def publish_masks(files: list[tuple[np.ndarray, Path]], *, overwrite: bool, remove: list[Path] | None = None) -> Iterator[None]:
    """推論maskを初めてPNGへ保存し、後続SQLの失敗時には復元する。

    Args:
        files: メモリー上の二値maskと正式保存先の組。
        overwrite: 既存maskの上書きに利用者が同意したか。
        remove: 呼び出し元が対象画像の所有と原本非共有を検証した不要mask。

    Yields:
        承認用PNGの配置が完了した状態。

    Raises:
        CaptureError: 上書き未同意、原本改変、配置または復元に失敗した場合。
    """
    remove = remove or []
    if not files and not remove:
        yield
        return
    destination = files[0][1].parent if files else remove[0].parent
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mask-save-", dir=destination) as temporary:
        staging = Path(temporary)
        changed: list[tuple[Path, Path | None]] = []
        # 全原本を先に準備し、不正な途中ファイルで既存の承認maskを壊さない。
        for index, (pixels, target) in enumerate(files):
            if target.exists() and not overwrite:
                raise CaptureError("保存済みmaskがある。上書きするか確認してください")
            Image.fromarray(pixels.astype(np.uint8) * MASK_FOREGROUND).save(staging / f"new-{index}.png", format="PNG")
        try:
            for index, target in enumerate(remove):
                if not target.exists():
                    continue
                backup = staging / f"removed-{index}.png"
                shutil.copyfile(target, backup)
                changed.append((target, backup))
                target.unlink()
            for index, (_, target) in enumerate(files):
                if target.exists() and not overwrite:
                    raise CaptureError("保存済みmaskがある。上書きするか確認してください")  # noqa: TRY301
                backup = staging / f"old-{index}.png" if target.exists() else None
                if backup is not None:
                    shutil.copyfile(target, backup)
                changed.append((target, backup))
                (staging / f"new-{index}.png").replace(target)
            yield
        except (CaptureError, OSError, ValueError, TypeError, KeyError) as error:
            # SQL失敗もここを通し、既存の承認済みファイルを復元する。
            try:
                for target, backup in reversed(changed):
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        backup.replace(target)
            except OSError as restore_error:
                raise CaptureError(f"承認用maskの復元に失敗した: {restore_error}; 元のエラー: {error}") from restore_error
            raise CaptureError(f"承認用maskを保存できない: {error}") from error
