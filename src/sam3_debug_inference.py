"""bbox・マスク・スコアを描画する、SAM3 単体のデバッグ推論スクリプト。

本スクリプトは本番のキャプチャワークフローから意図的に切り離している。
``sam3_pipeline`` を一切 import せず、SQLite やアノテーション保存先にも触れない。
``sam3`` パッケージを直接呼び出すことで、タグ照合・キャッシュ・レビューの
ロジックから独立してモデル本来の挙動を検証できる構造とした。
結果は専用のローカル出力ディレクトリにのみ書き込む。

Usage:
    ファイル冒頭付近の ``SETTINGS`` を編集し、引数なしで
    ``uv run python src/sam3_debug_inference.py`` を実行する。

Note:
    画像パスを指定した場合は 1 枚、ディレクトリを指定した場合は直下の対応画像
    すべて(再帰なし)を推論する。既定のプロンプトと閾値は ``configs/capture.json``
    と同じ値を、ツールの独立性を保つため意図的に複製して保持している。
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger(__name__)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROMPT = "metal tool"  # SETTINGS で未指定のときに使う SAM3 テキストプロンプト
DEFAULT_MIN_SCORE = 0.5  # 検出として採用する信頼度の下限
DEFAULT_OUTPUT_DIR = Path("sam3_debug_output")  # デバッグ結果の出力ルート(実行ごとに日時サブフォルダを作る)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})  # 推論対象とする拡張子(小文字)
SUMMARY_FILE = "summary.json"  # 検出数・score・box などの集計を書き出すファイル名
OUTPUT_SUFFIX = "_debug.png"  # 描画済み画像のファイル名末尾(<元の名前>_debug.png)
BPE_RELATIVE_PATH = Path("assets") / "bpe_simple_vocab_16e6.txt.gz"  # sam3 パッケージ内のトークナイザ語彙ファイルの相対パス
MASK_ALPHA = 0.45  # マスク塗りつぶしの不透明度(0=透明, 1=不透明)
MASK_THRESHOLD = 0.0  # マスクのロジットを真偽値化する境界値(これより大きい画素を物体とみなす)
MIN_LINE_WIDTH = 2  # bbox 枠線の最小太さ(px)
LINE_WIDTH_DIVISOR = 400  # 枠線の太さ = 画像幅 ÷ この値(画像サイズに比例させる)
MIN_FONT_SIZE = 14  # スコアラベルの最小フォントサイズ(px)
FONT_SIZE_DIVISOR = 45  # フォントサイズ = 画像幅 ÷ この値(画像サイズに比例させる)
# 隣り合う検出同士を区別できるよう、色相の離れた配色を用いる。
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 77, 77), (255, 176, 32), (242, 230, 61), (77, 217, 100), (46, 230, 214),
    (77, 157, 255), (179, 107, 255), (255, 107, 214), (255, 138, 92), (166, 255, 77),
)


@dataclass(frozen=True)
class DebugSettings:
    """ユーザーが編集する実行設定。コマンドライン引数ではなくコード内で完結させる。

    Attributes:
        target: 推論対象の画像ファイル、または画像を直下に含むディレクトリ。
        output: 出力ルート。内部に日時名のサブディレクトリを作成する。
        prompt: SAM3 に与えるテキストプロンプト。
        min_score: 信頼度の閾値。(0, 1] の範囲で指定する。
        checkpoint: ローカルの SAM3 チェックポイント。None の場合は既定の重みを使う。
        bpe_path: トークナイザの語彙ファイル。None の場合は同梱アセットを使う。
    """

    target: Path
    output: Path = DEFAULT_OUTPUT_DIR
    prompt: str = DEFAULT_PROMPT
    min_score: float = DEFAULT_MIN_SCORE
    checkpoint: Path | None = None
    bpe_path: Path | None = None


# 推論対象や条件を変えるときは、このブロックを編集する。
SETTINGS = DebugSettings(
    target=WORKSPACE_ROOT / "shared_data" / "2026-10-26" / "session_105202_000000_53c4b4" / "raw",
    min_score=0.5,
)


@dataclass(frozen=True)
class Detection:
    """元画像のピクセル座標で表した SAM3 の検出結果 1 件。

    Attributes:
        box: ``(xmin, ymin, xmax, ymax)`` のピクセル座標のボックス。
        score: モデルの信頼度。
        mask: 形状 ``(height, width)`` の真偽値マスク。
    """

    box: tuple[float, float, float, float]
    score: float
    mask: np.ndarray


def collect_images(target: Path) -> list[Path]:
    """推論対象の指定を、処理する画像ファイルの一覧に解決する。

    Args:
        target: 画像ファイル、または画像を含むディレクトリ。

    Returns:
        ソート済みの画像パス。ファイル指定の場合は 1 要素のリスト。

    Raises:
        FileNotFoundError: 対象が存在しない、またはディレクトリに画像がない場合。
        ValueError: ファイル指定で、未対応の拡張子だった場合。
    """
    if target.is_file():
        if target.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {target.suffix}")
        return [target]
    if target.is_dir():
        images = sorted(path for path in target.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise FileNotFoundError(f"No images found in directory: {target}")
        return images
    raise FileNotFoundError(f"Path does not exist: {target}")


def _to_numpy(value: Any) -> np.ndarray:
    """torch テンソルまたは配列様オブジェクトを float の numpy 配列へ変換する。"""
    return value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def parse_output(output: dict[str, Any]) -> list[Detection]:
    """Sam3Processor の出力辞書を ``Detection`` のリストへ変換する。

    Args:
        output: ``masks``・``boxes``・``scores`` を保持するプロセッサの出力。

    Returns:
        モデルの出力順の検出結果。検出がない場合は空リスト。
    """
    masks, boxes, scores = output.get("masks"), output.get("boxes"), output.get("scores")
    if masks is None or boxes is None or scores is None or len(masks) == 0:
        return []
    detections = []
    for mask, box, score in zip(masks, boxes, scores, strict=True):
        box_values = tuple(float(v) for v in _to_numpy(box).tolist())
        detections.append(Detection(box_values, float(_to_numpy(score)), _to_numpy(mask).squeeze() > MASK_THRESHOLD))  # type: ignore[arg-type]
    return detections


class DebugSam3Runner:
    """本スクリプト専用の、SAM3 テキストプロンプト推論の最小ラッパー。

    本番の ``sam3_pipeline`` に依存せず単体で動作させるため、モデル構築と
    推論呼び出しのみをここに閉じ込めている。

    Args:
        prompt: 全画像に共通で与えるテキストプロンプト。
        min_score: プロセッサの信頼度閾値。
        checkpoint: ローカルのチェックポイント。None の場合は既定の重みを取得する。
        bpe_path: トークナイザの語彙ファイル。None の場合は同梱アセットを使う。

    Note:
        torch と SAM3 は遅延 import しているため、GPU 環境がなくても import 自体は
        可能である。スレッドセーフではない。
    """

    def __init__(self, prompt: str, min_score: float, checkpoint: Path | None, bpe_path: Path | None) -> None:
        """モデルを読み込み、プロセッサを構築する。"""
        import sam3  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
        from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

        self._torch = torch
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bpe = bpe_path or Path(sam3.__file__).parent / BPE_RELATIVE_PATH
        LOGGER.info("device=%s bpe=%s checkpoint=%s", self.device, bpe, checkpoint or "default")
        model = build_sam3_image_model(bpe_path=str(bpe), checkpoint_path=str(checkpoint) if checkpoint else None).to(device=self.device)
        model.eval()
        self.processor = Sam3Processor(model, confidence_threshold=min_score)

    def _autocast(self) -> AbstractContextManager[Any]:
        """本番と同じ精度方針を適用する。CUDA では bf16、CPU では fp32 を用いる。"""
        if self.device == "cuda":
            return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)
        return nullcontext()

    def infer(self, image: Image.Image) -> list[Detection]:
        """RGB 画像 1 枚に対してテキストプロンプトで推論する。

        Args:
            image: RGB 画像。

        Returns:
            プロセッサの閾値以上の検出結果。
        """
        with self._torch.inference_mode(), self._autocast():
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=state, prompt=self.prompt)
        return parse_output(output)


def render(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """画像のコピーにマスク・bbox・スコアラベルを描画する。

    Args:
        image: 元の RGB 画像。
        detections: 可視化する検出結果。

    Returns:
        描画を重ねた新しい RGB 画像。
    """
    width, height = image.size
    canvas = np.asarray(image, dtype=np.float32).copy()
    for index, detection in enumerate(detections):
        blend_color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        mask = detection.mask
        if mask.shape == (height, width):
            canvas[mask] = canvas[mask] * (1 - MASK_ALPHA) + blend_color * MASK_ALPHA
    result = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(result)
    line_width = max(MIN_LINE_WIDTH, width // LINE_WIDTH_DIVISOR)
    font = ImageFont.load_default(size=max(MIN_FONT_SIZE, width // FONT_SIZE_DIVISOR))
    for index, detection in enumerate(detections):
        color = PALETTE[index % len(PALETTE)]
        xmin, ymin, xmax, ymax = detection.box
        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=line_width)
        label = f"#{index + 1} {detection.score:.3f}"
        text_box = draw.textbbox((xmin, ymin), label, font=font)
        text_height = text_box[3] - text_box[1]
        # ボックスが画像上端に接している場合にラベルが画面外へ出ないよう、余地がなければ内側に置く。
        top = ymin - text_height - line_width if ymin - text_height - line_width >= 0 else ymin
        label_box = draw.textbbox((xmin, top), label, font=font)
        draw.rectangle(label_box, fill=color)
        draw.text((xmin, top), label, fill=(0, 0, 0), font=font)
    return result


def _summarize(path: Path, detections: list[Detection], output_file: Path) -> dict[str, Any]:
    """処理した画像 1 枚分の、JSON 化可能な記録を組み立てる。"""
    return {
        "image": str(path),
        "debug_image": str(output_file),
        "count": len(detections),
        "detections": [
            {"index": index + 1, "score": round(item.score, 6), "box": [round(v, 2) for v in item.box], "mask_pixels": int(item.mask.sum())}
            for index, item in enumerate(detections)
        ],
    }


def run(
    target: Path,
    *,
    output: Path = DEFAULT_OUTPUT_DIR,
    prompt: str = DEFAULT_PROMPT,
    min_score: float = DEFAULT_MIN_SCORE,
    checkpoint: Path | None = None,
    bpe_path: Path | None = None,
) -> int:
    """対象の全画像を推論し、デバッグ用の可視化結果を保存する。

    本関数が本スクリプトの主たる入口であり、``main`` 経由のほか、Python から
    キーワード引数を指定して直接呼び出すこともできる。

    Args:
        target: 推論対象の画像ファイル、または画像を直下に含むディレクトリ。
        output: 出力ルート。内部に日時名のサブディレクトリを作成する。
        prompt: SAM3 に与えるテキストプロンプト。
        min_score: 信頼度の閾値。(0, 1] の範囲で指定する。
        checkpoint: ローカルの SAM3 チェックポイント。
        bpe_path: トークナイザの語彙ファイル。

    Returns:
        全画像を処理できた場合は 0、1 枚でも失敗した場合は 1。

    Raises:
        ValueError: プロンプトが空、または閾値が範囲外の場合。
    """
    if not prompt.strip() or not 0 < min_score <= 1:
        raise ValueError("A nonempty prompt and min_score in (0, 1] are required")
    images = collect_images(target)
    output_dir = output / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    runner = DebugSam3Runner(prompt, min_score, checkpoint, bpe_path)
    records: list[dict[str, Any]] = []
    failures = 0
    for number, path in enumerate(images, start=1):
        try:
            with Image.open(path) as raw:
                image = raw.convert("RGB")
            detections = runner.infer(image)
            output_file = output_dir / f"{path.stem}{OUTPUT_SUFFIX}"
            render(image, detections).save(output_file)
            records.append(_summarize(path, detections, output_file))
            LOGGER.info("[%d/%d] %s: %d detections", number, len(images), path.name, len(detections))
        except (OSError, RuntimeError, ValueError) as error:
            failures += 1
            LOGGER.exception("[%d/%d] %s failed", number, len(images), path.name)
            records.append({"image": str(path), "error": str(error)})
    summary = {"prompt": prompt, "min_score": min_score, "results": records}
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved debug results to %s", output_dir.resolve())
    return 1 if failures else 0


def main() -> int:
    """``SETTINGS`` に定義した設定でデバッグ推論を実行する。

    Returns:
        プロセスの終了コード。
    """
    return run(
        SETTINGS.target,
        output=SETTINGS.output,
        prompt=SETTINGS.prompt,
        min_score=SETTINGS.min_score,
        checkpoint=SETTINGS.checkpoint,
        bpe_path=SETTINGS.bpe_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())


