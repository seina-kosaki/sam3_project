"""SAM3 のラッパー。モデル構築・画像エンコード・プロンプト推論だけを担当する。

画像エンコード(set_image)は画像1枚につき1回だけ行い、その state を使い回して
複数のプロンプト（複数ラベル・フォールバック）を順に推論する。
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from PIL import Image

from .models import Box

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class RawDetection:
    """Represent one SAM3 box, confidence score, and binary mask."""

    box: Box
    score: float
    mask: np.ndarray  # (H, W) bool


class Sam3Engine:
    """Own model setup and processor state while pipelines decide annotation policy."""

    # Key in backbone_out; update only when SAM3 changes its feature-map API.
    FEATURE_MAP_KEY = "vision_features"

    def __init__(self, bpe_path: str | Path | None = None, *, checkpoint_path: str | Path | None = None, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        """Configure model loading without changing the legacy default checkpoint.

        Args:
            bpe_path: Optional tokenizer vocabulary path.
            checkpoint_path: Optional explicit local weights for reproducible capture runs.
            confidence_threshold: Processor candidate confidence threshold.
        """
        import torch  # noqa: PLC0415

        self._torch = torch
        self.bpe_path = Path(bpe_path).resolve() if bpe_path else None
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = None
        self.processor = None

    def build(self) -> None:
        """Load the configured checkpoint and construct the SAM3 processor."""
        import sam3  # noqa: PLC0415
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415
        from sam3.model_builder import build_sam3_image_model  # noqa: PLC0415

        bpe = self.bpe_path or Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        print(f"BPE Path: {bpe}")
        model = build_sam3_image_model(bpe_path=str(bpe), checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None).to(device=self.device)
        model.eval()
        self.model = model
        self.processor = Sam3Processor(model, confidence_threshold=self.confidence_threshold)

    def inference_context(self) -> AbstractContextManager[None]:
        """Disable gradient tracking for all images processed in the context.

        Returns:
            Torch inference-mode context manager.
        """
        return cast(AbstractContextManager[None], self._torch.inference_mode())

    def _autocast(self) -> AbstractContextManager[Any]:
        if self.device == "cuda":
            return self._torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    # ------------------------------------------------------------------
    def encode(self, image: Image.Image) -> dict:
        """画像を1回だけエンコードし、推論state（dict）を返す。"""
        with self._autocast():
            return self.processor.set_image(image)

    def detect(
        self,
        image: Image.Image,
        state: dict,
        prompt: str,
        gt_boxes: list[Box] | None = None,
    ) -> list[RawDetection]:
        """テキストプロンプト（+ 任意でbox prompt）で推論し、検出結果を返す。

        gt_boxes: [xmin,ymin,xmax,ymax]（ピクセル座標）のリスト。指定すると
        SAM3の add_geometric_prompt() でpositiveなvisual promptとして追加する。
        """
        with self._autocast():
            if hasattr(self.processor, "reset_all_prompts"):
                self.processor.reset_all_prompts(state)
            else:  # 古いSAM3: 前回プロンプトが残らないよう再エンコードする
                state.clear()
                state.update(self.processor.set_image(image))

            output = self.processor.set_text_prompt(state=state, prompt=prompt)

            if gt_boxes:
                w, h = image.size
                for xmin, ymin, xmax, ymax in gt_boxes:
                    box_norm = [(xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h, (xmax - xmin) / w, (ymax - ymin) / h]
                    output = self.processor.add_geometric_prompt(box=box_norm, label=True, state=state)

        return self._to_detections(output)

    @staticmethod
    def _to_detections(output: dict) -> list[RawDetection]:
        masks, boxes, scores = output.get("masks"), output.get("boxes"), output.get("scores")
        if masks is None or boxes is None or scores is None or len(masks) == 0 or len(boxes) == 0:
            return []

        def to_np(x: Any) -> np.ndarray:
            return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        dets = []
        for mask, box, score in zip(masks, boxes, scores, strict=True):
            mask_np = to_np(mask).squeeze() > 0
            box_t = tuple(float(v) for v in to_np(box).tolist())
            dets.append(RawDetection(box=box_t, score=float(to_np(score)), mask=mask_np))
        return dets

    def extract_features(self, state: dict) -> np.ndarray | None:
        """vision_features を float32 の numpy 配列で返す（先頭のバッチ次元1は除去）。"""
        backbone_out = state.get("backbone_out") if isinstance(state, dict) else None
        tensor = backbone_out.get(self.FEATURE_MAP_KEY) if backbone_out else None
        if tensor is None:
            return None
        arr = tensor.detach().float().cpu().numpy()
        return arr.squeeze(0) if arr.shape[0] == 1 else arr
