"""Adapt the existing SAM3 wrapper to one shared text prompt per capture image.

Explicit local checkpoints and implementation fingerprints make cache reuse
traceable. Model loading is lazy so cached tag reassignment needs no GPU memory.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from sam3_pipeline.domain.annotation import Candidate
from sam3_pipeline.domain.capture import CaptureError, CaptureSample
from sam3_pipeline.infrastructure.annotation_store import content_hash, digest_file
from sam3_pipeline.infrastructure.matching import ArrayMask

if TYPE_CHECKING:
    from sam3_pipeline.engine import Sam3Engine

PREDICTOR_VERSION = "single-prompt-capture-v2"
HF_CHECKPOINT_SOURCE = "facebook/sam3/sam3.pt"


class Sam3CapturePredictor:
    """Wrap the existing engine without tag-dependent prompts or fallback calls.

    Args:
        checkpoint: Optional local checkpoint; None loads the SAM3 checkpoint from Hugging Face.
        prompt: One fixed generic instrument description.
        min_score: Confidence threshold applied inside the processor and adapter.
        bpe_path: Optional explicit vocabulary; otherwise the installed SAM3 asset.

    Note:
        SAM3 and Torch must be installed for a real inference run. A local
        checkpoint is content-hashed; HF mode is identified by its model source.
    """

    def __init__(self, checkpoint: Path | None, prompt: str, min_score: float, bpe_path: Path | None = None) -> None:
        """Resolve model identity before permitting any cache lookup."""
        if checkpoint is not None and not checkpoint.is_file():
            raise CaptureError(f"SAM3 checkpoint does not exist: {checkpoint}")
        if not prompt.strip() or not 0 < min_score <= 1:
            raise CaptureError("A nonempty prompt and valid score threshold are required")
        self.checkpoint = checkpoint.resolve() if checkpoint is not None else None
        self.prompt = prompt.strip()
        self.min_score = min_score
        self.engine: Sam3Engine | None = None
        specification = importlib.util.find_spec("sam3")
        if specification is None or specification.origin is None:
            raise CaptureError("Install the project's SAM3 environment before annotation")
        package = Path(specification.origin).parent
        self.bpe_path = bpe_path or package / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        implementation = [package / "model_builder.py", package / "model" / "sam3_image_processor.py", Path(__file__), Path(__file__).parents[1] / "engine.py"]
        distribution = importlib.metadata.distribution("sam3")
        checkpoint_identity = digest_file(self.checkpoint) if self.checkpoint is not None else HF_CHECKPOINT_SOURCE
        self._signature = content_hash({"version": PREDICTOR_VERSION, "prompt": self.prompt, "min_score": min_score,
                        "checkpoint": checkpoint_identity, "bpe": digest_file(self.bpe_path),
                                        "implementation": [digest_file(path) for path in implementation],
                                        "sam3": distribution.version, "sam3_source": distribution.read_text("direct_url.json"),
                                        "torch": importlib.metadata.version("torch")})

    @property
    def signature(self) -> str:
        """Return the fully resolved prediction recipe fingerprint."""
        return self._signature

    def infer(self, sample: CaptureSample) -> tuple[Candidate, ...]:
        """Run one text detection and validate its geometry against the raw image.

        Args:
            sample: RGB image with recorded hash and dimensions.

        Returns:
            Generic mask candidates, including an empty result on missed detections.

        Raises:
            CaptureError: Source contents, inference, or result geometry is invalid.
        """
        try:
            if digest_file(sample.image_path) != sample.sha256:
                raise CaptureError("Image changed after the capture snapshot")
            if self.engine is None:
                # Review and cached reassignment must not initialize the GPU runtime.
                from sam3_pipeline.engine import Sam3Engine  # noqa: PLC0415

                self.engine = Sam3Engine(self.bpe_path, checkpoint_path=self.checkpoint, confidence_threshold=self.min_score)
                self.engine.build()
            with Image.open(sample.image_path) as raw:
                image = raw.convert("RGB")
            if image.size != (sample.width, sample.height):
                raise CaptureError("Capture metadata dimensions differ from RGB image")
            with self.engine.inference_context():
                state = self.engine.encode(image)
                detections = self.engine.detect(image, state, self.prompt)
            candidates = []
            for index, detection in enumerate(detections):
                if not math.isfinite(detection.score) or detection.score < self.min_score:
                    continue
                mask = np.asarray(detection.mask, dtype=bool)
                if mask.shape != (sample.height, sample.width):
                    raise CaptureError("SAM3 mask does not match original image dimensions")
                xmin, ymin, xmax, ymax = detection.box
                if not all(math.isfinite(value) for value in detection.box):
                    raise CaptureError("Nonfinite SAM3 box")
                box = (max(0.0, xmin), max(0.0, ymin), min(float(sample.width), xmax), min(float(sample.height), ymax))
                if box[0] >= box[2] or box[1] >= box[3] or not mask.any():
                    continue
                candidates.append(Candidate(index, box, detection.score, ArrayMask(mask)))
            return tuple(candidates)
        except (OSError, RuntimeError, ValueError, ImportError) as error:
            raise CaptureError(f"SAM3 inference failed for {sample.key}: {error}") from error
