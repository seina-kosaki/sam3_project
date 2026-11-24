"""Adapt the existing SAM3 wrapper to one shared text prompt per capture image.

Explicit local checkpoints and implementation fingerprints make cache reuse
traceable. Cache-only workflows can defer model loading; run initializes before
opening the selection viewer.
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
from yolov12.inference import YoloDetector, load_settings

if TYPE_CHECKING:
    from sam3_pipeline.engine import Sam3Engine

PREDICTOR_VERSION = "single-prompt-capture-v2"
HF_CHECKPOINT_SOURCE = "facebook/sam3/sam3.pt"
BOX_COORDINATE_COUNT = 4


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

    def __init__(
        self,
        checkpoint: Path | None,
        prompt: str,
        min_score: float,
        bpe_path: Path | None = None,
        *,
        use_yolo: bool = False,
        yolo_confidence: float = 0.8,
    ) -> None:
        """Resolve model identity before permitting any cache lookup."""
        if checkpoint is not None and not checkpoint.is_file():
            raise CaptureError(f"SAM3 checkpoint does not exist: {checkpoint}")
        if not prompt.strip() or not 0 < min_score <= 1:
            raise CaptureError("A nonempty prompt and valid score threshold are required")
        if not isinstance(use_yolo, bool):
            raise CaptureError("use_yolo must be a boolean")
        confidence_is_number = isinstance(yolo_confidence, (int, float)) and not isinstance(yolo_confidence, bool)
        confidence_is_valid = confidence_is_number and math.isfinite(yolo_confidence) and 0 <= yolo_confidence <= 1
        if not confidence_is_valid:
            raise CaptureError("yolo_confidence must be a finite number from 0 to 1")
        self.checkpoint = checkpoint.resolve() if checkpoint is not None else None
        self.prompt = prompt.strip()
        self.min_score = min_score
        self.use_yolo = use_yolo
        self.yolo_confidence = float(yolo_confidence)
        self.engine: Sam3Engine | None = None
        self.yolo_detector: YoloDetector | None = None
        specification = importlib.util.find_spec("sam3")
        if specification is None or specification.origin is None:
            raise CaptureError("Install the project's SAM3 environment before annotation")
        package = Path(specification.origin).parent
        self.bpe_path = bpe_path or package / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        implementation = [package / "model_builder.py", package / "model" / "sam3_image_processor.py", Path(__file__), Path(__file__).parents[1] / "engine.py"]
        distribution = importlib.metadata.distribution("sam3")
        checkpoint_identity = digest_file(self.checkpoint) if self.checkpoint is not None else HF_CHECKPOINT_SOURCE
        yolo_weights_sha256 = None
        yolo_implementation = digest_file(Path(__file__).parents[2] / "yolov12" / "inference.py")
        yolo_version = None
        try:
            weights_path, _ = load_settings()
            yolo_weights_sha256 = digest_file(weights_path)
        except (OSError, RuntimeError, ValueError) as error:
            if self.use_yolo:
                raise CaptureError(f"YOLO configuration could not be resolved: {error}") from error
        if self.use_yolo:
            try:
                yolo_version = importlib.metadata.version("ultralytics")
            except importlib.metadata.PackageNotFoundError as error:
                raise CaptureError(f"YOLO configuration could not be resolved: {error}") from error
        self.model_metadata: dict[str, object] = {
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "bpe_path": str(self.bpe_path),
            "prompt": self.prompt,
            "min_score": self.min_score,
            "use_yolo": self.use_yolo,
            "yolo_confidence": self.yolo_confidence,
            "yolo_weights_sha256": yolo_weights_sha256,
        }
        self._signature = content_hash({"version": PREDICTOR_VERSION, "prompt": self.prompt, "min_score": min_score,
                        "checkpoint": checkpoint_identity, "bpe": digest_file(self.bpe_path),
                                        "implementation": [digest_file(path) for path in implementation],
                                        "sam3": distribution.version, "sam3_source": distribution.read_text("direct_url.json"),
                                        "torch": importlib.metadata.version("torch"), "use_yolo": self.use_yolo,
                                        "yolo_confidence": self.yolo_confidence, "yolo_weights_sha256": yolo_weights_sha256,
                                        "yolo_implementation": yolo_implementation, "ultralytics": yolo_version})

    @property
    def signature(self) -> str:
        """Return the fully resolved prediction recipe fingerprint."""
        return self._signature

    def initialize(self) -> None:
        """Load the SAM3 engine before an interactive workflow is opened."""
        if self.engine is not None and (not self.use_yolo or self.yolo_detector is not None):
            return

        if self.engine is None:
            from sam3_pipeline.engine import Sam3Engine  # noqa: PLC0415

            engine = Sam3Engine(
                self.bpe_path,
                checkpoint_path=self.checkpoint,
                confidence_threshold=self.min_score,
            )
            engine.build()
            self.engine = engine
        if self.use_yolo:
            try:
                self.yolo_detector = YoloDetector(self.yolo_confidence)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise CaptureError(f"YOLO model initialization failed: {error}") from error

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
            self.initialize()
            assert self.engine is not None
            with Image.open(sample.image_path) as raw:
                image = raw.convert("RGB")
            if image.size != (sample.width, sample.height):
                raise CaptureError("Capture metadata dimensions differ from RGB image")
            with self.engine.inference_context():
                state = self.engine.encode(image)
                if self.use_yolo:
                    assert self.yolo_detector is not None
                    try:
                        yolo_detections = self.yolo_detector.infer(np.asarray(image, dtype=np.uint8))
                    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                        raise CaptureError(f"YOLO inference failed for {sample.key}: {error}") from error
                    gt_boxes = _valid_yolo_boxes(yolo_detections, sample.width, sample.height)
                    if gt_boxes:
                        detections = self.engine.detect(image, state, self.prompt, gt_boxes=gt_boxes)
                    else:
                        detections = self.engine.detect(image, state, self.prompt)
                else:
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


def _valid_yolo_boxes(detections: list[dict[str, object]], width: int, height: int) -> list[tuple[float, float, float, float]]:
    """Clip usable YOLO boxes to the image and discard malformed detections."""
    valid_boxes = []
    for detection in detections:
        coordinates = detection.get("xyxy")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != BOX_COORDINATE_COUNT:
            continue
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in coordinates):
            continue
        box = tuple(float(value) for value in coordinates)
        if not all(math.isfinite(value) for value in box):
            continue
        xmin, ymin, xmax, ymax = box
        clipped = (max(0.0, xmin), max(0.0, ymin), min(float(width), xmax), min(float(height), ymax))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            continue
        valid_boxes.append(clipped)
    return valid_boxes
