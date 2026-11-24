"""Verify SAM3 is fully built before the selective viewer is opened."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import capture_workflow
from capture_workflow import main
from sam3_pipeline.domain.capture import CaptureError, CaptureSample
from sam3_pipeline.infrastructure import capture_predictor

DEFAULT_YOLO_CONFIDENCE = 0.8
CONFIGURED_YOLO_CONFIDENCE = 0.73
DETECTOR_DEFAULT_CONFIDENCE = 0.5
EXPECTED_SIGNATURE_VARIANTS = 5


def test_predictor_initialize_builds_engine_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the engine explicitly and keep initialization idempotent."""
    events: list[str] = []

    class FakeEngine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def build(self) -> None:
            events.append("build")

    engine_module = ModuleType("sam3_pipeline.engine")
    engine_module.Sam3Engine = FakeEngine
    monkeypatch.setitem(sys.modules, "sam3_pipeline.engine", engine_module)

    predictor = object.__new__(capture_predictor.Sam3CapturePredictor)
    predictor.engine = None
    predictor.bpe_path = Path("vocabulary")
    predictor.checkpoint = None
    predictor.min_score = 0.5
    predictor.use_yolo = False
    predictor.yolo_detector = None

    predictor.initialize()
    predictor.initialize()

    assert events == ["build"]
    assert isinstance(predictor.engine, FakeEngine)


def _patch_cli_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
    *,
    initialization_error: Exception | None = None,
) -> None:
    """Replace external workflow components with small event-recording fakes."""
    session_id = "session"
    image_id = "image"
    sample = CaptureSample(
        session_id,
        image_id,
        "declaration",
        1,
        tmp_path / "image.png",
        "digest",
        10,
        10,
        1,
        (),
        (),
        "image.png",
    )

    class FakeSource:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def current_sample(self, *_args: object) -> CaptureSample:
            return sample

    class FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.review_store = None

        def sync_object_reviews(self, *_args: object) -> None:
            pass

    class FakePredictionStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def initialize(self) -> None:
            pass

    class FakeReviews:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def initialize(self) -> None:
            pass

    class FakePredictor:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.model_metadata = {}

        def initialize(self) -> None:
            events.append("initialize")
            if initialization_error is not None:
                raise initialization_error

    class FakeSelection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.entry_by_identity = {
                (session_id, image_id): {"source_path": "image.png"}
            }
            self.review_complete = threading.Event()
            self.url = "http://127.0.0.1:12345/"

        def select(self) -> list[tuple[str, str]]:
            events.append("viewer")
            return [(session_id, image_id)]

        def begin_review(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeMatcher:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class FakeReviewServer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def serve(self, **_kwargs: object) -> None:
            pass

    def fake_annotate(*_args: object, **_kwargs: object) -> list[str]:
        events.append("infer")
        return []

    monkeypatch.setattr(capture_workflow, "SqliteCaptureSource", FakeSource)
    monkeypatch.setattr(capture_workflow, "FileAnnotationStore", FakeStore)
    monkeypatch.setattr(capture_workflow, "SqliteAnnotationLedger", FakeStore)
    monkeypatch.setattr(capture_workflow, "SqlitePredictionStore", FakePredictionStore)
    monkeypatch.setattr(capture_workflow, "SqliteObjectReviewStore", FakeReviews)
    monkeypatch.setattr(capture_workflow, "Sam3CapturePredictor", FakePredictor)
    monkeypatch.setattr(capture_workflow, "SelectiveInferenceServer", FakeSelection)
    monkeypatch.setattr(capture_workflow, "ScipyInstanceMatcher", FakeMatcher)
    monkeypatch.setattr(capture_workflow, "BatchReviewServer", FakeReviewServer)
    monkeypatch.setattr(capture_workflow, "annotate_samples", fake_annotate)

    config_path = tmp_path / "capture.json"
    config_path.write_text(
        json.dumps(
            {
                "capture_root": "capture",
                "output_root": "output",
                "prompt": "instrument",
                "min_score": 0.5,
                "matching": {"max_distance": 0.03, "ambiguity_margin": 0.05},
                "review_confidence": {"medium_score": 0.5, "high_score": 0.9},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "capture").mkdir()


def test_run_initializes_model_before_opening_selection_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run model setup before selection and inference."""
    events: list[str] = []
    _patch_cli_dependencies(monkeypatch, tmp_path, events)

    result = main(["--config", str(tmp_path / "capture.json"), "run"])

    assert result == 0
    assert events == ["initialize", "viewer", "infer"]


def test_run_initialization_failure_does_not_open_selection_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stop before viewer startup when loading the SAM3 model fails."""
    events: list[str] = []
    _patch_cli_dependencies(
        monkeypatch,
        tmp_path,
        events,
        initialization_error=RuntimeError("model build failed"),
    )

    with pytest.raises(CaptureError, match="SAM3モデル初期化に失敗"):
        main(["--config", str(tmp_path / "capture.json"), "run"])

    assert events == ["initialize"]


def _sample_with_image(tmp_path: Path) -> CaptureSample:
    """Create an image snapshot suitable for the predictor boundary."""
    image_path = tmp_path / "capture.png"
    Image.new("RGB", (30, 20), (12, 34, 56)).save(image_path)
    return CaptureSample(
        "session",
        "image",
        "declaration",
        1,
        image_path,
        hashlib.sha256(image_path.read_bytes()).hexdigest(),
        30,
        20,
        1,
        (),
    )


def _predictor_for_inference(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_yolo: bool,
    yolo_detections: list[dict[str, object]] | None = None,
    yolo_error: Exception | None = None,
) -> tuple[capture_predictor.Sam3CapturePredictor, list[tuple[object, ...]], list[str]]:
    """Build a predictor around small fakes and record SAM3 calls."""
    calls: list[tuple[object, ...]] = []
    image_events: list[str] = []

    class FakeEngine:
        def inference_context(self) -> object:
            return nullcontext()

        def encode(self, image: Image.Image) -> dict[str, object]:
            image_events.append("encode")
            return {"encoded": True}

        def detect(self, image: Image.Image, state: dict[str, object], prompt: str, **kwargs: object) -> list[SimpleNamespace]:
            calls.append((image, state, prompt, kwargs))
            mask = np.ones((20, 30), dtype=bool)
            detection = SimpleNamespace(box=(4.0, 5.0, 12.0, 15.0), score=0.9, mask=mask)
            return [detection]

    class FakeYolo:
        def infer(self, _image: np.ndarray) -> list[dict[str, object]]:
            if yolo_error is not None:
                raise yolo_error
            return yolo_detections or []

    predictor = object.__new__(capture_predictor.Sam3CapturePredictor)
    predictor.use_yolo = use_yolo
    predictor.yolo_detector = FakeYolo() if use_yolo else None
    predictor.engine = FakeEngine()
    predictor.prompt = "metal tool"
    predictor.min_score = 0.5
    monkeypatch.setattr(predictor, "initialize", lambda: None)
    return predictor, calls, image_events


def test_infer_passes_all_valid_yolo_boxes_and_returns_sam3_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep edge-touching boxes, discard malformed boxes, and return SAM3 output."""
    predictor, calls, image_events = _predictor_for_inference(
        monkeypatch,
        use_yolo=True,
        yolo_detections=[
            {"xyxy": [1.0, 2.0, 20.0, 15.0]},
            {"xyxy": [-2.0, 1.0, 40.0, 25.0]},
            {"xyxy": [0.0, 2.0, 20.0, 15.0]},
            {"xyxy": [1.0, 2.0, float("nan"), 15.0]},
            {"xyxy": [0.0, 0.0, 0.0, 0.0]},
        ],
    )
    sample = _sample_with_image(tmp_path)

    candidates = predictor.infer(sample)

    assert len(image_events) == 1
    assert calls[0][2] == "metal tool"
    assert calls[0][3] == {
        "gt_boxes": [
            (1.0, 2.0, 20.0, 15.0),
            (0.0, 1.0, 30.0, 20.0),
            (0.0, 2.0, 20.0, 15.0),
        ]
    }
    assert len(candidates) == 1
    assert candidates[0].box == (4.0, 5.0, 12.0, 15.0)


def test_infer_falls_back_to_text_for_zero_yolo_boxes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use the text prompt alone when YOLO succeeds without valid boxes."""
    predictor, calls, _ = _predictor_for_inference(monkeypatch, use_yolo=True, yolo_detections=[])

    predictor.infer(_sample_with_image(tmp_path))

    assert calls[0][2] == "metal tool"
    assert calls[0][3] == {}


def test_infer_does_not_fallback_after_yolo_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Report YOLO errors without running SAM3 text-only inference."""
    predictor, calls, _ = _predictor_for_inference(
        monkeypatch,
        use_yolo=True,
        yolo_error=RuntimeError("test failure"),
    )

    with pytest.raises(CaptureError, match="YOLO inference failed"):
        predictor.infer(_sample_with_image(tmp_path))

    assert calls == []


def test_infer_keeps_sam3_only_when_yolo_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not invoke YOLO or change the text-only engine call when disabled."""
    predictor, calls, _ = _predictor_for_inference(monkeypatch, use_yolo=False)

    predictor.infer(_sample_with_image(tmp_path))

    assert calls[0][2] == "metal tool"
    assert calls[0][3] == {}


def test_capture_config_injects_yolo_defaults_and_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pass config defaults or explicit confidence through to the predictor."""
    events: list[str] = []
    _patch_cli_dependencies(monkeypatch, tmp_path, events)
    config_path = tmp_path / "capture.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"use_yolo": True, "yolo_confidence": CONFIGURED_YOLO_CONFIDENCE})
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class FakePredictor:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            events.append(f"config:{kwargs['use_yolo']}:{kwargs['yolo_confidence']}")
            self.model_metadata = {"use_yolo": kwargs["use_yolo"]}

        def initialize(self) -> None:
            events.append("initialize")

    monkeypatch.setattr(capture_workflow, "Sam3CapturePredictor", FakePredictor)

    main(["--config", str(config_path), "run"])

    assert events[:2] == [f"config:True:{CONFIGURED_YOLO_CONFIDENCE}", "initialize"]


def test_capture_config_uses_yolo_defaults_when_keys_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Supply the agreed defaults when older capture files omit YOLO options."""
    events: list[str] = []
    _patch_cli_dependencies(monkeypatch, tmp_path, events)

    class FakePredictor:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            events.append(f"config:{kwargs['use_yolo']}:{kwargs['yolo_confidence']}")
            self.model_metadata = {}

        def initialize(self) -> None:
            events.append("initialize")

    monkeypatch.setattr(capture_workflow, "Sam3CapturePredictor", FakePredictor)

    main(["--config", str(tmp_path / "capture.json"), "run"])

    assert events[:2] == [f"config:False:{DEFAULT_YOLO_CONFIDENCE}", "initialize"]


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), True])
def test_capture_config_rejects_invalid_yolo_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, confidence: float | bool
) -> None:
    """Reject invalid confidence before starting inference."""
    _patch_cli_dependencies(monkeypatch, tmp_path, [])
    config_path = tmp_path / "capture.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["yolo_confidence"] = confidence
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(CaptureError, match="yolo_confidence"):
        main(["--config", str(config_path), "run"])


def test_yolo_confidence_injection_preserves_detector_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override capture confidence while retaining configured standalone default."""
    weights = Path("weights.pt")
    monkeypatch.setattr("yolov12.inference.load_settings", lambda: (weights, DETECTOR_DEFAULT_CONFIDENCE))

    class FakeModel:
        def __init__(self) -> None:
            self.names: dict[int, str] = {}
            self.confidences: list[float] = []

        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            self.confidences.append(kwargs["conf"])
            return [SimpleNamespace(boxes=None)]

    model = FakeModel()
    monkeypatch.setattr("yolov12.inference._load_model", lambda _path: model)

    configured = capture_predictor.YoloDetector()
    injected = capture_predictor.YoloDetector(DEFAULT_YOLO_CONFIDENCE)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    configured.infer(image)
    injected.infer(image)

    assert configured.confidence_threshold == DETECTOR_DEFAULT_CONFIDENCE
    assert injected.confidence_threshold == DEFAULT_YOLO_CONFIDENCE
    assert model.confidences == [DETECTOR_DEFAULT_CONFIDENCE, DEFAULT_YOLO_CONFIDENCE]


def test_yolo_signature_tracks_mode_confidence_and_weight_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Include the resolved YOLO model identity in the cache signature."""
    package = tmp_path / "sam3"
    (package / "model").mkdir(parents=True)
    (package / "assets").mkdir()
    (package / "model_builder.py").write_text("builder", encoding="utf-8")
    (package / "model" / "sam3_image_processor.py").write_text("processor", encoding="utf-8")
    bpe = package / "assets" / "vocab.gz"
    bpe.write_bytes(b"vocab")
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"initial")
    monkeypatch.setattr(capture_predictor.importlib.util, "find_spec", lambda _name: SimpleNamespace(origin=str(package / "__init__.py")))
    distribution = SimpleNamespace(version="sam3-test", read_text=lambda _name: "{}")
    monkeypatch.setattr(capture_predictor.importlib.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(capture_predictor.importlib.metadata, "version", lambda name: {"torch": "torch-test", "ultralytics": "ultralytics-test"}[name])
    monkeypatch.setattr(capture_predictor, "load_settings", lambda: (weights, 0.5))

    disabled = capture_predictor.Sam3CapturePredictor(None, "tool", 0.5, bpe)
    enabled = capture_predictor.Sam3CapturePredictor(None, "tool", 0.5, bpe, use_yolo=True, yolo_confidence=0.8)
    changed_confidence = capture_predictor.Sam3CapturePredictor(None, "tool", 0.5, bpe, use_yolo=True, yolo_confidence=0.7)
    weights.write_bytes(b"changed")
    disabled_changed_weights = capture_predictor.Sam3CapturePredictor(None, "tool", 0.5, bpe)
    changed_weights = capture_predictor.Sam3CapturePredictor(None, "tool", 0.5, bpe, use_yolo=True, yolo_confidence=0.8)

    predictors = (disabled, enabled, changed_confidence, disabled_changed_weights, changed_weights)
    assert len({predictor.signature for predictor in predictors}) == EXPECTED_SIGNATURE_VARIANTS
    prediction_ids = {
        hashlib.sha256(f"sample:{predictor.signature}".encode()).hexdigest()
        for predictor in predictors
    }
    assert len(prediction_ids) == EXPECTED_SIGNATURE_VARIANTS
    assert enabled.model_metadata["use_yolo"] is True
    assert enabled.model_metadata["yolo_confidence"] == DEFAULT_YOLO_CONFIDENCE
    assert disabled.model_metadata["yolo_weights_sha256"] != disabled_changed_weights.model_metadata["yolo_weights_sha256"]
    assert enabled.model_metadata["yolo_weights_sha256"] != changed_weights.model_metadata["yolo_weights_sha256"]
