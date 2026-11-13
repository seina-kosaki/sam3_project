"""Verify SAM3 is fully built before the selective viewer is opened."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import capture_workflow
from capture_workflow import main
from sam3_pipeline.domain.capture import CaptureError, CaptureSample
from sam3_pipeline.infrastructure import capture_predictor


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
            pass

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
