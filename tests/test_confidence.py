"""Verify confidence triage rules independently from model and storage adapters."""

from __future__ import annotations

import json
from typing import Any

import pytest

from sam3_pipeline.domain.confidence import ConfidenceLevel, ConfidenceThresholds, assess_review_confidence

THRESHOLDS = ConfidenceThresholds(medium_score=0.65, high_score=0.85)


def _review(score: float, *, point_x: float = 0.5, issues: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one assigned candidate and one SQL tag for policy tests."""
    metadata = {
        "source": {
            "width": 100,
            "height": 100,
            "tags": [{
                "instrument_uid": "tool_a",
                "instance_no": 1,
                "center": {"x": point_x, "y": 0.5, "status": "visible"},
                "end_a": {"x": 0.4, "y": 0.4, "status": "visible"},
                "end_b": None,
            }],
        },
        "matching": {
            "assignments": [{"candidate_index": 0, "instrument_uid": "tool_a", "instance_no": 1}],
            "issues": issues or [],
        },
    }
    annotation = {
        "shapes": [{
            "label": "tool_a",
            "group_id": 0,
            "points": [[30, 30], [70, 70]],
            "description": json.dumps({"score": score}),
        }],
    }
    return metadata, annotation


def test_assess_review_confidence_high_score_and_all_tag_points_inside_is_high() -> None:
    """Mark strong, geometrically consistent detections high."""
    metadata, annotation = _review(0.9)

    assessment = assess_review_confidence(metadata, annotation, THRESHOLDS)

    assert assessment.level is ConfidenceLevel.HIGH


def test_assess_review_confidence_medium_score_is_medium() -> None:
    """Keep mid-band scores available for normal review."""
    metadata, annotation = _review(0.7)

    assessment = assess_review_confidence(metadata, annotation, THRESHOLDS)

    assert assessment.level is ConfidenceLevel.MEDIUM


def test_assess_review_confidence_tag_point_outside_bbox_is_low() -> None:
    """Escalate candidates that exclude a tagged location."""
    metadata, annotation = _review(0.95, point_x=0.9)

    assessment = assess_review_confidence(metadata, annotation, THRESHOLDS)

    assert assessment.level is ConfidenceLevel.LOW
    assert "tag_point_outside_bbox" in assessment.reasons


def test_assess_review_confidence_unmatched_tag_is_low() -> None:
    """Escalate images with an unmatched tagged instance."""
    metadata, annotation = _review(0.95, issues=["unmatched_tag:tool_a:1"])

    assessment = assess_review_confidence(metadata, annotation, THRESHOLDS)

    assert assessment.level is ConfidenceLevel.LOW


def test_assess_review_confidence_changed_label_is_low() -> None:
    """Reclassify edited labels when they disagree with the original match."""
    metadata, annotation = _review(0.95)
    annotation["shapes"][0]["label"] = "tool_b"

    assessment = assess_review_confidence(metadata, annotation, THRESHOLDS)

    assert assessment.level is ConfidenceLevel.LOW
    assert "label_assignment_mismatch" in assessment.reasons


def test_confidence_thresholds_reject_reversed_order() -> None:
    """Reject contradictory confidence bands before reviewing images."""
    with pytest.raises(ValueError, match="medium < high"):
        ConfidenceThresholds(medium_score=0.9, high_score=0.8)
