"""Classify review effort from model and capture-tag evidence.

The policy keeps model confidence separate from human approval: high confidence
reduces review effort but never certifies a training annotation automatically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ConfidenceLevel(StrEnum):
    """Represent the amount of human review needed for a candidate."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Configure score bands without changing model candidate generation.

    Args:
        medium_score: Minimum score for medium confidence.
        high_score: Minimum score for high confidence.

    Note:
        Positional evidence and matching ambiguity can lower a score-derived band.
    """

    medium_score: float
    high_score: float

    def __post_init__(self) -> None:
        """Require ordered score thresholds in the model's normalized range."""
        if not 0 <= self.medium_score < self.high_score <= 1:
            raise ValueError("Confidence thresholds must satisfy 0 <= medium < high <= 1")


@dataclass(frozen=True)
class CandidateConfidence:
    """Pair a candidate's review tier with machine-readable reasons."""

    candidate_index: int
    level: ConfidenceLevel
    score: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReviewConfidence:
    """Summarize candidate confidence for image-level triage."""

    level: ConfidenceLevel
    candidates: tuple[CandidateConfidence, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _ReviewContext:
    source: Mapping[str, Any]
    assignments: Mapping[int, Mapping[str, Any]]
    tags: Mapping[tuple[str, int], Mapping[str, Any]]
    issues: tuple[str, ...]
    thresholds: ConfidenceThresholds


def assess_review_confidence(
    metadata: Mapping[str, Any],
    annotation: Mapping[str, Any],
    thresholds: ConfidenceThresholds,
) -> ReviewConfidence:
    """Rank review effort using score, tag-point containment, and match issues.

    Args:
        metadata: Immutable draft metadata containing source tags and assignments.
        annotation: Editable Labelme document for this source image.
        thresholds: Configured medium and high score boundaries.

    Returns:
        Image-level confidence and per-candidate review reasons.

    Note:
        A high result is a triage hint only. It does not replace image review or
        the explicit approval record consumed by YOLO dataset construction.
    """
    source = metadata["source"]
    assignments = {
        int(item["candidate_index"]): item
        for item in metadata["matching"].get("assignments", [])
    }
    tags = {
        (tag["instrument_uid"], int(tag["instance_no"])): tag
        for tag in source["tags"]
    }
    issues = tuple(str(issue) for issue in metadata["matching"].get("issues", []))
    context = _ReviewContext(source, assignments, tags, issues, thresholds)
    assessments = tuple(
        _assess_candidate(shape, context)
        for shape in annotation.get("shapes", [])
    )
    image_reasons = list(dict.fromkeys(reason for item in assessments for reason in item.reasons))
    if not assessments:
        image_reasons.append("no_detection")
    if any(_requires_attention(issue) for issue in issues):
        image_reasons.extend(_issue_code(issue) for issue in issues if _requires_attention(issue))
    image_reasons = list(dict.fromkeys(image_reasons))
    levels = [item.level for item in assessments]
    if not assessments or any(_requires_attention(issue) for issue in issues):
        level = ConfidenceLevel.LOW
    elif ConfidenceLevel.LOW in levels:
        level = ConfidenceLevel.LOW
    elif ConfidenceLevel.MEDIUM in levels or "estimated_tag_point" in image_reasons:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.HIGH
    return ReviewConfidence(level, assessments, tuple(image_reasons))


def _assess_candidate(
    shape: Mapping[str, Any],
    context: _ReviewContext,
) -> CandidateConfidence:
    """Evaluate one labelme rectangle against its assigned SQLite tag."""
    group_id = shape.get("group_id")
    candidate_index = int(group_id) if isinstance(group_id, int) else -1
    reasons: list[str] = []
    assignment = context.assignments.get(candidate_index)
    if assignment is None or shape.get("label") == "__unresolved__":
        return CandidateConfidence(candidate_index, ConfidenceLevel.LOW, _score(shape), ("unresolved_candidate",))
    if shape.get("label") != assignment["instrument_uid"]:
        return CandidateConfidence(candidate_index, ConfidenceLevel.LOW, _score(shape), ("label_assignment_mismatch",))

    identity = (str(assignment["instrument_uid"]), int(assignment["instance_no"]))
    tag = context.tags.get(identity)
    if tag is None:
        return CandidateConfidence(candidate_index, ConfidenceLevel.LOW, _score(shape), ("missing_tag",))
    score = _score(shape)
    if score is None:
        return CandidateConfidence(candidate_index, ConfidenceLevel.LOW, None, ("missing_score",))
    relevant_issues = tuple(issue for issue in context.issues if _issue_matches(issue, candidate_index, identity))
    if any(_requires_attention(issue) for issue in relevant_issues):
        return CandidateConfidence(candidate_index, ConfidenceLevel.LOW, score, tuple(dict.fromkeys(_issue_code(issue) for issue in relevant_issues)))

    points = [tag.get(name) for name in ("center", "end_a", "end_b")]
    points = [point for point in points if isinstance(point, Mapping)]
    if not points:
        return CandidateConfidence(candidate_index, ConfidenceLevel.LOW, score, ("missing_tag_points",))
    xmin, ymin, xmax, ymax = _bounds(shape)
    width = float(context.source["width"])
    height = float(context.source["height"])
    inside = [xmin <= float(point["x"]) * width <= xmax and ymin <= float(point["y"]) * height <= ymax for point in points]
    if not all(inside):
        return CandidateConfidence(candidate_index, ConfidenceLevel.LOW, score, ("tag_point_outside_bbox",))
    if any(point.get("status") != "visible" for point in points):
        reasons.append("estimated_tag_point")
    if score < context.thresholds.medium_score:
        reasons.append("score_below_medium")
        return CandidateConfidence(candidate_index, ConfidenceLevel.LOW, score, tuple(reasons))
    if score >= context.thresholds.high_score and not reasons:
        return CandidateConfidence(candidate_index, ConfidenceLevel.HIGH, score, ())
    reasons.append("score_medium" if score < context.thresholds.high_score else "estimated_tag_point")
    return CandidateConfidence(candidate_index, ConfidenceLevel.MEDIUM, score, tuple(dict.fromkeys(reasons)))


def _score(shape: Mapping[str, Any]) -> float | None:
    """Read a saved score defensively from the Labelme description."""
    try:
        value = float(json.loads(str(shape.get("description", "{}")))["score"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if 0 <= value <= 1 else None


def _bounds(shape: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Normalize opposite rectangle corners into left/top/right/bottom bounds."""
    first, second = shape["points"]
    return min(first[0], second[0]), min(first[1], second[1]), max(first[0], second[0]), max(first[1], second[1])


def _issue_matches(issue: str, candidate_index: int, identity: tuple[str, int]) -> bool:
    """Associate a matcher warning with its candidate or tagged instance."""
    if issue.startswith("unmatched_candidate:"):
        return issue.endswith(f":{candidate_index}")
    return f":{identity[0]}:{identity[1]}" in issue


def _requires_attention(issue: str) -> bool:
    """Treat estimate-only notes as medium, and unresolved matching as low."""
    return not issue.startswith("estimated_center:")


def _issue_code(issue: str) -> str:
    """Keep review reasons stable and independent from localized UI text."""
    return issue.split(":", maxsplit=1)[0]
