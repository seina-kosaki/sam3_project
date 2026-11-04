"""Use SciPy's established assignment solver rather than a custom Hungarian algorithm.

Distance transforms tolerate small holes in thin instrument masks. Dummy columns
allow missed tags to remain unmatched instead of forcing incorrect class labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

from sam3_pipeline.domain.annotation import Assignment, Candidate, MatchResult
from sam3_pipeline.domain.capture import ESTIMATED, VISIBLE, CaptureError, CaptureSample, InstanceTag

MATCH_VERSION = "mask-distance-assignment-v1"
UNMATCHED_COST = 2.0
FORBIDDEN_COST = 1_000_000.0
SCORE_WEIGHT = 0.1
ENDPOINT_WEIGHT = 0.1
ESTIMATED_DISTANCE_MULTIPLIER = 2.0
MASK_DIMENSIONS = 2


@dataclass(frozen=True)
class MatchSettings:
    """Inject dataset-dependent tolerances rather than embedding capture assumptions.

    Args:
        max_distance: Maximum center-to-mask distance, relative to image diagonal.
        ambiguity_margin: Minimum cost separation from competing row/column matches.
    """

    max_distance: float
    ambiguity_margin: float

    def __post_init__(self) -> None:
        """Reject thresholds that would disable meaningful geometric gating."""
        if not 0 < self.max_distance < 1 or not 0 <= self.ambiguity_margin < 1:
            raise CaptureError("Invalid matching thresholds")


class ArrayMask:
    """Adapt binary NumPy masks to a dependency-free distance protocol.

    The lazily cached SciPy transform avoids repeatedly scanning all pixels per tag.
    """

    def __init__(self, pixels: np.ndarray) -> None:
        """Keep a binary mask and defer the distance transform until needed."""
        self.pixels = np.asarray(pixels, dtype=bool)
        if self.pixels.ndim != MASK_DIMENSIONS or not self.pixels.size:
            raise CaptureError("Expected a nonempty two-dimensional mask")
        self._distances: np.ndarray | None = None

    def distance(self, x: float, y: float) -> float:
        """Measure distance to foreground in units independent of image resolution.

        Args:
            x: Normalized horizontal position.
            y: Normalized vertical position.

        Returns:
            Distance divided by the image diagonal, or infinity for an empty mask.
        """
        if not self.pixels.any():
            return math.inf
        height, width = self.pixels.shape
        column = min(width - 1, max(0, int(x * width)))
        row = min(height - 1, max(0, int(y * height)))
        if self._distances is None:
            self._distances = distance_transform_edt(~self.pixels)
        return float(self._distances[row, column]) / math.hypot(width, height)


class ScipyInstanceMatcher:
    """Make conservative global assignments; ambiguous identities require review.

    Args:
        settings: Injected geometric thresholds.

    Note:
        Estimated centers are tolerated more widely but always recorded for review.
        Invisible and pointless hints never receive automatic positional assignments.
    """

    def __init__(self, settings: MatchSettings) -> None:
        """Inject geometric tolerances for the target capture setup."""
        self.settings = settings

    @property
    def signature(self) -> str:
        """Return the matching policy identity used to version draft outputs."""
        return hashlib.sha256(json.dumps({"version": MATCH_VERSION, **asdict(self.settings)}, sort_keys=True).encode()).hexdigest()

    def match(self, sample: CaptureSample, candidates: tuple[Candidate, ...]) -> MatchResult:
        """Assign at most one candidate per usable tag without forced matches.

        Args:
            sample: Captured tags and image identity.
            candidates: Generic model detections.

        Returns:
            Accepted assignments plus issues that remain visible during review.
        """
        tags = [tag for tag in sample.tags if tag.usable]
        issues = [f"pointless_tag:{tag.instrument_uid}:{tag.instance_no}" for tag in sample.tags if tag.visible and not tag.usable]
        if not tags or not candidates:
            issues.extend(f"unmatched_tag:{tag.instrument_uid}:{tag.instance_no}" for tag in tags)
            issues.extend(f"unmatched_candidate:{candidate.index}" for candidate in candidates)
            return MatchResult((), tuple(issues))
        costs = np.array([[self._cost(tag, candidate) for candidate in candidates] for tag in tags])
        padded = np.concatenate((costs, np.full((len(tags), len(tags)), UNMATCHED_COST)), axis=1)
        rows, columns = linear_sum_assignment(padded)
        assignments = []
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if column >= len(candidates) or costs[row, column] >= UNMATCHED_COST:
                continue
            cost = costs[row, column]
            competitors = np.concatenate((np.delete(costs[row, :], column), np.delete(costs[:, column], row)))
            if competitors.size and float(competitors.min()) - cost <= self.settings.ambiguity_margin:
                issues.append(f"ambiguous:{tags[row].instrument_uid}:{tags[row].instance_no}")
                continue
            tag = tags[row]
            assignments.append(Assignment(candidates[column].index, tag.instrument_uid, tag.instance_no))
            if tag.center is not None and tag.center.status == ESTIMATED:
                issues.append(f"estimated_center:{tag.instrument_uid}:{tag.instance_no}")
        matched_tags = {(assignment.instrument_uid, assignment.instance_no) for assignment in assignments}
        matched_candidates = {assignment.candidate_index for assignment in assignments}
        issues.extend(f"unmatched_tag:{tag.instrument_uid}:{tag.instance_no}" for tag in tags if (tag.instrument_uid, tag.instance_no) not in matched_tags)
        issues.extend(f"unmatched_candidate:{candidate.index}" for candidate in candidates if candidate.index not in matched_candidates)
        return MatchResult(tuple(assignments), tuple(issues))

    def _cost(self, tag: InstanceTag, candidate: Candidate) -> float:
        """Use visible endpoint evidence without treating estimated tips as ground truth."""
        center = tag.center
        if center is None:
            return FORBIDDEN_COST
        tolerance = self.settings.max_distance * (ESTIMATED_DISTANCE_MULTIPLIER if center.status == ESTIMATED else 1)
        center_distance = candidate.mask.distance(center.x, center.y)
        if center_distance > tolerance:
            return FORBIDDEN_COST
        endpoint_distances = [candidate.mask.distance(point.x, point.y) for point in (tag.end_a, tag.end_b) if point is not None and point.status == VISIBLE]
        endpoint_cost = min(1.0, sum(endpoint_distances) / len(endpoint_distances) / tolerance) if endpoint_distances else 0.0
        return center_distance / tolerance + ENDPOINT_WEIGHT * endpoint_cost + SCORE_WEIGHT * (1 - candidate.score)
