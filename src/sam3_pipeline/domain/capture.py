"""Immutable capture contracts keep tagging identity separate from model predictions."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SCHEMA_VERSION = 1
VISIBLE = "visible"
ESTIMATED = "estimated"
TAGGED = "tagged"
NONE_SOURCE = "none"
NORMALIZED_MIN = 0.0
NORMALIZED_MAX = 1.0


class CaptureError(Exception):
    """Reject inconsistent inputs rather than silently generating training labels."""


@dataclass(frozen=True)
class TagPoint:
    """Store normalized coordinates so display resolution cannot alter identity."""

    x: float
    y: float
    status: str = VISIBLE

    def __post_init__(self) -> None:
        """Reject invalid coordinates before any pixel-space matching."""
        if self.status not in (VISIBLE, ESTIMATED) or not all(math.isfinite(value) and NORMALIZED_MIN <= value <= NORMALIZED_MAX for value in (self.x, self.y)):
            raise CaptureError("Invalid normalized tag point")


@dataclass(frozen=True)
class InstanceTag:
    """Preserve capture instance identity independently of inference ordering."""

    instrument_uid: str
    instrument_name: str
    instance_no: int
    visible: bool
    hint_source: str
    center: TagPoint | None
    end_a: TagPoint | None
    end_b: TagPoint | None
    occluded: bool

    @property
    def usable(self) -> bool:
        """Return whether positional evidence exists for an in-frame instrument."""
        return self.visible and self.hint_source != NONE_SOURCE and self.center is not None


@dataclass(frozen=True)
class CaptureSample:
    """SQL上の撮影データとタグ版を固定した、アノテーション処理の入力契約である。"""

    session_id: str
    image_id: str
    declaration_id: str
    arrangement_no: int
    image_path: Path
    sha256: str
    width: int
    height: int
    tag_revision: int
    tags: tuple[InstanceTag, ...]
    instruments: tuple[tuple[str, str], ...] = ()
    source_path: str = ""

    @property
    def key(self) -> str:
        """Return a globally usable identity without trusting filename uniqueness."""
        return f"{self.session_id}__{self.image_id}"


class CaptureSource(Protocol):
    """撮影SQLを正本とし、処理時点のデータと現行性を提供する境界である。"""

    def iter_samples(self) -> Iterator[CaptureSample]:
        """現行のtagged画像を検証して返す。

        Returns:
            SQLに存在するタグ版固定済みサンプル。

        Raises:
            CaptureError: 撮影データのメタデータが不整合である場合。
        """
        ...

    def current_sample(self, session_id: str, image_id: str, source_path: str) -> CaptureSample | None:
        """指定画像の現行SQLスナップショットを返す。

        Args:
            session_id: 撮影セッションID。
            image_id: SQL上の画像ID。
            source_path: SQLに記録されたdataルート相対パス。

        Returns:
            現行のtagged画像。削除済み、未タグ、またはパス変更済みならNone。

        Raises:
            CaptureError: SQLレコードや画像内容が不正である場合。
        """
        ...

    def is_current(self, sample: CaptureSample) -> bool:
        """SQL行が同じ画像・タグ版を指し、元画像が現存するか確認する。

        Args:
            sample: 過去に取得した撮影スナップショット。

        Returns:
            SQL上で同一のtaggedレコードと画像ファイルが現存する場合はTrue。
        """
        ...


class AnnotationLedger(Protocol):
    """Record review provenance in the authoritative capture metadata database."""

    def upsert_annotation_revision(
        self,
        *,
        session_id: str,
        image_id: str,
        tag_revision: int,
        revision: str,
        status: str,
        annotation_path: str,
        annotation_sha256: str,
        approval_id: str | None = None,
        reviewer: str | None = None,
        approved_at: str | None = None,
    ) -> None:
        """Persist one inference, review, approval, or revocation state."""
        ...

    def mark_annotation_exported(
        self,
        *,
        session_id: str,
        image_id: str,
        revision: str,
        manifest_path: str,
    ) -> None:
        """Link an approved annotation revision to its published manifest."""
        ...
