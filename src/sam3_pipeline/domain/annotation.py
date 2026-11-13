"""Annotation ports separate geometric evidence, inference, and artifact ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .capture import CaptureSample

UNRESOLVED_LABEL = "__unresolved__"
RECTANGLE = "rectangle"


class MaskGeometry(Protocol):
    """Expose normalized distances without coupling business contracts to NumPy."""

    def distance(self, x: float, y: float) -> float:
        """Return distance to visible foreground divided by the image diagonal.

        Args:
            x: Normalized horizontal position.
            y: Normalized vertical position.

        Returns:
            Nonnegative distance; zero means foreground containment.
        """
        ...


@dataclass(frozen=True)
class Candidate:
    """Keep generic model detections independent of surgical instrument classes."""

    index: int
    box: tuple[float, float, float, float]
    score: float
    mask: MaskGeometry


@dataclass(frozen=True)
class Assignment:
    """Attach a capture UID only when positional evidence is sufficiently unique."""

    candidate_index: int
    instrument_uid: str
    instance_no: int


@dataclass(frozen=True)
class MatchResult:
    """Retain uncertainty instead of turning unmatched hints into background."""

    assignments: tuple[Assignment, ...]
    issues: tuple[str, ...]


class Predictor(Protocol):
    """Support one generic text inference per image and independent cache identity."""

    @property
    def signature(self) -> str:
        """Return a fingerprint of weights, implementation, prompt, and settings."""
        ...

    def infer(self, sample: CaptureSample) -> tuple[Candidate, ...]:
        """Infer generic objects without using class-specific tag names.

        Args:
            sample: Capture image and dimensions.

        Returns:
            All candidates passing the configured model threshold.
        """
        ...


class InstanceMatcher(Protocol):
    """Allow matching policy changes without rerunning the model."""

    @property
    def signature(self) -> str:
        """Return a fingerprint of the matching policy and thresholds."""
        ...

    def match(self, sample: CaptureSample, candidates: tuple[Candidate, ...]) -> MatchResult:
        """Match tags to candidates while retaining unresolved objects.

        Args:
            sample: Source instance tags.
            candidates: Generic mask detections.

        Returns:
            Conservative one-to-one assignments and review issues.
        """
        ...


class PredictionStore(Protocol):
    """推論原本の保存先を抽象化し、レビュー注釈の編集から分離する。"""

    def load(self, sample: CaptureSample, prediction_id: str) -> tuple[Candidate, ...] | None:
        """画像に対応する完了済み推論を取得する。

        Args:
            sample: 呼び出し元が検証した撮影スナップショット。
            prediction_id: 画像hashと推論署名から決定したID。

        Returns:
            完了結果。0件は空tuple、未保存はNoneである。

        Raises:
            CaptureError: 保存済み結果や原本の整合性が失われている場合。
        """
        ...

    def save(self, sample: CaptureSample, prediction_id: str, candidates: tuple[Candidate, ...], model_signature: str | None) -> None:
        """不変の推論結果を、0件も含めて公開する。

        Args:
            sample: 保存対象の撮影スナップショット。
            prediction_id: 画像hashと推論署名から決定したID。
            candidates: 番号を固定した候補列。
            model_signature: 推論署名。旧キャッシュからの移行時のみNoneである。

        Raises:
            CaptureError: 原本の変更、不正な候補、または保存失敗が発生した場合。
        """
        ...


class ObjectReviewStore(Protocol):
    """レビューと物体確認の永続化を抽象化し、推論原本の変更を防ぐ。"""

    def sync_revision(self, sample: CaptureSample, revision: str, metadata: dict[str, Any], annotation: dict[str, Any]) -> None:
        """現行画像・推論・器具対応を固定したレビュー版を登録する。

        Args:
            sample: 原本を照合済みの撮影スナップショット。
            revision: 最新レビュー版。
            metadata: 推論キーと器具対応を含む草案。
            annotation: YOLO用の編集済み注釈。

        Raises:
            CaptureError: 推論・器具対応が不正、または保存に失敗した場合。
        """
        ...

    def record_annotation(
        self,
        revision: str,
        annotation: dict[str, Any],
        status: str,
        approval: dict[str, Any] | None = None,
        *,
        decisions: dict[int, str] | None = None,
        sample: CaptureSample | None = None,
        overwrite_masks: bool = False,
        assignments: dict[int, tuple[str, int] | None] | None = None,
        decision_reviewer: str | None = None,
    ) -> None:
        """編集済みYOLO注釈と画像承認を同期する。

        Args:
            revision: 対象レビュー版。
            annotation: YOLO用の編集済み注釈。
            status: 確認・承認・取消・公開状態。
            approval: 明示承認の記録。
            decisions: 全物体の確定採否。同時指定時は画像承認と原子的に記録する。
            sample: 一括確定時に原本と撮影SQLを再照合する画像。
            overwrite_masks: 利用者が既存の承認用maskの置換に同意したか。
            assignments: DINO候補ごとの手動タグ対応。省略時は現在の対応を維持する。
            decision_reviewer: DINO確認の担当者。既存のYOLO承認者を変更しないために分離する。

        Raises:
            CaptureError: 対象版が存在しないか、記録に失敗した場合。
        """
        ...

    def objects(self, sample: CaptureSample, revision: str) -> list[dict[str, Any]]:
        """検証済み推論原本と現在の物体別判定を返す。

        Args:
            sample: 現行性を検証済みの撮影データ。
            revision: 最新レビュー版。

        Returns:
            元bbox・maskパス・器具対応・採否を持つ物体列。

        Raises:
            CaptureError: 旧版または原本不整合の場合。
        """
        ...

    def decide(self, sample: CaptureSample, revision: str, object_index: int, *, status: str, reviewer: str) -> None:
        """元bbox・maskに対する担当者の採否を履歴付きで記録する。

        Args:
            sample: 現行性を検証済みの撮影データ。
            revision: 最新レビュー版。
            object_index: 固定した推論候補番号。
            status: pending、accepted、rejectedのいずれか。
            reviewer: 確認担当者。

        Raises:
            CaptureError: 不正な要求、旧版、器具未確定、または原本不整合の場合。
        """
        ...

    def reject_all(self, sample: CaptureSample, revision: str, *, reviewer: str) -> None:
        """画像承認を行わず、現行候補すべての不採用を一括記録する。

        Args:
            sample: 現行性を検証済みの撮影データ。
            revision: 最新レビュー版。
            reviewer: 確認担当者。

        Raises:
            CaptureError: 担当者、レビュー版、または候補一覧が不正な場合。
        """
        ...

    def mask_png(self, sample: CaptureSample, revision: str, object_index: int) -> bytes:
        """元maskを透明背景の重ね表示用PNGとして返す。

        Args:
            sample: 現行性を検証済みの撮影データ。
            revision: 最新レビュー版。
            object_index: 固定した推論候補番号。

        Returns:
            物体領域だけに色を付けたRGBA PNG。

        Raises:
            CaptureError: 旧版、不明物体、または原本不整合の場合。
        """
        ...

    def progress(self) -> dict[str, dict[str, Any]]:
        """レビュー版ごとの一覧確認状態を返す。

        Returns:
            レビュー版をキーとする確認状態。

        Raises:
            CaptureError: 読み取りに失敗した場合。
        """
        ...

    def set_progress(self, revision: str, state: dict[str, Any]) -> None:
        """レビュー版の一覧確認状態を保存する。

        Args:
            revision: 最新レビュー版。
            state: 一覧確認・空画像確認・例外理由。

        Raises:
            CaptureError: 旧版または記録失敗の場合。
        """
        ...

    def approved(self, revision: str, *, approval_id: str | None = None, scope: str | None = None) -> bool:
        """画像承認が現行SQL上で有効かを返す。

        Args:
            revision: 対象レビュー版。
            approval_id: 指定時はこの承認IDとの一致も要求する。
            scope: yolo指定時はDINO専用承認を除外する。

        Returns:
            現行版に有効な画像承認がある場合はTrue。

        Raises:
            CaptureError: 読み取りに失敗した場合。
        """
        ...


class AnnotationStore(Protocol):
    """Own cache and review files independently of the inference use case."""

    def load_candidates(self, key: str) -> tuple[Candidate, ...] | None:
        """Return cached candidates, or None when no completed cache exists."""
        ...

    def save_candidates(self, key: str, candidates: tuple[Candidate, ...]) -> None:
        """Persist a completed candidate set, including zero-detection results."""
        ...

    def save_draft(self, sample: CaptureSample, candidates: tuple[Candidate, ...], result: MatchResult, prediction_key: str, matching_signature: str) -> str:
        """Create a revision without overwriting an existing reviewer workspace.

        Returns:
            The immutable source revision identifier.
        """
        ...
