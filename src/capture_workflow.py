"""セッション別SQLiteを使うSAM3推論・レビュー・公開を組み立てる。

責務:
    設定と実行段階から保存先・推論・レビューを接続し、DB操作自体は実装しない。
処理の流れ:
    1. 設定から共有データと作業領域を解決する。
    2. 各セッションの推論・レビューschemaを初期化する。
    3. 指定された推論・確認・承認・公開処理を実行する。
関連ファイル:
    configs/capture.json、infrastructure/、usecase/capture.py。
設計意図:
    既存の実行段階を維持し、各保存ストアへの共有ルート注入でセッション境界を揃える。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from sam3_pipeline.domain.capture import CaptureError
from sam3_pipeline.domain.confidence import ConfidenceThresholds
from sam3_pipeline.infrastructure.annotation_ledger import SqliteAnnotationLedger
from sam3_pipeline.infrastructure.annotation_store import FileAnnotationStore, read_json
from sam3_pipeline.infrastructure.capture_predictor import Sam3CapturePredictor
from sam3_pipeline.infrastructure.capture_source import SqliteCaptureSource
from sam3_pipeline.infrastructure.matching import MatchSettings, ScipyInstanceMatcher
from sam3_pipeline.infrastructure.object_review_store import SqliteObjectReviewStore
from sam3_pipeline.infrastructure.prediction_store import SqlitePredictionStore
from sam3_pipeline.infrastructure.review_server import BatchReviewServer
from sam3_pipeline.infrastructure.selection_server import REVIEW_RETURN_QUERY, SelectiveInferenceServer
from sam3_pipeline.usecase.capture import annotate_samples

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "capture.json"
DEFAULT_SQLITE_TIMEOUT = 10.0
DEFAULT_MASK_DIRECTORY = "annotations/sam3"


def _path(config: dict[str, Any], name: str, base: Path) -> Path:
    """Resolve configurable paths without depending on the working directory."""
    return (base / str(config[name])).resolve()


def main(argv: list[str] | None = None) -> int:
    """Run one explicitly selected capture workflow stage.

    Args:
        argv: Optional command arguments; defaults to the process arguments.

    Returns:
        Zero on success; exceptions are reported through argparse for CLI use.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("run", aliases=["annotate"])
    infer.add_argument("--limit", type=int)
    commands.add_parser("review")
    approve = commands.add_parser("approve")
    approve.add_argument("revision")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--confirm-complete", action="store_true", required=True)
    approve.add_argument("--allow-empty", action="store_true")
    approve.add_argument("--exception-note", default="")
    revoke = commands.add_parser("revoke")
    revoke.add_argument("revision")
    commands.add_parser("status")
    commands.add_parser("migrate-predictions", help="旧NPZをSQLと物体別PNGへ非破壊移行する")
    export = commands.add_parser("export")
    export.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config_path = arguments.config.resolve()
    config = read_json(config_path)
    use_yolo = config.get("use_yolo", False)
    yolo_confidence = config.get("yolo_confidence", 0.8)
    if not isinstance(use_yolo, bool):
        raise CaptureError("capture.json use_yolo must be a boolean")
    confidence_is_number = isinstance(yolo_confidence, (int, float)) and not isinstance(yolo_confidence, bool)
    confidence_is_valid = confidence_is_number and math.isfinite(yolo_confidence) and 0 <= yolo_confidence <= 1
    if not confidence_is_valid:
        raise CaptureError("capture.json yolo_confidence must be a finite number from 0 to 1")
    base = config_path.parent
    root = _path(config, "capture_root", base)
    output = _path(config, "output_root", base)
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise CaptureError("Annotation artifacts and capture originals must use separate roots")
    targets = config.get("targets")
    if "targets" in config and targets is None:
        raise CaptureError("capture.json targets must be a non-empty list")
    source = SqliteCaptureSource(root, timeout=config.get("sqlite_timeout", DEFAULT_SQLITE_TIMEOUT), targets=targets)
    scoped_databases = source.databases() if "targets" in config else None
    ledger = SqliteAnnotationLedger(root, config.get("sqlite_timeout", DEFAULT_SQLITE_TIMEOUT), scoped_databases)
    store = FileAnnotationStore(output, source, ledger)
    prediction_store: SqlitePredictionStore | None = None
    reviews: SqliteObjectReviewStore | None = None
    if arguments.command in {"run", "annotate", "migrate-predictions", "review", "approve", "revoke", "export"}:
        mask_root = _path(config, "mask_root", base) if config.get("mask_root") else root / DEFAULT_MASK_DIRECTORY
        model_metadata = {name: config.get(name) for name in ("checkpoint", "bpe_path", "prompt", "min_score")}
        model_metadata.update({"use_yolo": use_yolo, "yolo_confidence": yolo_confidence, "yolo_weights_sha256": None})
        prediction_store = SqlitePredictionStore(root, root, mask_root, config.get("sqlite_timeout", DEFAULT_SQLITE_TIMEOUT), model_metadata=model_metadata)
        predictor: Sam3CapturePredictor | None = None

        def preview_predictor() -> Sam3CapturePredictor:
            """確認用maskが必要になった時だけSAM3モデルを生成する。

            Returns:
                このプロセス内で共用する推論器。
            """
            nonlocal predictor
            if predictor is None:
                load_dotenv(Path(__file__).parent / ".env")
                checkpoint = _path(config, "checkpoint", base) if config.get("checkpoint") else None
                predictor = Sam3CapturePredictor(
                    checkpoint,
                    config["prompt"],
                    config["min_score"],
                    _path(config, "bpe_path", base) if config.get("bpe_path") else None,
                    use_yolo=use_yolo,
                    yolo_confidence=yolo_confidence,
                )
                if prediction_store is not None:
                    prediction_store.model_metadata = predictor.model_metadata
            return predictor

        prediction_store.predictor_factory = preview_predictor
        if scoped_databases is None:
            prediction_store.initialize()
        else:
            prediction_store.initialize(scoped_databases)
        reviews = SqliteObjectReviewStore(root, config.get("sqlite_timeout", DEFAULT_SQLITE_TIMEOUT), prediction_store)
        if scoped_databases is None:
            reviews.initialize()
        else:
            reviews.initialize(scoped_databases)
        if arguments.command == "migrate-predictions":
            store.migrate_predictions(prediction_store)
        store.review_store = reviews
        if arguments.command not in {"run", "annotate"}:
            store.sync_object_reviews()
    if arguments.command in {"run", "annotate"}:
        if arguments.limit is not None and arguments.limit <= 0:
            raise CaptureError("Limit must be positive")
        try:
            predictor = preview_predictor()
            predictor.initialize()
        except CaptureError as error:
            if "YOLO" in str(error):
                raise
            LOGGER.exception("SAM3 model initialization failed")
            raise CaptureError("SAM3 model initialization failed") from error
        except Exception as error:
            LOGGER.exception("SAM3モデルの初期化に失敗しました")
            raise CaptureError("SAM3モデル初期化に失敗したため、推論を開始できません") from error
        matcher = ScipyInstanceMatcher(MatchSettings(**config["matching"]))
        assert prediction_store is not None
        assert reviews is not None
        selection = SelectiveInferenceServer(source, store, reviews)
        thresholds = ConfidenceThresholds(**config["review_confidence"])
        try:
            while True:
                selected = selection.select()
                if arguments.limit is not None:
                    if len(selected) > arguments.limit:
                        LOGGER.warning("選択した %s 件のうち --limit により先頭 %s 件だけを処理します", len(selected), arguments.limit)
                    selected = selected[: arguments.limit]
                store.sync_object_reviews(source_paths=set(selected))
                selected_samples = []
                for source_path in selected:
                    item = selection.entry_by_source_path[source_path]
                    sample = source.current_sample(item["session_id"], item["image_id"], source_path)
                    if sample is not None:
                        selected_samples.append(sample)
                samples = tuple(selected_samples)
                if len(samples) != len(selected):
                    raise CaptureError("選択した画像の一部は撮影SQL上で削除または変更されています。推論を中止しました")
                revisions = annotate_samples(samples, predictor, matcher, store, arguments.limit, prediction_store=prediction_store)
                LOGGER.info("Review revisions: %s", json.dumps(revisions))
                selection.begin_review()
                assert selection.url is not None
                review = BatchReviewServer(store, thresholds, samples)
                return_url = f"{selection.url}?{REVIEW_RETURN_QUERY}"
                review.serve(selection_return_url=return_url, stop_event=selection.review_complete)
                if not selection.review_complete.is_set():
                    break
        finally:
            selection.close()
    elif arguments.command == "migrate-predictions":
        assert prediction_store is not None
        LOGGER.info("推論結果の移行処理件数: %s", store.migrate_predictions(prediction_store))
    elif arguments.command == "review":
        thresholds = ConfidenceThresholds(**config["review_confidence"])
        samples = tuple(source.iter_samples())
        BatchReviewServer(store, thresholds, samples).serve()
    elif arguments.command == "approve":
        approval_id = store.approve(arguments.revision, arguments.reviewer, confirm_complete=arguments.confirm_complete, allow_empty=arguments.allow_empty, exception_note=arguments.exception_note)
        LOGGER.info("Approved: %s", approval_id)
    elif arguments.command == "revoke":
        store.revoke(arguments.revision)
    elif arguments.command == "status":
        store.sync_existing_exports()
        sample_by_identity = {(sample.session_id, sample.image_id): sample for sample in source.iter_samples()}
        for path in sorted((output / "drafts").glob("*/metadata.json")):
            data = read_json(path)
            source_data = data["source"]
            sample = sample_by_identity.get((source_data["session_id"], source_data["image_id"]))
            if sample is None or not store.is_current_snapshot(source_data, sample):
                continue
            LOGGER.info(
                "revision=%s image=%s tag_revision=%s approved=%s issues=%s", path.parent.name, data["source"]["image_id"], data["source"]["tag_revision"], (output / "approvals" / f"{path.parent.name}.json").exists(), data["matching"]["issues"]
            )
    else:
        LOGGER.info("Approved batch: %s", store.export_approved(arguments.destination))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
