"""一時領域のSQLとPNGで、推論原本の保存契約を検証する。"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from sam3_pipeline.domain.annotation import Candidate
from sam3_pipeline.domain.capture import CaptureError, CaptureSample
from sam3_pipeline.infrastructure.annotation_store import digest_file
from sam3_pipeline.infrastructure.matching import ArrayMask
from sam3_pipeline.infrastructure.prediction_store import SqlitePredictionStore

IMAGE_SIZE = 8
SIGNATURE = "test-model-signature"
PAIR_COUNT = 2


@pytest.fixture
def storage(tmp_path: Path) -> tuple[SqlitePredictionStore, CaptureSample]:
    """実運用データへ触れず共有ルートと中央DBを作成する。"""
    image_path = tmp_path / "session" / "image.png"
    image_path.parent.mkdir()
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)).save(image_path)
    database = tmp_path / "metadata.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE capture_sentinel(value TEXT)")
        connection.execute("INSERT INTO capture_sentinel VALUES ('unchanged')")
    sample = CaptureSample("session", "image", "declaration", 1, image_path, digest_file(image_path), IMAGE_SIZE, IMAGE_SIZE, 1, (), (), "session/image.png")
    store = SqlitePredictionStore(database, tmp_path, tmp_path / "annotations" / "sam3", 1.0)
    store.initialize()
    return store, sample


def _candidate(index: int = 0) -> Candidate:
    pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    pixels[1:4, 1:4] = True
    return Candidate(index, (1.0, 1.0, 4.0, 4.0), 0.9, ArrayMask(pixels))


def _prediction_id(sample: CaptureSample) -> str:
    return hashlib.sha256(f"{sample.sha256}:{SIGNATURE}".encode()).hexdigest()


def test_initialize_repeated_preserves_capture_tables(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """追加schemaの再実行が撮影側テーブルへ影響しないことを確認する。"""
    store, _ = storage

    store.initialize()

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT value FROM capture_sentinel").fetchall() == [("unchanged",)]
        assert connection.execute("SELECT COUNT(*) FROM sam3_schema_migrations").fetchone() == (1,)


@pytest.mark.parametrize("count", [0, 2])
def test_save_completed_roundtrip_preserves_objects(storage: tuple[SqlitePredictionStore, CaptureSample], count: int) -> None:
    """0件も完了結果として保持し、物体別maskと番号を復元する。"""
    store, sample = storage
    candidates = tuple(_candidate(index) for index in range(count))

    store.save(sample, _prediction_id(sample), candidates, SIGNATURE)
    loaded = store.load(sample, _prediction_id(sample))

    assert loaded is not None
    assert [(item.index, item.box, item.score) for item in loaded] == [(item.index, item.box, item.score) for item in candidates]
    for candidate in loaded:
        assert isinstance(candidate.mask, ArrayMask)
        assert candidate.mask.pixels.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert candidate.mask.pixels.any()
    assert not list(store.mask_root.rglob("*.png"))


def test_load_missing_result_returns_none(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """未保存と検出0件の完了を区別する。"""
    store, sample = storage

    result = store.load(sample, _prediction_id(sample))

    assert result is None


def test_save_repeated_keeps_preview_without_png(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """同一結果の保存は物体行を増やさず、承認前のPNGも作らない。"""
    store, sample = storage
    key = _prediction_id(sample)
    store.save(sample, key, (_candidate(),), SIGNATURE)
    store.save(sample, key, (_candidate(),), SIGNATURE)

    assert not list(store.mask_root.rglob("*.png"))
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_prediction_objects").fetchone() == (1,)


@pytest.mark.parametrize("field", ["score", "box", "mask"])
def test_save_changed_result_rejects_overwrite(storage: tuple[SqlitePredictionStore, CaptureSample], field: str) -> None:
    """推論IDを維持したまま結果の一部を差し替える操作を拒否する。"""
    store, sample = storage
    original = _candidate()
    different = {"score": replace(original, score=0.8), "box": replace(original, box=(0.0, 0.0, 5.0, 5.0)), "mask": replace(original, mask=ArrayMask(np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)))}[field]
    store.save(sample, _prediction_id(sample), (original,), SIGNATURE)

    with pytest.raises(CaptureError, match="immutable"):
        store.save(sample, _prediction_id(sample), (different,), SIGNATURE)

    loaded = store.load(sample, _prediction_id(sample))
    assert loaded is not None and loaded[0].box == original.box and loaded[0].score == original.score


def test_save_sparse_indices_preserves_file_numbers(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """欠番と1000以上の番号を詰め直したり切り詰めたりしない。"""
    store, sample = storage

    store.save(sample, _prediction_id(sample), (_candidate(1000), _candidate(2)), SIGNATURE)

    loaded = store.load(sample, _prediction_id(sample))
    assert loaded is not None
    assert [candidate.index for candidate in loaded] == [2, 1000]
    assert not list(store.mask_root.rglob("*.png"))


@pytest.mark.parametrize("invalid", ["empty_mask", "dimensions", "score", "nan_score", "bbox", "duplicate", "negative_index"])
def test_save_invalid_candidate_leaves_no_completed_prediction(storage: tuple[SqlitePredictionStore, CaptureSample], invalid: str) -> None:
    """不正な候補はPNGの公開前に拒否する。"""
    store, sample = storage
    candidate = _candidate()
    invalid_cases = {
        "empty_mask": (replace(candidate, mask=ArrayMask(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool))),),
        "dimensions": (replace(candidate, mask=ArrayMask(np.ones((1, 1), dtype=bool))),),
        "score": (replace(candidate, score=1.1),),
        "nan_score": (replace(candidate, score=float("nan")),),
        "bbox": (replace(candidate, box=(-1.0, 0.0, 4.0, 4.0)),),
        "duplicate": (candidate, candidate),
        "negative_index": (replace(candidate, index=-1),),
    }

    with pytest.raises(CaptureError):
        store.save(sample, _prediction_id(sample), invalid_cases[invalid], SIGNATURE)

    assert store.load(sample, _prediction_id(sample)) is None
    assert not list(store.mask_root.rglob("*.png"))


def test_load_after_restart_unapproved_mask_requires_inference(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """承認前のmaskは再起動で失われ、PNGなしで正常に未生成扱いになる。"""
    store, sample = storage
    store.save(sample, _prediction_id(sample), (_candidate(),), SIGNATURE)
    restarted = SqlitePredictionStore(store.database_path, store.data_root, store.mask_root, store.timeout)

    assert restarted.load(sample, _prediction_id(sample)) is None
    assert not list(store.mask_root.rglob("*.png"))


def test_load_after_restart_regenerates_preview_without_saving_png(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """既存maskがなくても遅延推論で表示用maskを復元し、PNGを保存しない。"""
    store, sample = storage
    store.save(sample, _prediction_id(sample), (_candidate(),), SIGNATURE)
    calls: list[str] = []

    class PreviewPredictor:
        """実モデルなしで再生成回数と推論署名を検証する。"""

        signature = SIGNATURE

        def infer(self, source: CaptureSample) -> tuple[Candidate, ...]:
            """推論対象を記録し、固定maskを生成する。"""
            calls.append(source.image_id)
            return (_candidate(),)

    restarted = SqlitePredictionStore(store.database_path, store.data_root, store.mask_root, store.timeout,
                                     predictor_factory=PreviewPredictor)

    assert restarted.load(sample, _prediction_id(sample)) is not None
    assert restarted.load(sample, _prediction_id(sample)) is not None
    assert calls == [sample.image_id]
    assert not list(store.mask_root.rglob("*.png"))


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
@pytest.mark.parametrize("regenerate", [False, True])
def test_load_missing_mask_logs_only_debug(
    storage: tuple[SqlitePredictionStore, CaptureSample], caplog: pytest.LogCaptureFixture, level: int, regenerate: bool,
) -> None:
    """実ファイル欠損の詳細はDEBUGに限定し、未生成扱いと再推論を維持する。"""
    store, sample = storage
    key = _prediction_id(sample)
    store.save(sample, key, (_candidate(),), SIGNATURE)
    relative_path = "session/missing-mask.png"
    missing_path = store.data_root / relative_path
    assert not missing_path.exists()
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("UPDATE sam3_prediction_objects SET mask_path=?", (relative_path,))
    calls: list[str] = []

    class PreviewPredictor:
        """欠損maskの再推論対象を記録する。"""

        signature = SIGNATURE

        def infer(self, source: CaptureSample) -> tuple[Candidate, ...]:
            """固定maskを返し、実モデルを使わず再推論を検証する。"""
            calls.append(source.image_id)
            return (_candidate(),)

    restarted = SqlitePredictionStore(
        store.database_path, store.data_root, store.mask_root, store.timeout,
        predictor_factory=PreviewPredictor if regenerate else None,
    )
    logger_name = "sam3_pipeline.infrastructure.prediction_store"
    reason = "保存済みmaskがないため確認時に再推論が必要である"
    with caplog.at_level(level, logger=logger_name):
        loaded = restarted.load(sample, key)
        cached = restarted.load(sample, key)

    messages = [record for record in caplog.records if record.name == logger_name and reason in record.getMessage()]
    if level == logging.INFO:
        assert not messages
    else:
        assert messages
        assert all(record.levelno == logging.DEBUG and str(missing_path) in record.getMessage() for record in messages)
    if regenerate:
        assert loaded is not None and cached is loaded
        assert calls == [sample.image_id]
        assert isinstance(loaded[0].mask, ArrayMask)
        np.testing.assert_array_equal(loaded[0].mask.pixels, _candidate().mask.pixels)
        assert any(record.levelno == logging.INFO and "確認用maskを再推論する" in record.getMessage() for record in caplog.records)
    else:
        assert loaded is None and cached is None
        assert not calls
    assert not list(store.mask_root.rglob("*.png"))


def test_save_png_failure_retry_publishes_only_complete_result(storage: tuple[SqlitePredictionStore, CaptureSample], monkeypatch: pytest.MonkeyPatch) -> None:
    """2枚目の書き込み失敗時にSQLを公開せず、残った1枚を再実行で照合する。"""
    store, sample = storage
    original_save = Image.Image.save
    calls = 0

    def failing_save(image: Image.Image, *args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == PAIR_COUNT:
            raise OSError("simulated PNG failure")
        original_save(image, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Image.Image, "save", failing_save)
        with pytest.raises(CaptureError, match="simulated PNG failure"):
            store.save(sample, _prediction_id(sample), (_candidate(0), _candidate(1)), SIGNATURE)

    assert store.load(sample, _prediction_id(sample)) is None
    store.save(sample, _prediction_id(sample), (_candidate(0), _candidate(1)), SIGNATURE)
    loaded = store.load(sample, _prediction_id(sample))
    assert loaded is not None and len(loaded) == PAIR_COUNT
    assert not list(store.mask_root.rglob("*.tmp"))


def test_save_sql_failure_retry_preserves_png(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """SQL登録失敗時も完了結果を公開せず、安全に再登録できる。"""
    store, sample = storage
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("CREATE TRIGGER reject_prediction BEFORE INSERT ON sam3_predictions BEGIN SELECT RAISE(ABORT, 'simulated SQL failure'); END")

    with pytest.raises(CaptureError, match="simulated SQL failure"):
        store.save(sample, _prediction_id(sample), (_candidate(),), SIGNATURE)

    assert store.load(sample, _prediction_id(sample)) is None
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TRIGGER reject_prediction")
    store.save(sample, _prediction_id(sample), (_candidate(),), SIGNATURE)
    assert store.load(sample, _prediction_id(sample)) is not None


def test_load_changed_source_rejects_prediction(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """撮影原本が変更された場合は以前の推論を再利用しない。"""
    store, sample = storage
    store.save(sample, _prediction_id(sample), (_candidate(),), SIGNATURE)
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="red").save(sample.image_path)

    with pytest.raises(CaptureError, match="source hash"):
        store.load(sample, _prediction_id(sample))


@pytest.mark.parametrize("path", ["../outside.png", "/outside.png", "session\\image.png"])
def test_save_unsafe_source_rejects_path(storage: tuple[SqlitePredictionStore, CaptureSample], path: str) -> None:
    """SQLへ持ち込む相対パスのルート外参照を拒否する。"""
    store, sample = storage
    unsafe = replace(sample, source_path=path)

    with pytest.raises(CaptureError, match="Unsafe"):
        store.save(unsafe, _prediction_id(sample), (_candidate(),), SIGNATURE)


def test_save_wrong_signature_rejects_prediction(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """画像hashとモデル署名に対応しないIDを拒否する。"""
    store, sample = storage

    with pytest.raises(CaptureError, match="model signature"):
        store.save(sample, "a" * 64, (_candidate(),), SIGNATURE)

    assert store.load(sample, "a" * 64) is None


def test_initialize_future_schema_rejects_downgrade(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """未対応schemaを既知の版として扱わない。"""
    store, _ = storage
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("INSERT INTO sam3_schema_migrations(version) VALUES (999)")

    with pytest.raises(CaptureError, match="Unsupported"):
        store.initialize()

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sam3_schema_migrations").fetchone() == (2,)


def test_constructor_external_mask_root_rejects_destination(tmp_path: Path) -> None:
    """画像の共有ルート外にマスクを書き出さない。"""
    with pytest.raises(CaptureError, match="Mask root"):
        SqlitePredictionStore(tmp_path / "metadata.sqlite3", tmp_path / "data", tmp_path / "outside", 1.0)


def test_save_model_metadata_records_recipe(storage: tuple[SqlitePredictionStore, CaptureSample]) -> None:
    """非機密のモデル設定を署名とともに保存する。"""
    store, sample = storage
    metadata = {"prompt": "metal tool", "min_score": 0.5}
    configured = SqlitePredictionStore(store.database_path, store.data_root, store.mask_root, store.timeout, model_metadata=metadata)

    configured.save(sample, _prediction_id(sample), (_candidate(),), SIGNATURE)

    with sqlite3.connect(store.database_path) as connection:
        value = connection.execute("SELECT model_metadata_json FROM sam3_predictions").fetchone()[0]
        assert json.loads(value) == metadata
