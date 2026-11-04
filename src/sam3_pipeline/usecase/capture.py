"""推論原本の再利用とタグ依存のレビュー版を分離して、既存注釈の上書きを防ぐ。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from sam3_pipeline.domain.annotation import AnnotationStore, InstanceMatcher, PredictionStore, Predictor
from sam3_pipeline.domain.capture import CaptureSample, CaptureSource


def annotate(source: CaptureSource, predictor: Predictor, matcher: InstanceMatcher, store: AnnotationStore, limit: int | None = None, *, prediction_store: PredictionStore | None = None) -> list[str]:
    """撮影スナップショットから草案を生成し、タグだけの変更では推論を再利用する。

    Args:
        source: 検証済み撮影スナップショットの取得元。
        predictor: 共通テキストを使う推論アダプター。
        matcher: 曖昧な器具対応を確定しない対応付け方針。
        store: 旧キャッシュと版管理された注釈の保存先。
        limit: 処理する画像数の上限。
        prediction_store: 推論原本の保存先。省略時は従来のキャッシュを使う。

    Returns:
        生成または再利用されたレビュー版の識別子。

    Raises:
        CaptureError: 入力・推論結果・成果物の検証に失敗した場合。
    """
    return annotate_samples(source.iter_samples(), predictor, matcher, store, limit, prediction_store=prediction_store)


def annotate_samples(samples: Iterable[CaptureSample], predictor: Predictor, matcher: InstanceMatcher, store: AnnotationStore, limit: int | None = None, *, prediction_store: PredictionStore | None = None) -> list[str]:
    """推論開始前に固定した撮影スナップショットだけを処理する。

    Args:
        samples: 呼び出し元が固定したSQL対象サンプル列。
        predictor: 共通テキスト推論アダプター。
        matcher: タグと候補の対応付け方針。
        store: 候補キャッシュと版管理成果物の保存先。
        limit: 処理する画像数の上限。
        prediction_store: 推論原本の保存先。省略時は従来のキャッシュを使う。

    Returns:
        生成または再利用されたレビューリビジョンID。
    """
    revisions = []
    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        prediction_key = hashlib.sha256(f"{sample.sha256}:{predictor.signature}".encode()).hexdigest()
        candidates = prediction_store.load(sample, prediction_key) if prediction_store is not None else store.load_candidates(prediction_key)
        if candidates is None:
            if prediction_store is None:
                candidates = predictor.infer(sample)
                store.save_candidates(prediction_key, candidates)
            else:
                candidates = store.load_candidates(prediction_key)
                legacy_cache = candidates is not None
                if candidates is None:
                    candidates = predictor.infer(sample)
                prediction_store.save(sample, prediction_key, candidates, None if legacy_cache else predictor.signature)
        result = matcher.match(sample, candidates)
        revisions.append(store.save_draft(sample, candidates, result, prediction_key, matcher.signature))
    return revisions
