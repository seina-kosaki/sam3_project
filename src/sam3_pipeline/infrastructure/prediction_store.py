"""推論情報をSQLiteへ記録し、承認前のmaskをメモリー上で保持する。

責務:
    推論bbox・score・モデル情報と確認用maskを扱い、正式PNGの保存は行わない。
処理の流れ:
    1. 所属セッションDBを特定し、元画像と候補を検証する。
    2. 候補のbbox・score・maskを検証する。
    3. 推論情報をSQLへ記録し、確認用maskをメモリーへ保持する。
関連ファイル:
    session_database.py、object_review_store.py、usecase/capture.py。
設計意図:
    利用者が承認するまでPNGを書き出さず、再起動時は必要に応じて再推論する。

Note:
    メモリー上のmaskはプロセス終了で失われる。正式保存は承認処理の責務である。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path, PurePosixPath
from threading import RLock

import numpy as np
from PIL import Image

from sam3_pipeline.domain.annotation import Candidate, Predictor
from sam3_pipeline.domain.capture import CaptureError, CaptureSample
from sam3_pipeline.infrastructure.annotation_store import digest_file
from sam3_pipeline.infrastructure.matching import ArrayMask
from sam3_pipeline.infrastructure.session_database import metadata_databases, session_database

SCHEMA_VERSION = 1
HASH_LENGTH = 64
MASK_FOREGROUND = 255
MASK_MODE = "L"
MASK_FORMAT = "PNG"
PREVIEW_PREFIX = "memory:"
LOGGER = logging.getLogger(__name__)
BBOX_COORDINATES = 4
SCHEMA = """
CREATE TABLE IF NOT EXISTS sam3_schema_migrations (
    version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sam3_predictions (
    session_id TEXT NOT NULL, image_id TEXT NOT NULL, prediction_id TEXT NOT NULL,
    source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL,
    width INTEGER NOT NULL CHECK(width > 0), height INTEGER NOT NULL CHECK(height > 0),
    model_signature TEXT, model_metadata_json TEXT NOT NULL,
    provenance TEXT NOT NULL CHECK(provenance IN ('native', 'legacy_npz')),
    object_count INTEGER NOT NULL CHECK(object_count >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(session_id, image_id, prediction_id)
);
CREATE TABLE IF NOT EXISTS sam3_prediction_objects (
    session_id TEXT NOT NULL, image_id TEXT NOT NULL, prediction_id TEXT NOT NULL,
    object_index INTEGER NOT NULL CHECK(object_index >= 0),
    bbox_json TEXT NOT NULL, sam3_score REAL NOT NULL CHECK(sam3_score >= 0 AND sam3_score <= 1),
    mask_path TEXT NOT NULL UNIQUE, mask_sha256 TEXT NOT NULL,
    PRIMARY KEY(session_id, image_id, prediction_id, object_index),
    FOREIGN KEY(session_id, image_id, prediction_id)
        REFERENCES sam3_predictions(session_id, image_id, prediction_id)
);
"""


class SqlitePredictionStore:
    """推論情報と確認用maskを分離し、承認前のPNG出力を防ぐ。

    Args:
        database_path: 共有データルート、またはSAM3専用テーブルを追加するセッションDB。
        data_root: SQLの相対パスを解決する共有データルート。
        mask_root: 旧PNG参照との設定互換用ルート。新規PNGは出力しない。
        timeout: SQLiteのロック待ち秒数。
        model_metadata: 署名を補足するモデル取得元・prompt・閾値等の非機密情報。
        predictor_factory: 確認用maskがメモリーにない場合の遅延モデル生成。

    Note:
        完了行の存在は0件を含む推論完了を示す。DINO採否とレビューは後続層の責務である。
        既存PNGは読み取り専用で扱い、欠損時は確認用の再推論を行う。
    """

    def __init__(self, database_path: Path, data_root: Path, mask_root: Path, timeout: float, *, model_metadata: Mapping[str, object] | None = None, predictor_factory: Callable[[], Predictor] | None = None) -> None:
        """設定済みの保存先を束縛し、共有ルート外へのマスク保存を拒否する。"""
        self.database_path = database_path.resolve()
        self.data_root = data_root.resolve()
        self.mask_root = mask_root.resolve()
        self.timeout = timeout
        self.model_metadata = dict(model_metadata or {})
        self.predictor_factory = predictor_factory
        self._previews: dict[tuple[str, str, str], tuple[Candidate, ...]] = {}
        self._preview_lock = RLock()
        if not self.mask_root.is_relative_to(self.data_root):
            raise CaptureError("Mask root must be inside data root")

    def initialize(self) -> None:
        """既存テーブルを変更せず、SAM3専用schemaを冪等に追加する。

        Raises:
            CaptureError: DBが存在しない、未対応版、またはSQLite操作に失敗した場合。
        """
        # 共有ルートにはDBを作らず、存在するセッションDBにだけschemaを追加する。
        if self.database_path.is_dir():
            for database in metadata_databases(self.database_path):
                SqlitePredictionStore(database, self.data_root, self.mask_root, self.timeout, model_metadata=self.model_metadata).initialize()
            return
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sam3_schema_migrations'").fetchone()
            if exists:
                versions = [row[0] for row in connection.execute("SELECT version FROM sam3_schema_migrations")]
                if any(version != SCHEMA_VERSION for version in versions):
                    raise CaptureError("Unsupported SAM3 prediction schema version")
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute("INSERT OR IGNORE INTO sam3_schema_migrations(version) VALUES (?)", (SCHEMA_VERSION,))

    def load(self, sample: CaptureSample, prediction_id: str) -> tuple[Candidate, ...] | None:
        """確認用maskをメモリーまたは検証済みPNGから取得し、必要なら再推論する。

        Args:
            sample: 呼び出し元が検証した撮影スナップショット。
            prediction_id: 画像hashと推論署名から決定したID。

        Returns:
            候補列。0件は空tuple、未生成で再推論器もない場合はNoneである。

        Raises:
            CaptureError: 画像・保存済み結果・maskが不整合である場合。
        """
        # GPU推論を直列化し、同時のmask表示要求でモデルを重複実行しない。
        with self._preview_lock:
            with self._connection(sample) as connection:
                self._validate_source(sample, prediction_id)
                result = self._load(connection, sample, prediction_id)
                header = connection.execute("SELECT model_signature FROM sam3_predictions WHERE session_id=? AND image_id=? AND prediction_id=?",
                                            self._identity(sample, prediction_id)).fetchone()
            if result is not None or header is None or self.predictor_factory is None:
                return result
            predictor = self.predictor_factory()
            expected = hashlib.sha256(f"{sample.sha256}:{predictor.signature}".encode()).hexdigest()
            if expected != prediction_id:
                raise CaptureError("確認用maskの推論設定が現在と異なる。runで新しいレビュー版を生成してください")
            LOGGER.info("確認用maskを再推論する: %s", sample.key)
            candidates = predictor.infer(sample)
            self.save(sample, prediction_id, candidates, predictor.signature)
            return self._previews[self._identity(sample, prediction_id)]

    def save(self, sample: CaptureSample, prediction_id: str, candidates: tuple[Candidate, ...], model_signature: str | None) -> None:
        """PNGを出力せず推論情報と確認用maskを保持する。

        Args:
            sample: 保存対象の撮影スナップショット。
            prediction_id: 画像hashと推論署名から決定したID。
            candidates: 番号・bbox・score・配列maskを持つ候補。
            model_signature: 推論設定の署名。旧NPZ移行時のみ不明を示すNone。

        Raises:
            CaptureError: 候補が不正、原本が変更済み、または保存に失敗した場合。
        """
        with self._preview_lock, self._connection(sample) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_source(sample, prediction_id)
            self._validate_candidates(sample, candidates)
            if model_signature is not None and prediction_id != hashlib.sha256(f"{sample.sha256}:{model_signature}".encode()).hexdigest():
                raise CaptureError("Prediction ID does not match model signature")
            ordered = tuple(sorted(candidates, key=lambda item: item.index))
            previous = self._load(connection, sample, prediction_id)
            if previous is not None:
                if not self._same_candidates(previous, ordered):
                    raise CaptureError("Prediction results are immutable")
                stored = connection.execute(
                    "SELECT model_signature FROM sam3_predictions WHERE session_id=? AND image_id=? AND prediction_id=?", self._identity(sample, prediction_id)
                ).fetchone()
                if stored[0] is not None and model_signature is not None and stored[0] != model_signature:
                    raise CaptureError("Prediction signature differs from stored result")
                return
            records = []
            for candidate in ordered:
                assert isinstance(candidate.mask, ArrayMask)
                stream = io.BytesIO()
                Image.fromarray(candidate.mask.pixels.astype(np.uint8) * MASK_FOREGROUND).save(stream, format=MASK_FORMAT)
                marker = f"{PREVIEW_PREFIX}{sample.session_id}:{sample.image_id}:{prediction_id}:{candidate.index}"
                records.append((*self._identity(sample, prediction_id), candidate.index, json.dumps(candidate.box), candidate.score,
                                marker, hashlib.sha256(stream.getvalue()).hexdigest()))
            self._validate_source(sample, prediction_id)
            stored = connection.execute("SELECT object_count FROM sam3_predictions WHERE session_id=? AND image_id=? AND prediction_id=?",
                                        self._identity(sample, prediction_id)).fetchone()
            # 再推論が既存のbboxやmaskと異なる場合、旧承認を同一版のまま流用しない。
            if stored is not None:
                rows = connection.execute("SELECT object_index, bbox_json, sam3_score, mask_sha256 FROM sam3_prediction_objects WHERE session_id=? AND image_id=? AND prediction_id=? ORDER BY object_index",
                                          self._identity(sample, prediction_id)).fetchall()
                comparable = [(record[3], record[4], record[5], record[7]) for record in records]
                if stored["object_count"] != len(ordered) or [tuple(row) for row in rows] != comparable:
                    raise CaptureError("再推論結果が既存の推論情報と異なる。設定を更新して新しいレビュー版を生成してください")
                self._previews[self._identity(sample, prediction_id)] = ordered
                return
            connection.execute(
                """INSERT INTO sam3_predictions(session_id, image_id, prediction_id, source_path,
                    source_sha256, width, height, model_signature, model_metadata_json, provenance, object_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (*self._identity(sample, prediction_id), sample.source_path, sample.sha256, sample.width, sample.height, model_signature,
                 json.dumps(self.model_metadata if model_signature is not None else {}, sort_keys=True, allow_nan=False), "native" if model_signature is not None else "legacy_npz", len(ordered)),
            )
            connection.executemany("INSERT INTO sam3_prediction_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records)
        # SQL確定に失敗した候補を完了済みとして返さない。
        self._previews[self._identity(sample, prediction_id)] = ordered

    @contextmanager
    def _connection(self, sample: CaptureSample | None = None) -> Iterator[sqlite3.Connection]:
        """SQLiteのロック待ちを制限し、失敗時は完了行を公開せず例外を変換する。

        Args:
            sample: 保存先セッションを特定する画像。単一DBのschema初期化時はNone。

        Yields:
            所属セッションのSQLite接続。

        Raises:
            CaptureError: 保存先が欠損、またはSQLite操作に失敗した場合。
        """
        database = session_database(self.database_path, sample.session_id) if sample is not None else self.database_path
        if not database.is_file():
            raise CaptureError(f"セッションDBが存在しない: {database}")
        try:
            with closing(sqlite3.connect(database, timeout=self.timeout)) as connection, connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                yield connection
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as error:
            raise CaptureError(f"推論保存先が不正である: {database}: {error}") from error

    @staticmethod
    def _identity(sample: CaptureSample, prediction_id: str) -> tuple[str, str, str]:
        return sample.session_id, sample.image_id, prediction_id

    def _validate_source(self, sample: CaptureSample, prediction_id: str) -> None:
        """原本のパス・hash・寸法と安全な推論識別子を検証する。"""
        if len(prediction_id) != HASH_LENGTH or any(character not in "0123456789abcdef" for character in prediction_id):
            raise CaptureError("Invalid prediction ID")
        if not sample.session_id or sample.session_id in {".", ".."} or any(character in sample.session_id for character in '/\\:'):
            raise CaptureError("Invalid session ID")
        path = self._asset(sample.source_path)
        if path != sample.image_path.resolve() or digest_file(path) != sample.sha256:
            raise CaptureError("Prediction source hash or path changed")
        with Image.open(path) as image:
            if image.size != (sample.width, sample.height):
                raise CaptureError("Prediction source dimensions changed")

    def _asset(self, value: str) -> Path:
        relative = PurePosixPath(value)
        path = (self.data_root / Path(*relative.parts)).resolve()
        if not value or relative.is_absolute() or ".." in relative.parts or "\\" in value or not path.is_relative_to(self.data_root):
            raise CaptureError("Unsafe prediction asset path")
        return path

    @staticmethod
    def _validate_candidates(sample: CaptureSample, candidates: tuple[Candidate, ...]) -> None:
        indices = set()
        for candidate in candidates:
            if not isinstance(candidate.index, int) or candidate.index < 0 or candidate.index in indices:
                raise CaptureError("Invalid or duplicate object index")
            indices.add(candidate.index)
            if not isinstance(candidate.mask, ArrayMask) or candidate.mask.pixels.shape != (sample.height, sample.width) or not candidate.mask.pixels.any():
                raise CaptureError("Mask must be nonempty and match source dimensions")
            if not math.isfinite(candidate.score) or not 0 <= candidate.score <= 1:
                raise CaptureError("Invalid SAM3 score")
            if len(candidate.box) != BBOX_COORDINATES:
                raise CaptureError("Invalid prediction bbox coordinate count")
            left, top, right, bottom = candidate.box
            if not all(math.isfinite(value) for value in candidate.box) or not (0 <= left < right <= sample.width and 0 <= top < bottom <= sample.height):
                raise CaptureError("Invalid prediction bbox")

    @staticmethod
    def _same_candidates(first: tuple[Candidate, ...], second: tuple[Candidate, ...]) -> bool:
        return len(first) == len(second) and all(
            left.index == right.index and left.box == right.box and left.score == right.score
            and isinstance(left.mask, ArrayMask) and isinstance(right.mask, ArrayMask) and np.array_equal(left.mask.pixels, right.mask.pixels)
            for left, right in zip(first, second, strict=True)
        )

    def _load(self, connection: sqlite3.Connection, sample: CaptureSample, prediction_id: str) -> tuple[Candidate, ...] | None:
        """推論情報を照合し、表示可能なmaskがなければ未生成として返す。"""
        identity = self._identity(sample, prediction_id)
        header = connection.execute("SELECT * FROM sam3_predictions WHERE session_id=? AND image_id=? AND prediction_id=?", identity).fetchone()
        if header is None:
            return None
        if (header["source_sha256"], header["source_path"], header["width"], header["height"]) != (sample.sha256, sample.source_path, sample.width, sample.height):
            raise CaptureError("Stored prediction source differs")
        if header["model_signature"] is not None and prediction_id != hashlib.sha256(f"{sample.sha256}:{header['model_signature']}".encode()).hexdigest():
            raise CaptureError("Stored prediction signature differs")
        rows = connection.execute("SELECT * FROM sam3_prediction_objects WHERE session_id=? AND image_id=? AND prediction_id=? ORDER BY object_index", identity).fetchall()
        if len(rows) != header["object_count"]:
            raise CaptureError("Incomplete prediction objects")
        preview = self._previews.get(identity)
        if preview is not None:
            return preview
        candidates = []
        for row in rows:
            # 新規の確認用maskはメモリー限定であり、未保存は正常な状態である。
            approved = connection.execute("""SELECT m.mask_path, m.mask_sha256 FROM sam3_approved_masks m
                JOIN sam3_review_revisions r USING(revision)
                WHERE r.session_id=? AND r.image_id=? AND r.prediction_id=? AND m.object_index=?
                AND m.mask_sha256=? LIMIT 1""", (*identity, row["object_index"], row["mask_sha256"])).fetchone() if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='sam3_approved_masks'").fetchone() else None
            value = approved["mask_path"] if approved is not None else row["mask_path"]
            if value.startswith(PREVIEW_PREFIX):
                return None
            path = self._asset(value)
            if not path.is_file():
                LOGGER.debug("保存済みmaskがないため確認時に再推論が必要である: %s", path)
                return None
            if digest_file(path) != row["mask_sha256"]:
                raise CaptureError("Prediction mask hash changed")
            with Image.open(path) as image:
                pixels = np.asarray(image)
                if image.format != MASK_FORMAT or image.mode != MASK_MODE or not np.isin(pixels, (0, MASK_FOREGROUND)).all():
                    raise CaptureError("Invalid binary mask PNG")
                candidates.append(Candidate(row["object_index"], tuple(json.loads(row["bbox_json"])), row["sam3_score"], ArrayMask(pixels > 0)))
        result = tuple(candidates)
        self._validate_candidates(sample, result)
        return result
