"""YOLOモデルの設定読み込みと画像物体検出推論を提供する。"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "configs" / "settings_yolo.yaml"
IMAGE_DIMENSIONS = 3
IMAGE_CHANNELS = 3


def _validate_image(image: NDArray[np.generic]) -> None:
    """推論入力画像の配列形状とデータ型を検証する。

    Args:
        image: 検証する画像配列。

    Raises:
        TypeError: 入力がNumPy配列でない場合。
        ValueError: 配列の形状またはデータ型が推論要件を満たさない場合。
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("入力画像はNumPy配列で指定してください。")
    if image.ndim != IMAGE_DIMENSIONS or image.shape[2] != IMAGE_CHANNELS:
        raise ValueError("入力画像の形状は H×W×3 にしてください。")
    if image.dtype != np.uint8:
        raise ValueError("入力画像のデータ型は uint8 にしてください。")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("入力画像が空です。画像の内容を確認してください。")


def load_settings() -> tuple[Path, float]:
    """YOLO設定を読み込み、重みファイルと信頼度閾値を返す。

    Returns:
        解決済みの重みファイルパスと信頼度閾値。

    Raises:
        FileNotFoundError: 設定ファイルまたは重みファイルが存在しない場合。
        OSError: 設定ファイルを読み込めない場合。
        TypeError: 設定値の型が不正な場合。
        ValueError: 設定ファイルの内容が不正な場合。
    """
    try:
        with _SETTINGS_PATH.open(encoding="utf-8") as settings_file:
            settings = yaml.safe_load(settings_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"YOLO設定ファイルが見つかりません: {_SETTINGS_PATH}") from exc
    except (OSError, UnicodeError) as exc:
        raise OSError(f"YOLO設定ファイルを読み込めませんでした: {_SETTINGS_PATH}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YOLO設定ファイルのYAML形式が正しくありません: {_SETTINGS_PATH}") from exc

    if not isinstance(settings, dict):
        raise TypeError(f"YOLO設定ファイルはキーと値の対応形式で記述してください: {_SETTINGS_PATH}")

    weights_value = settings.get("weights")
    if not isinstance(weights_value, str) or not weights_value.strip():
        raise ValueError(f"YOLO設定ファイルのキー 'weights' に重みファイルのパスを指定してください: {_SETTINGS_PATH}")
    if "\x00" in weights_value:
        raise ValueError(f"YOLO設定ファイルのキー 'weights' に不正なパスが指定されています: {_SETTINGS_PATH}")

    try:
        weights_path = Path(weights_value)
        if not weights_path.is_absolute():
            weights_path = _SETTINGS_PATH.parent / weights_path
        weights_path = weights_path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"YOLO設定ファイルのキー 'weights' に不正なパスが指定されています: {_SETTINGS_PATH}") from exc
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO重みファイルが見つかりません: {weights_path}")

    confidence_threshold = settings.get("confidence_threshold")
    if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
        raise TypeError(f"YOLO設定ファイルのキー 'confidence_threshold' は0から1の数値で指定してください: {_SETTINGS_PATH}")
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError(f"YOLO設定ファイルのキー 'confidence_threshold' は0から1の範囲で指定してください: {_SETTINGS_PATH}")

    return weights_path, float(confidence_threshold)


def _load_model(weights_path: Path) -> Any:
    """指定された重みファイルからYOLOモデルを読み込む。

    Args:
        weights_path: YOLO重みファイルのパス。

    Returns:
        Ultralyticsが生成した推論モデル。

    Raises:
        RuntimeError: YOLOモデルを読み込めない場合。
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415

        return YOLO(str(weights_path))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("YOLOモデルを読み込めませんでした。重みファイルと実行環境を確認してください。") from exc


class YoloDetector:
    """設定に基づいてYOLO画像推論を実行する。"""

    def __init__(self, confidence_threshold: float | None = None) -> None:
        """設定と重みファイルから推論器を初期化する。"""
        weights_path, configured_confidence = load_settings()
        if confidence_threshold is not None:
            confidence_is_number = isinstance(confidence_threshold, (int, float)) and not isinstance(confidence_threshold, bool)
            confidence_is_valid = confidence_is_number and math.isfinite(confidence_threshold) and 0 <= confidence_threshold <= 1
            if not confidence_is_valid:
                raise ValueError("confidence_threshold must be a finite number from 0 to 1")
            configured_confidence = confidence_threshold
        self.weights_path = weights_path
        self.confidence_threshold = configured_confidence
        self.model: Any = _load_model(weights_path)

    def infer(self, image: NDArray[np.uint8]) -> list[dict[str, Any]]:
        """画像内の物体を検出し、クラス名・信頼度・矩形座標を返す。

        Args:
            image: H×W×3形式のuint8画像配列。

        Returns:
            検出ごとのクラス名、信頼度、xyxy形式の座標を持つ辞書。

        Raises:
            ValueError: 入力画像が推論要件を満たさない場合。
            RuntimeError: YOLO推論に失敗した場合。
        """
        _validate_image(image)
        try:
            results = self.model.predict(source=image, conf=self.confidence_threshold, verbose=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("YOLO推論に失敗しました。入力画像と実行環境を確認してください。") from exc

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        coordinates = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_names = self.model.names

        detections = []
        for coordinates_for_box, class_id_value, confidence in zip(coordinates, class_ids, confidences, strict=True):
            class_index = int(class_id_value)
            if isinstance(class_names, dict):
                class_name = class_names.get(class_index, class_names.get(str(class_index)))
            else:
                class_name = class_names[class_index]

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "xyxy": [float(coordinate) for coordinate in coordinates_for_box],
                }
            )

        return detections
