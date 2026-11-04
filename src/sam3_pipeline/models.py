"""Shared image-space geometry types used by the SAM3 adapter."""

from __future__ import annotations

#: [xmin, ymin, xmax, ymax]、元画像のピクセル座標
Box = tuple[float, float, float, float]
