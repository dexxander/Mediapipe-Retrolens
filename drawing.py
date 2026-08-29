"""
drawing.py
==========
The air-drawing feature: a persistent canvas the index fingertip paints
onto, rendered as a glowing neon trail composited over the live camera
image. The canvas is intentionally a separate layer from the camera frame
(np.zeros_like) so the drawing survives frames where no hand is visible,
and is only ever cleared explicitly (C / R keys).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from utils import smooth_point

Point = Tuple[float, float]


class DrawingCanvas:
    def __init__(
        self,
        width: int,
        height: int,
        color: Tuple[int, int, int],
        thickness: int,
        smoothing: float,
    ) -> None:
        self.width = width
        self.height = height
        self.color = color
        self.thickness = thickness
        self.smoothing = smoothing
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self._previous_point: Optional[Tuple[int, int]] = None
        self._smoothed_point: Optional[Point] = None

    def resize(self, width: int, height: int) -> None:
        """Reallocate (and clear) the canvas if the camera resolution changes."""
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.reset_stroke()

    def add_point(self, x: float, y: float) -> None:
        """Feed in the current (already-smoothed-by-caller-or-not) fingertip position."""
        current_raw = (x, y)
        self._smoothed_point = smooth_point(current_raw, self._smoothed_point, self.smoothing)
        current = (int(self._smoothed_point[0]), int(self._smoothed_point[1]))

        if self._previous_point is not None:
            cv2.line(self.layer, self._previous_point, current, self.color, self.thickness, cv2.LINE_AA)
        self._previous_point = current

    def reset_stroke(self) -> None:
        """Call when the drawing gesture stops, so the next stroke doesn't jump-connect."""
        self._previous_point = None
        self._smoothed_point = None

    def clear(self) -> None:
        self.layer[:] = 0
        self.reset_stroke()


def composite_drawing(base_frame: np.ndarray, drawing_layer: np.ndarray, large_sigma: float, small_sigma: float) -> np.ndarray:
    """
    Layer a soft outer glow, a tighter inner glow, and a sharp core line over
    the camera frame — the combination that reads as "neon" rather than a
    flat drawn line.
    """
    if not np.any(drawing_layer):
        return base_frame

    glow_large = cv2.GaussianBlur(drawing_layer, (0, 0), sigmaX=large_sigma, sigmaY=large_sigma)
    glow_small = cv2.GaussianBlur(drawing_layer, (0, 0), sigmaX=small_sigma, sigmaY=small_sigma)
    glow = cv2.addWeighted(glow_large, 0.55, glow_small, 0.75, 0)

    out = cv2.addWeighted(base_frame, 1.0, glow, 0.85, 0)
    out = cv2.addWeighted(out, 1.0, drawing_layer, 1.0, 0)  # sharp core on top
    return out
