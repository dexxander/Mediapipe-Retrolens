"""
face_tracker.py
================
Wraps MediaPipe's FaceLandmarker task (modern Tasks API). Runs in VIDEO
mode, same pattern as hand_tracker.py. Beyond raw landmarks, this module
also derives the face bounding box / ROI that effects.py operates on,
with coordinates always clamped inside the frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from utils import MonotonicMsClock, clamp, ensure_model

Point = Tuple[float, float]
Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


class FaceTracker:
    def __init__(
        self,
        models_dir: Path,
        max_faces: int = 1,
        detection_confidence: float = 0.6,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        model_path = ensure_model(models_dir, "face_landmarker.task")

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._clock = MonotonicMsClock()
        self._frame_index = 0
        self._last_result = None

    def process(self, frame_rgb: np.ndarray, elapsed_seconds: float):
        """Runs face landmark detection on an RGB frame. Returns a FaceLandmarkerResult."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = self._clock.next_ms(elapsed_seconds)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def process_throttled(self, frame_rgb: np.ndarray, elapsed_seconds: float, interval: int = 2):
        """Reuse the last face result between updates to reduce CPU load."""
        self._frame_index += 1
        if self._last_result is None or self._frame_index % max(1, interval) == 0:
            self._last_result = self.process(frame_rgb, elapsed_seconds)
        return self._last_result

    def reset_tracking(self) -> None:
        self._frame_index = 0
        self._last_result = None

    @staticmethod
    def get_landmarks_px(result, width: int, height: int) -> Optional[List[Point]]:
        """The first detected face's landmarks as pixel-space points, or None."""
        if not result or not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        return [(lm.x * width, lm.y * height) for lm in face]

    @staticmethod
    def get_bounding_box(
        landmarks_px: List[Point],
        width: int,
        height: int,
        margin: float,
    ) -> Box:
        """
        Bounding box around every landmark, expanded by `margin` (a fraction
        of the box's own size) on each side and clamped to the frame.
        """
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        box_w = x2 - x1
        box_h = y2 - y1
        x1 -= box_w * margin
        x2 += box_w * margin
        y1 -= box_h * margin
        y2 += box_h * margin

        x1_i = int(clamp(x1, 0, width - 1))
        x2_i = int(clamp(x2, 0, width - 1))
        y1_i = int(clamp(y1, 0, height - 1))
        y2_i = int(clamp(y2, 0, height - 1))
        return x1_i, y1_i, x2_i, y2_i

    @staticmethod
    def get_center(box: Box) -> Point:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def close(self) -> None:
        self._landmarker.close()
