"""
hand_tracker.py
================
Wraps MediaPipe's HandLandmarker task (the modern Tasks API — the older
`mp.solutions.hands` API has been removed from current MediaPipe releases).
Runs in VIDEO mode: synchronous, one call per frame, driven by a
monotonically increasing millisecond timestamp.

Also owns the "retro" hand landmark visualization: small pink/red dots and
thin connecting lines, deliberately understated so the camera image stays
the star of the show.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from utils import MonotonicMsClock, ensure_model

Point = Tuple[float, float]

_LANDMARK_COLOR = (140, 110, 255)    # pink/red, BGR
_CONNECTION_COLOR = (90, 60, 160)    # subdued magenta, BGR
_LANDMARK_RADIUS = 2
_CONNECTION_THICKNESS = 1


class HandTracker:
    def __init__(
        self,
        models_dir: Path,
        max_hands: int = 2,
        detection_confidence: float = 0.6,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        model_path = ensure_model(models_dir, "hand_landmarker.task")

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._clock = MonotonicMsClock()
        self._connections = [
            (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
        ]

    def process(self, frame_rgb: np.ndarray, elapsed_seconds: float):
        """Runs hand detection on an RGB frame. Returns a HandLandmarkerResult."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = self._clock.next_ms(elapsed_seconds)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def get_landmarks_px(result, width: int, height: int) -> List[List[Point]]:
        """Each detected hand becomes a list of 21 (x, y) pixel-space points."""
        hands_px: List[List[Point]] = []
        if not result or not result.hand_landmarks:
            return hands_px
        for hand in result.hand_landmarks:
            hands_px.append([(lm.x * width, lm.y * height) for lm in hand])
        return hands_px

    def draw_landmarks(self, frame: np.ndarray, hands_px: List[List[Point]]) -> None:
        """Subtle retro-style overlay: thin connecting lines + small dots per hand."""
        for hand in hands_px:
            for start_idx, end_idx in self._connections:
                p1 = (int(hand[start_idx][0]), int(hand[start_idx][1]))
                p2 = (int(hand[end_idx][0]), int(hand[end_idx][1]))
                cv2.line(frame, p1, p2, _CONNECTION_COLOR, _CONNECTION_THICKNESS, cv2.LINE_AA)
            for x, y in hand:
                cv2.circle(frame, (int(x), int(y)), _LANDMARK_RADIUS, _LANDMARK_COLOR, -1, cv2.LINE_AA)

    def close(self) -> None:
        self._landmarker.close()
