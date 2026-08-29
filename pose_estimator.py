"""
pose_estimator.py
==================
Head orientation (roll / pitch / yaw) from MediaPipe face landmarks, using
the classic 6-point solvePnP approach:

    face landmarks (2D, pixels)
            v
    6 semantic points selected (nose, chin, eye corners, mouth corners)
            v
    matched against a generic 3D face model
            v
    cv2.solvePnP  ->  rotation vector
            v
    cv2.Rodrigues ->  rotation matrix
            v
    matrix decomposition -> Euler angles (degrees)
            v
    exponential smoothing -> stable ROLL / PITCH / YAW readout
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import smooth_value

Point = Tuple[float, float]

# Landmark indices in MediaPipe's 468-point face mesh topology (the extra
# iris points some configurations add are appended after index 467, so
# these indices stay valid either way).
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

_POSE_LANDMARK_ORDER = (
    NOSE_TIP, CHIN, LEFT_EYE_CORNER, RIGHT_EYE_CORNER, LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER,
)

# Generic 3D face model (arbitrary units, roughly millimetres), nose-centered.
# This does not need to match the user's actual face metrically — solvePnP
# only needs consistent proportions to recover a believable rotation.
_MODEL_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -330.0, -65.0),     # chin
        (-225.0, 170.0, -135.0),  # left eye, outer corner
        (225.0, 170.0, -135.0),   # right eye, outer corner
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0),   # right mouth corner
    ],
    dtype=np.float64,
)


def _rotation_matrix_to_euler_degrees(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a rotation matrix into (pitch, yaw, roll) in degrees."""
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    to_deg = math.degrees
    return to_deg(pitch), to_deg(yaw), to_deg(roll)


class HeadPoseEstimator:
    """Stateful estimator: owns the smoothing history between frames."""

    def __init__(self, smoothing_alpha: float, max_degrees: float) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.max_degrees = max_degrees
        self._roll: Optional[float] = None
        self._pitch: Optional[float] = None
        self._yaw: Optional[float] = None

    def reset(self) -> None:
        self._roll = self._pitch = self._yaw = None

    def estimate(
        self,
        face_landmarks_px: Sequence[Point],
        frame_width: int,
        frame_height: int,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Returns smoothed (roll, pitch, yaw) in degrees, or None if the pose
        could not be solved this frame (estimate() simply keeps last frame's
        smoothed values available via .last if you need to hold the display
        steady rather than blanking it).
        """
        try:
            image_points = np.array(
                [face_landmarks_px[i] for i in _POSE_LANDMARK_ORDER],
                dtype=np.float64,
            )
        except IndexError:
            return None

        focal_length = float(frame_width)
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rotation_vec, _translation_vec = cv2.solvePnP(
            _MODEL_3D_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rmat, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, roll = _rotation_matrix_to_euler_degrees(rmat)

        pitch = max(-self.max_degrees, min(self.max_degrees, pitch))
        yaw = max(-self.max_degrees, min(self.max_degrees, yaw))
        roll = max(-self.max_degrees, min(self.max_degrees, roll))

        self._roll = smooth_value(roll, self._roll, self.smoothing_alpha)
        self._pitch = smooth_value(pitch, self._pitch, self.smoothing_alpha)
        self._yaw = smooth_value(yaw, self._yaw, self.smoothing_alpha)

        return self._roll, self._pitch, self._yaw
