"""
camera.py
=========
A thin, well-behaved wrapper around cv2.VideoCapture.

Responsibilities:
  * open the webcam (with a Windows-friendly backend when available)
  * request a resolution, but trust whatever the driver actually gives back
  * mirror every frame so the app behaves like a selfie camera
  * fail with a clear CameraError instead of an obscure OpenCV exception
"""

from __future__ import annotations

import platform
from typing import Optional

import cv2
import numpy as np

import config


class CameraError(Exception):
    """Raised when the webcam cannot be opened or stops delivering frames."""


class Camera:
    def __init__(self, index: int, requested_width: int, requested_height: int) -> None:
        self.index = index
        self._cap = self._open(index)

        if not self._cap or not self._cap.isOpened():
            raise CameraError(
                f"Unable to open webcam at index {index}.\n"
                f"  Please check that a camera is connected, not in use by\n"
                f"  another application, and that CAMERA_INDEX in config.py\n"
                f"  matches your device."
            )

        # Ask for the desired resolution, but never assume the driver honored it.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or requested_width
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or requested_height
        self.width = actual_width
        self.height = actual_height

    @staticmethod
    def _open(index: int) -> Optional[cv2.VideoCapture]:
        # DirectShow tends to open faster and more reliably than the default
        # backend (MSMF) on Windows; elsewhere we just use the platform default.
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        return cv2.VideoCapture(index)

    def read(self) -> Optional[np.ndarray]:
        """
        Grab one frame, mirrored horizontally (selfie-style).
        Returns None if the camera has stopped delivering frames (e.g. disconnected)
        instead of raising, so the main loop can shut down cleanly.
        """
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return cv2.flip(frame, 1)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
