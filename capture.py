"""
capture.py
==========
Everything to do with turning a frame into a saved photo: making sure
captures/ exists, picking the next filename, writing the file, and
tracking the short-lived flash / "CAPTURE" state the UI reacts to.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_FILENAME_RE = re.compile(r"retrolens_(\d+)\.jpg$")


class PhotoCapture:
    def __init__(self, output_dir: Path, flash_duration_frames: int, capture_text_frames: int) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flash_duration_frames = flash_duration_frames
        self.capture_text_frames = capture_text_frames
        self.count = self._infer_starting_count()
        self._flash_frames_remaining = 0
        self._capture_text_frames_remaining = 0

    def _infer_starting_count(self) -> int:
        """Resume numbering from whatever is already in captures/, so re-runs don't overwrite old photos."""
        highest = 0
        for path in self.output_dir.glob("retrolens_*.jpg"):
            match = _FILENAME_RE.search(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest

    def next_path(self) -> Path:
        return self.output_dir / f"retrolens_{self.count + 1:03d}.jpg"

    def save(self, frame: np.ndarray) -> Optional[Path]:
        """Writes `frame` to the next sequential filename. Returns the path, or None on failure."""
        path = self.next_path()
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            return None
        self.count += 1
        self._flash_frames_remaining = self.flash_duration_frames
        self._capture_text_frames_remaining = self.capture_text_frames
        return path

    def consume_flash(self) -> float:
        """
        Call once per frame after a save. Returns a flash intensity in
        [0, 1] for *this* frame (0 once the flash has finished) and
        advances the internal countdown. The flash itself is brief — a
        real camera flash, not a fade.
        """
        if self._flash_frames_remaining <= 0:
            return 0.0
        intensity = self._flash_frames_remaining / self.flash_duration_frames
        self._flash_frames_remaining -= 1
        return intensity

    def consume_capture_text(self) -> bool:
        """
        Call once per frame after a save. Returns whether the "CAPTURE"
        HUD text should still be shown this frame, and advances the
        countdown. Lasts a little longer than the flash itself so the
        word is actually readable.
        """
        if self._capture_text_frames_remaining <= 0:
            return False
        self._capture_text_frames_remaining -= 1
        return True

    @property
    def flash_active(self) -> bool:
        return self._flash_frames_remaining > 0

    @property
    def capture_text_active(self) -> bool:
        return self._capture_text_frames_remaining > 0
