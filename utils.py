"""
utils.py
========
Small, generic, reusable helpers shared across the application:
geometry (distance, clamping), smoothing, FPS measurement, and the
one-time downloader that fetches MediaPipe's .task model bundles.

Nothing in this module knows about hands, faces, or gestures
specifically — that logic lives in gesture_detector.py and friends.
"""

from __future__ import annotations

import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Sequence, Tuple

Point = Tuple[float, float]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value to the inclusive [min_value, max_value] range."""
    return max(min_value, min(max_value, value))


def clamp_point(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    """Clamp a pixel coordinate so it always lies inside a width x height image."""
    return (
        int(clamp(x, 0, max(width - 1, 0))),
        int(clamp(y, 0, max(height - 1, 0))),
    )


def landmarks_to_pixels(landmarks, width: int, height: int) -> list[Point]:
    """Convert a list of normalized MediaPipe landmarks (.x/.y in [0,1]) to pixel tuples."""
    return [(lm.x * width, lm.y * height) for lm in landmarks]


# --------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------
def smooth_value(current: float, previous: Optional[float], alpha: float) -> float:
    """
    Exponential moving average: smoothed = alpha * current + (1 - alpha) * previous.
    If there is no previous value yet, the current value is returned as-is
    (avoids a "ramp up from zero" artifact on the very first sample).
    """
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous


def smooth_point(current: Point, previous: Optional[Point], alpha: float) -> Point:
    """Exponential moving average applied independently to x and y."""
    if previous is None:
        return current
    return (
        alpha * current[0] + (1.0 - alpha) * previous[0],
        alpha * current[1] + (1.0 - alpha) * previous[1],
    )


# --------------------------------------------------------------------------
# FPS measurement
# --------------------------------------------------------------------------
class FPSCounter:
    """
    Tracks a smoothed frames-per-second value.

    A naive 1/dt readout jitters wildly frame to frame, so we blend it
    with an exponential moving average — this is what gives RETROLENS a
    stable "FPS 30" instead of a number flickering between 24 and 41.
    """

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self._fps = 0.0
        self._last_time = time.perf_counter()

    def tick(self) -> float:
        """Call once per frame. Returns the current smoothed FPS value."""
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            instantaneous = 1.0 / dt
            if self._fps <= 0:
                self._fps = instantaneous
            else:
                self._fps = self.smoothing * self._fps + (1.0 - self.smoothing) * instantaneous
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# --------------------------------------------------------------------------
# Monotonic timestamp guard (MediaPipe VIDEO-mode tasks require strictly
# increasing millisecond timestamps)
# --------------------------------------------------------------------------
class MonotonicMsClock:
    """Hands out strictly-increasing millisecond timestamps for MediaPipe VIDEO mode."""

    def __init__(self) -> None:
        self._last_ms = -1

    def next_ms(self, elapsed_seconds: float) -> int:
        ms = int(elapsed_seconds * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        return ms


# --------------------------------------------------------------------------
# One-time MediaPipe model download
# --------------------------------------------------------------------------
# MediaPipe's modern Tasks API (used by this project) ships without bundled
# model weights — it loads a small .task bundle from disk. These are the
# official Google-hosted CPU model bundles. They are downloaded once and
# cached under assets/models/, so every run after the first works offline.
MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def ensure_model(models_dir: Path, filename: str) -> Path:
    """
    Guarantee that models_dir/filename exists locally, downloading it from
    Google's model store on first run if needed. Raises RuntimeError with a
    clear, actionable message (including a manual-download URL) on failure —
    callers should catch this and show a clean startup error rather than
    letting a raw traceback surface.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / filename

    if target.exists() and target.stat().st_size > 0:
        return target

    url = MODEL_URLS.get(filename)
    if url is None:
        raise RuntimeError(f"No known download URL for model '{filename}'.")

    print(f"[RETROLENS] Downloading required model '{filename}' (first run only)...")
    tmp_path = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(target)
        print(f"[RETROLENS] Saved {filename} -> {target}")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the required model '{filename}'.\n"
            f"  Reason: {exc}\n"
            f"  This only needs to happen once, but it does need an internet\n"
            f"  connection the first time RETROLENS runs.\n"
            f"  You can also download it manually from:\n"
            f"    {url}\n"
            f"  and place the file at:\n"
            f"    {target}"
        ) from exc

    return target
