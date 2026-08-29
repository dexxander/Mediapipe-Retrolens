"""
ui.py
=====
Every piece of the retro camera HUD. Nothing here hard-codes a resolution —
every position is computed from the frame's actual (height, width) so the
layout holds together whether the camera is 640x480 or 1920x1080.

Drawing order (composed by main.py) roughly matches a real camera's
viewfinder: live image first, gesture/drawing overlays next, then the
static HUD chrome (REC dot, timestamp, crosshair, bottom status panel) on
top of everything.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from gesture_detector import FrameGestureResult

Point = Tuple[float, float]
Box = Tuple[int, int, int, int]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _put_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    font: int = _FONT,
    shadow: bool = True,
) -> None:
    """Draws text with a faint black shadow so it stays legible over any background."""
    if shadow:
        cv2.putText(frame, text, (org[0] + 1, org[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _text_width(text: str, scale: float, thickness: int, font: int = _FONT) -> int:
    (w, _h), _ = cv2.getTextSize(text, font, scale, thickness)
    return w


# --------------------------------------------------------------------------
# Crosshair
# --------------------------------------------------------------------------
def draw_crosshair(frame: np.ndarray, color: Tuple[int, int, int], size: int = 9, gap: int = 3) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - size - gap, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + gap + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size - gap), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + gap + size), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 1, color, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# REC indicator (top-left) + timestamp (top-right)
# --------------------------------------------------------------------------
def draw_rec_indicator(frame: np.ndarray, t: float, color: Tuple[int, int, int], text_color: Tuple[int, int, int]) -> None:
    margin = 16
    blink_on = int(t * 1.6) % 2 == 0
    if blink_on:
        cv2.circle(frame, (margin + 5, margin + 2), 5, color, -1, cv2.LINE_AA)
    _put_text(frame, "REC", (margin + 16, margin + 7), 0.55, text_color, 1)


def draw_timestamp(frame: np.ndarray, color: Tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    text = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    tw = _text_width(text, 0.55, 1)
    margin = 16
    _put_text(frame, text, (w - tw - margin, margin + 7), 0.55, color, 1)


# --------------------------------------------------------------------------
# Bottom HUD panel: status line + head-pose readout
# --------------------------------------------------------------------------
def draw_bottom_hud(
    frame: np.ndarray,
    label: str,
    mode: str,
    fps: float,
    photo_count: int,
    roll: Optional[float],
    pitch: Optional[float],
    yaw: Optional[float],
    accent_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    dim_color: Tuple[int, int, int],
) -> None:
    h, w = frame.shape[:2]
    panel_top = h - 96

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.line(frame, (0, panel_top), (w, panel_top), accent_color, 1, cv2.LINE_AA)

    status = f"{label} | {mode} | FPS {fps:.0f} | FOTO {photo_count}"
    _put_text(frame, status, (16, panel_top + 24), 0.6, text_color, 1)

    def pose_line(value: Optional[float]) -> str:
        if value is None:
            return "--"
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.0f}"

    pose_color = dim_color if roll is None else accent_color
    _put_text(frame, f"ROLL {pose_line(roll)}", (16, panel_top + 46), 0.48, pose_color, 1)
    _put_text(frame, f"PITCH {pose_line(pitch)}", (16, panel_top + 65), 0.48, pose_color, 1)
    _put_text(frame, f"YAW {pose_line(yaw)}", (16, panel_top + 84), 0.48, pose_color, 1)


# --------------------------------------------------------------------------
# Face ROI label, e.g. "MONO 364x240"
# --------------------------------------------------------------------------
def draw_roi_label(frame: np.ndarray, box: Box, mode: str, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    label = f"{mode} {x2 - x1}x{y2 - y1}"
    label_y = y1 - 8 if y1 - 8 > 12 else y2 + 18
    _put_text(frame, label, (x1, label_y), 0.42, color, 1)


# --------------------------------------------------------------------------
# Virtual two-hand frame + countdown
# --------------------------------------------------------------------------
def draw_virtual_frame(frame: np.ndarray, result: FrameGestureResult, color: Tuple[int, int, int]) -> None:
    if not result.valid:
        return
    tl = (int(result.top_left[0]), int(result.top_left[1]))
    tr = (int(result.top_right[0]), int(result.top_right[1]))
    bl = (int(result.bottom_left[0]), int(result.bottom_left[1]))
    br = (int(result.bottom_right[0]), int(result.bottom_right[1]))

    for p1, p2 in ((tl, tr), (tr, br), (br, bl), (bl, tl)):
        cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

    bracket = max(10, int(min(result.width, result.height) * 0.08))
    for (x, y), (dx, dy) in (
        (tl, (1, 1)), (tr, (-1, 1)), (bl, (1, -1)), (br, (-1, -1)),
    ):
        cv2.line(frame, (x, y), (x + dx * bracket, y), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * bracket), color, 2, cv2.LINE_AA)
    for corner in (tl, tr, bl, br):
        cv2.circle(frame, corner, 3, color, -1, cv2.LINE_AA)


def draw_countdown(
    frame: np.ndarray,
    remaining: Optional[float],
    result: FrameGestureResult,
    show_capture_text: bool,
    color: Tuple[int, int, int],
) -> None:
    h, w = frame.shape[:2]
    if result.valid:
        cx = int((result.top_left[0] + result.top_right[0] + result.bottom_left[0] + result.bottom_right[0]) / 4)
        cy = int((result.top_left[1] + result.top_right[1] + result.bottom_left[1] + result.bottom_right[1]) / 4)
    else:
        cx, cy = w // 2, h // 2

    if show_capture_text:
        text = "CAPTURE"
        scale, thickness = 1.1, 3
    elif remaining is not None:
        text = str(max(1, math.ceil(remaining)))
        scale, thickness = 2.6, 4
    else:
        return

    tw = _text_width(text, scale, thickness, _FONT_BOLD)
    org = (cx - tw // 2, cy + 12)
    _put_text(frame, text, org, scale, color, thickness, font=_FONT_BOLD)


# --------------------------------------------------------------------------
# Photo flash
# --------------------------------------------------------------------------
def apply_flash(frame: np.ndarray, intensity: float) -> np.ndarray:
    if intensity <= 0:
        return frame
    white = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 1.0 - intensity, white, intensity, 0)


# --------------------------------------------------------------------------
# Subtle full-frame vignette (always-on cinematic touch, independent of FILM mode)
# --------------------------------------------------------------------------
_vignette_mask_cache: Dict[Tuple[int, int], np.ndarray] = {}


def apply_global_vignette(frame: np.ndarray, strength: float) -> np.ndarray:
    h, w = frame.shape[:2]
    mask = _vignette_mask_cache.get((h, w))
    if mask is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        max_r = math.hypot(cx, cy)
        r = np.hypot(xx - cx, yy - cy) / max_r
        falloff = 1.0 - strength * np.clip((r - 0.6) / 0.4, 0, 1)
        mask = np.clip(falloff, 1.0 - strength, 1.0)[..., None]
        _vignette_mask_cache[(h, w)] = mask
    return np.clip(frame.astype(np.float32) * mask, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Debug overlay
# --------------------------------------------------------------------------
def draw_debug_overlay(frame: np.ndarray, lines: List[str], accent_color: Tuple[int, int, int], text_color: Tuple[int, int, int]) -> None:
    x, y0 = 16, 56
    line_h = 18
    panel_h = line_h * len(lines) + 16
    panel_w = 220

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 8, y0 - 14), (x - 8 + panel_w, y0 - 14 + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)
    cv2.rectangle(frame, (x - 8, y0 - 14), (x - 8 + panel_w, y0 - 14 + panel_h), accent_color, 1)

    for i, line in enumerate(lines):
        _put_text(frame, line, (x, y0 + i * line_h), 0.42, text_color, 1, shadow=False)


# --------------------------------------------------------------------------
# Saved-photo timestamp burn-in (distinct from the live on-screen timestamp)
# --------------------------------------------------------------------------
def stamp_photo_timestamp(frame: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    text = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    tw = _text_width(text, 0.5, 1)
    margin = 14
    _put_text(frame, text, (w - tw - margin, h - margin), 0.5, color, 1)
    return frame
