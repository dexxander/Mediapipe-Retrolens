"""
ui.py
=====
Every piece of the retro camera HUD. Nothing here hard-codes a resolution —
every position is computed from the frame's actual (height, width) so the
layout holds together whether the camera is 640x480 or 1920x1080.

Visual language: text floats directly over the live video (no background
bars/panels), in white with a cyan/magenta chromatic-aberration "glitch"
fringe. The gesture-driven frame box and its countdown share one accent
color that runs amber/orange normally and switches to magenta for the
final second before a capture, as an "about to shoot" cue.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Box = Tuple[int, int, int, int]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------
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
    """Plain text with a faint black shadow so it stays legible over any background."""
    if shadow:
        cv2.putText(frame, text, (org[0] + 1, org[1] + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _glitch_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    font: int = _FONT,
    offset: int = 1,
) -> None:
    """
    Text with a small chromatic-aberration fringe: a cyan copy offset one
    way and a magenta copy offset the other, with the main color on top.
    Cheap to draw and reads as a retro/glitch digital HUD at a glance.
    """
    cv2.putText(frame, text, (org[0] - offset, org[1]), font, scale, (255, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (org[0] + offset, org[1]), font, scale, (200, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _text_width(text: str, scale: float, thickness: int, font: int = _FONT) -> int:
    (w, _h), _ = cv2.getTextSize(text, font, scale, thickness)
    return w


# --------------------------------------------------------------------------
# Crosshair
# --------------------------------------------------------------------------
def draw_crosshair(frame: np.ndarray, color: Tuple[int, int, int], size: int = 8, gap: int = 3) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - size - gap, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + gap + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size - gap), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + gap + size), color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# REC indicator (top-left) + timestamp (top-right)
# --------------------------------------------------------------------------
def draw_rec_indicator(frame: np.ndarray, t: float, dot_color: Tuple[int, int, int], text_color: Tuple[int, int, int]) -> None:
    margin = 16
    blink_on = int(t * 1.6) % 2 == 0
    if blink_on:
        cv2.circle(frame, (margin + 6, margin + 4), 7, dot_color, -1, cv2.LINE_AA)
    _glitch_text(frame, "REC", (margin + 20, margin + 10), 0.65, text_color, 2)


def draw_timestamp(frame: np.ndarray, color: Tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    text = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    tw = _text_width(text, 0.6, 2)
    margin = 16
    _glitch_text(frame, text, (w - tw - margin, margin + 10), 0.6, color, 2)


# --------------------------------------------------------------------------
# Bottom-left status (label|mode|fps|foto) + optional secondary hint/status
# line above it, bottom-right head-pose readout. No background panel —
# text floats directly on the live video, matching the reference.
# --------------------------------------------------------------------------
def draw_bottom_left(
    frame: np.ndarray,
    label: str,
    mode: str,
    fps: float,
    photo_count: int,
    secondary_line: Optional[str],
    text_color: Tuple[int, int, int],
) -> None:
    h, w = frame.shape[:2]
    margin = 16
    main_y = h - margin
    status = f"{label}|{mode}|FPS {fps:.0f}|FOTO {photo_count}"
    if secondary_line:
        _glitch_text(frame, secondary_line, (margin, main_y - 26), 0.55, text_color, 1)
    _glitch_text(frame, status, (margin, main_y), 0.65, text_color, 2)


def draw_pose_readout(frame: np.ndarray, roll: Optional[float], pitch: Optional[float], yaw: Optional[float], color: Tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    margin = 16

    def fmt(value: Optional[float]) -> str:
        if value is None:
            return "--"
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.0f}"

    text = f"ROLL {fmt(roll)} PITCH {fmt(pitch)} YAW {fmt(yaw)}"
    tw = _text_width(text, 0.6, 2)
    _glitch_text(frame, text, (w - tw - margin, h - margin), 0.6, color, 2)


# --------------------------------------------------------------------------
# Gesture-frame box: thin outline + thick side accent bars + corner ticks
# + corner dots + "MODE WxH" label + big countdown number. This is drawn
# on the same box used as the active effect ROI while the two-hand "L"
# gesture is engaged.
# --------------------------------------------------------------------------
def draw_gesture_box(
    frame: np.ndarray,
    top_left: Point,
    top_right: Point,
    bottom_left: Point,
    bottom_right: Point,
    mode: str,
    accent_color: Tuple[int, int, int],
    cyan_color: Tuple[int, int, int],
) -> None:
    tl = (int(top_left[0]), int(top_left[1]))
    tr = (int(top_right[0]), int(top_right[1]))
    bl = (int(bottom_left[0]), int(bottom_left[1]))
    br = (int(bottom_right[0]), int(bottom_right[1]))

    # Thin cyan outline, full quadrilateral (may be a touch non-rectangular -
    # it follows the actual hand-tracked corners, not a forced-square box)
    for p1, p2 in ((tl, tr), (tr, br), (br, bl), (bl, tl)):
        cv2.line(frame, p1, p2, cyan_color, 1, cv2.LINE_AA)

    # thick accent bars on the left and right edges, full height
    cv2.line(frame, tl, bl, accent_color, 3, cv2.LINE_AA)
    cv2.line(frame, tr, br, accent_color, 3, cv2.LINE_AA)

    # short inward corner ticks (top-left, top-right, bottom-left)
    tick = max(14, int(0.12 * max(abs(tr[0] - tl[0]), 1)))
    cv2.line(frame, tl, (tl[0] + tick, tl[1]), accent_color, 3, cv2.LINE_AA)
    cv2.line(frame, tr, (tr[0] - tick, tr[1]), accent_color, 3, cv2.LINE_AA)
    cv2.line(frame, bl, (bl[0] + tick, bl[1]), accent_color, 3, cv2.LINE_AA)

    for corner in (tl, tr, bl, br):
        cv2.circle(frame, corner, 4, cyan_color, -1, cv2.LINE_AA)

    box_w = abs(tr[0] - tl[0])
    box_h = abs(bl[1] - tl[1])
    label = f"{mode} {box_w}x{box_h}"
    _glitch_text(frame, label, (tl[0], tl[1] - 10), 0.5, accent_color, 1)


def draw_countdown_number(
    frame: np.ndarray,
    text: str,
    center: Point,
    color: Tuple[int, int, int],
    scale: float = 2.8,
    thickness: int = 4,
) -> None:
    tw = _text_width(text, scale, thickness, _FONT_BOLD)
    org = (int(center[0] - tw / 2), int(center[1] + 15))
    _put_text(frame, text, org, scale, color, thickness, font=_FONT_BOLD)


# --------------------------------------------------------------------------
# Draw-mode "hold to activate" ring + DRAW MODE label
# --------------------------------------------------------------------------
def draw_charge_ring(
    frame: np.ndarray,
    center: Tuple[int, int],
    progress: float,
    color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    radius: int = 26,
) -> None:
    cv2.circle(frame, center, radius, (60, 60, 60), 2, cv2.LINE_AA)
    sweep = max(1, int(360 * max(0.0, min(1.0, progress))))
    cv2.ellipse(frame, center, (radius, radius), -90, 0, sweep, color, 3, cv2.LINE_AA)
    label = "DRAW MODE"
    tw = _text_width(label, 0.5, 1)
    _glitch_text(frame, label, (center[0] - tw // 2, center[1] - radius - 12), 0.5, text_color, 1)


# --------------------------------------------------------------------------
# Idle hint text (shown when nothing else is going on, guiding the user
# toward the frame gesture)
# --------------------------------------------------------------------------
def draw_hint(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    margin = 16
    _glitch_text(frame, text, (margin, h - margin - 26), 0.5, color, 1)


# --------------------------------------------------------------------------
# Photo flash
# --------------------------------------------------------------------------
def apply_flash(frame: np.ndarray, intensity: float) -> np.ndarray:
    if intensity <= 0:
        return frame
    white = np.full_like(frame, 255)
    return cv2.addWeighted(frame, 1.0 - intensity, white, intensity, 0)


# --------------------------------------------------------------------------
# Debug overlay
# --------------------------------------------------------------------------
def draw_debug_overlay(frame: np.ndarray, lines: List[str], accent_color: Tuple[int, int, int], text_color: Tuple[int, int, int]) -> None:
    x, y0 = 16, 70
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
