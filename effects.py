"""
effects.py
==========
The three retro face treatments applied to the face ROI only — never the
whole frame. Function names (abu / ke_bgr / m_mono / m_kontras / m_film)
match the app's own internal vocabulary on purpose.

    abu       -> BGR to grayscale ("abu-abu" = grey)
    ke_bgr    -> grayscale back to 3-channel BGR ("ke bgr" = to BGR)
    m_mono    -> CLAHE-enhanced monochrome
    m_kontras -> punchy S-curve contrast via a LUT
    m_film    -> desaturated + grain + flicker + scanlines/vignette

All three share one CLAHE instance and cache their expensive-to-build
lookup tables / noise arrays at module scope, so the hot path of the main
loop never rebuilds them.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np

import config

MODE_MONO = "MONO"
MODE_KONTRAS = "KONTRAS"
MODE_FILM = "FILM"

_clahe = cv2.createCLAHE(
    clipLimit=config.CLAHE_CLIP_LIMIT,
    tileGridSize=config.CLAHE_GRID_SIZE,
)


def abu(roi: np.ndarray) -> np.ndarray:
    """Convert a BGR ROI to single-channel grayscale."""
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def ke_bgr(gray: np.ndarray) -> np.ndarray:
    """Convert a single-channel grayscale image back to 3-channel BGR."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------
# MONO
# --------------------------------------------------------------------------
def m_mono(roi: np.ndarray, t: float) -> np.ndarray:
    """CLAHE-enhanced monochrome."""
    return ke_bgr(_clahe.apply(abu(roi)))


# --------------------------------------------------------------------------
# KONTRAS (LUT-based S-curve contrast)
# --------------------------------------------------------------------------
_kontras_lut_cache: Dict[float, np.ndarray] = {}


def _build_contrast_lut(intensity: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float64)
    curve = 255.0 / (1.0 + np.exp(-intensity * (x - 128.0) / 32.0))
    return np.clip(curve, 0, 255).astype(np.uint8)


def m_kontras(roi: np.ndarray, t: float, intensity: float = config.KONTRAS_INTENSITY) -> np.ndarray:
    """Dramatic S-curve contrast, applied via a cached lookup table."""
    gray = abu(roi)
    gray = _clahe.apply(gray)
    lut = _kontras_lut_cache.get(intensity)
    if lut is None:
        lut = _build_contrast_lut(intensity)
        _kontras_lut_cache[intensity] = lut
    gray = cv2.LUT(gray, lut)
    return ke_bgr(gray)


# --------------------------------------------------------------------------
# FILM (grain + flicker + optional scanlines/vignette)
# --------------------------------------------------------------------------
_GRAIN_FRAMES = 6  # small pool of pre-generated grain frames, cycled by time
_grain_cache: Dict[Tuple[int, int, int], np.ndarray] = {}


def _get_grain_stack(h: int, w: int, strength: int, seed: int = 7) -> np.ndarray:
    """
    A (N, h, w) stack of pre-generated Gaussian noise, cached by (h, w,
    strength). Cycling through the N cached frames each render gives the
    grain a "living" feel without ever calling np.random inside the hot
    per-frame path after the first time a given ROI size is seen.
    """
    key = (h, w, strength)
    stack = _grain_cache.get(key)
    if stack is None:
        rng = np.random.default_rng(seed)
        stack = np.stack(
            [rng.normal(0, strength, (h, w)).astype(np.int16) for _ in range(_GRAIN_FRAMES)]
        )
        _grain_cache[key] = stack
    return stack


def _apply_vignette(gray_f: np.ndarray) -> np.ndarray:
    h, w = gray_f.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    max_r = math.hypot(cx, cy)
    r = np.hypot(xx - cx, yy - cy) / max_r
    mask = 1.0 - 0.35 * np.clip(r - 0.55, 0, None) / 0.45
    return gray_f * np.clip(mask, 0.55, 1.0)


def m_film(roi: np.ndarray, t: float, grain_strength: int = config.GRAIN_STRENGTH) -> np.ndarray:
    """Analog-film look: grayscale + CLAHE + grain + subtle flicker + optional scanlines/vignette."""
    gray = abu(roi)
    gray = _clahe.apply(gray)
    h, w = gray.shape

    flicker = 1.0 + 0.025 * math.sin(t * 6.0) + 0.01 * math.sin(t * 17.3)
    gray_f = gray.astype(np.float32) * flicker

    if config.FILM_VIGNETTE:
        gray_f = _apply_vignette(gray_f)

    stack = _get_grain_stack(h, w, grain_strength)
    grain_index = int(t * config.GRAIN_FRAME_RATE) % _GRAIN_FRAMES
    gray_f = gray_f + stack[grain_index]

    gray_u8 = np.clip(gray_f, 0, 255).astype(np.uint8)
    bgr = ke_bgr(gray_u8)

    if config.FILM_SCANLINES:
        bgr[::3, :, :] = (bgr[::3, :, :].astype(np.float32) * 0.82).astype(np.uint8)

    return bgr


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
def apply_effect(roi: np.ndarray, mode: str, t: float) -> np.ndarray:
    """Route to the active effect function. Falls back to MONO for an unknown mode."""
    if mode == MODE_KONTRAS:
        return m_kontras(roi, t)
    if mode == MODE_FILM:
        return m_film(roi, t)
    return m_mono(roi, t)
