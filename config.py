"""
config.py
=========
Central configuration for RETROLENS.

Every tunable constant lives here so the rest of the application never
hard-codes a magic number. Tweak these values to adapt RETROLENS to a
different camera, a slower/faster machine, or a different visual taste.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CAPTURES_DIR = PROJECT_ROOT / "captures"
MODELS_DIR = PROJECT_ROOT / "assets" / "models"

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --------------------------------------------------------------------------
# MediaPipe — hands
# --------------------------------------------------------------------------
MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.6
HAND_PRESENCE_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5

# --------------------------------------------------------------------------
# MediaPipe — face
# --------------------------------------------------------------------------
MAX_FACES = 1
FACE_DETECTION_CONFIDENCE = 0.6
FACE_PRESENCE_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5

# --------------------------------------------------------------------------
# Face ROI / effects
# --------------------------------------------------------------------------
FACE_ROI_MARGIN = 0.28          # fraction of box size added on every side
MIN_ROI_SIZE = 24               # px — below this we skip processing entirely

CLAHE_CLIP_LIMIT = 2.5
CLAHE_GRID_SIZE = (8, 8)

KONTRAS_INTENSITY = 1.6         # steepness of the contrast S-curve
GRAIN_STRENGTH = 11             # std-dev of film grain noise (0-255 scale)
GRAIN_FRAME_RATE = 18.0         # how many grain "frames" cycle per second
FILM_SCANLINES = True           # scanlines drawn only inside FILM face ROI
FILM_VIGNETTE = True            # soft vignette drawn only inside FILM face ROI

# --------------------------------------------------------------------------
# Head pose estimation
# --------------------------------------------------------------------------
POSE_SMOOTHING_ALPHA = 0.35     # higher = snappier, lower = smoother/laggier
POSE_MAX_DEGREES = 90.0         # clamp displayed angles to +/- this value

# --------------------------------------------------------------------------
# Gesture detection
# --------------------------------------------------------------------------
FINGER_EXTENSION_THRESHOLD = 0.16
THUMB_EXTENSION_THRESHOLD = 0.09
DRAWING_FOLDED_FINGERS_REQUIRED = 2   # of {middle, ring, pinky} must be folded

MIN_FRAME_WIDTH_RATIO = 0.12    # virtual frame must span at least this
MIN_FRAME_HEIGHT_RATIO = 0.10   # fraction of the camera frame
MIN_FRAME_ASPECT = 0.3
MAX_FRAME_ASPECT = 4.0

COUNTDOWN_SECONDS = 3.0
GESTURE_MISS_TOLERANCE_FRAMES = 5     # brief dropout grace before cancelling

# --------------------------------------------------------------------------
# Air drawing
# --------------------------------------------------------------------------
DRAWING_SMOOTHING = 0.35        # weight given to the new point each frame
DRAWING_THICKNESS = 4
DRAWING_COLOR = (255, 255, 0)   # cyan, BGR
DRAWING_GLOW_LARGE_SIGMA = 18
DRAWING_GLOW_SMALL_SIGMA = 6

# --------------------------------------------------------------------------
# Photo capture
# --------------------------------------------------------------------------
FLASH_DURATION_FRAMES = 6
CAPTURE_HOLD_FRAMES = 10        # how long "CAPTURE" text stays on screen
BURN_IN_TIMESTAMP = True        # stamp saved photos with date/time

# --------------------------------------------------------------------------
# UI / HUD
# --------------------------------------------------------------------------
HUD_LABEL = "Lensa"
FPS_SMOOTHING = 0.9
DEBUG_DEFAULT = False
GLOBAL_VIGNETTE = True
GLOBAL_VIGNETTE_STRENGTH = 0.35

ACCENT_COLOR = (255, 230, 60)   # cyan/electric-blue accent, BGR
REC_COLOR = (40, 40, 255)       # red, BGR
TEXT_COLOR = (235, 235, 235)    # near-white
DIM_TEXT_COLOR = (160, 160, 160)
WARN_COLOR = (0, 165, 255)      # amber, BGR
FRAME_GESTURE_COLOR = (255, 240, 60)  # virtual frame lines, BGR
