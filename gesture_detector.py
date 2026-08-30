"""
gesture_detector.py
====================
Pure geometry over MediaPipe's 21-point hand landmark topology. Nothing in
this file touches MediaPipe, OpenCV drawing, or timing — it only answers
questions like "is this finger extended?" and "do these two hands form a
valid framing gesture?" given pixel-space landmark points. That keeps it
easy to reason about and easy to unit-test.

Landmark indices (MediaPipe 21-point hand topology):
    0  wrist
    1-4    thumb   (CMC, MCP, IP, TIP)
    5-8    index   (MCP, PIP, DIP, TIP)
    9-12   middle  (MCP, PIP, DIP, TIP)
    13-16  ring    (MCP, PIP, DIP, TIP)
    17-20  pinky   (MCP, PIP, DIP, TIP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from utils import distance

Point = Tuple[float, float]

# --- Landmark indices -------------------------------------------------------
WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Fingertip lookup used by drawing / effects code elsewhere in the app.
FINGERTIPS = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}


# --------------------------------------------------------------------------
# Finger extension
# --------------------------------------------------------------------------
def _extension_ratio(points: Sequence[Point], tip_idx: int, pip_idx: int) -> float:
    """
    How far the tip has straightened away from the palm, relative to hand
    size. Using *distance from the wrist* (rather than a raw y-coordinate
    comparison) keeps this reasonably tolerant of a rotated or tilted hand:
    an extended finger's tip sits meaningfully farther from the wrist than
    its own pip joint does; a folded finger's tip sits at or near the pip.
    """
    wrist = points[WRIST]
    palm_size = distance(wrist, points[MIDDLE_MCP])
    if palm_size < 1e-6:
        return 0.0
    tip_dist = distance(wrist, points[tip_idx])
    pip_dist = distance(wrist, points[pip_idx])
    return (tip_dist - pip_dist) / palm_size


def is_finger_extended(
    points: Sequence[Point],
    tip_idx: int,
    pip_idx: int,
    threshold: float,
) -> bool:
    """True if the finger identified by (tip_idx, pip_idx) is extended."""
    return _extension_ratio(points, tip_idx, pip_idx) > threshold


def get_extended_fingers(
    points: Sequence[Point],
    finger_threshold: float,
    thumb_threshold: float,
) -> dict:
    """Returns {"thumb": bool, "index": bool, "middle": bool, "ring": bool, "pinky": bool}."""
    return {
        "thumb": is_finger_extended(points, THUMB_TIP, THUMB_IP, thumb_threshold),
        "index": is_finger_extended(points, INDEX_TIP, INDEX_PIP, finger_threshold),
        "middle": is_finger_extended(points, MIDDLE_TIP, MIDDLE_PIP, finger_threshold),
        "ring": is_finger_extended(points, RING_TIP, RING_PIP, finger_threshold),
        "pinky": is_finger_extended(points, PINKY_TIP, PINKY_PIP, finger_threshold),
    }


# --------------------------------------------------------------------------
# Drawing gesture: index finger out, the rest mostly folded
# --------------------------------------------------------------------------
def is_drawing_gesture(
    points: Sequence[Point],
    finger_threshold: float,
    thumb_threshold: float,
    folded_required: int,
    require_upward: bool = True,
) -> bool:
    """
    Tolerant "pointing" detector: the index finger must be extended, and at
    least `folded_required` of {middle, ring, pinky} must be folded. The
    thumb is deliberately ignored — most people can't fully tuck it away,
    and requiring that would make the gesture feel fragile.
    """
    fingers = get_extended_fingers(points, finger_threshold, thumb_threshold)
    if not fingers["index"]:
        return False
    if require_upward and not (
        points[INDEX_TIP][1] < points[INDEX_PIP][1]
        and points[INDEX_TIP][1] < points[WRIST][1]
    ):
        return False
    folded_count = sum(not fingers[name] for name in ("middle", "ring", "pinky"))
    return folded_count >= folded_required


# --------------------------------------------------------------------------
# "L" shape: thumb + index extended (perpendicular, like an L), the rest
# folded. Each hand must form this shape for the two-hand frame gesture.
# --------------------------------------------------------------------------
def is_l_shape(
    points: Sequence[Point],
    finger_threshold: float,
    thumb_threshold: float,
    folded_required: int = 2,
) -> bool:
    """True if thumb and index are both extended and at least `folded_required` of {middle, ring, pinky} are folded."""
    fingers = get_extended_fingers(points, finger_threshold, thumb_threshold)
    if not (fingers["thumb"] and fingers["index"]):
        return False
    folded_count = sum(not fingers[name] for name in ("middle", "ring", "pinky"))
    return folded_count >= folded_required


# --------------------------------------------------------------------------
# Two-hand framing gesture
# --------------------------------------------------------------------------
@dataclass
class FrameGestureResult:
    valid: bool
    top_left: Optional[Point] = None
    top_right: Optional[Point] = None
    bottom_left: Optional[Point] = None
    bottom_right: Optional[Point] = None
    width: float = 0.0
    height: float = 0.0


def _hand_center_x(hand_points: Sequence[Point]) -> float:
    return sum(p[0] for p in hand_points) / len(hand_points)


def detect_frame_gesture(
    hands_points: List[Sequence[Point]],
    frame_width: int,
    frame_height: int,
    min_width_ratio: float,
    min_height_ratio: float,
    min_aspect: float,
    max_aspect: float,
    finger_threshold: float,
    thumb_threshold: float,
) -> FrameGestureResult:
    """
    Detects the two-hand "L-frame" gesture: form an L with each hand (thumb
    + index extended). The four fingertips become a perspective-safe frame;
    the cross-hand connectors are left-index -> right-thumb and
    right-index -> left-thumb, matching the requested framing gesture.

        left index ------------------- right index
             |          FRAME               |
        left thumb ------------------- right thumb

    "Left"/"right" are determined by on-screen horizontal position (not
    MediaPipe's own handedness label), since the frame we detect on has
    already been mirrored for display — whichever detected hand sits
    further left on screen *is* the left hand as the user perceives it.
    """
    if len(hands_points) != 2:
        return FrameGestureResult(valid=False)

    if not all(is_l_shape(hand, finger_threshold, thumb_threshold) for hand in hands_points):
        return FrameGestureResult(valid=False)

    ordered = sorted(hands_points, key=_hand_center_x)
    left_hand, right_hand = ordered[0], ordered[1]

    left_index = left_hand[INDEX_TIP]
    left_thumb = left_hand[THUMB_TIP]
    right_index = right_hand[INDEX_TIP]
    right_thumb = right_hand[THUMB_TIP]

    # Left hand must actually be to the left of the right hand.
    if left_index[0] >= right_index[0] or left_thumb[0] >= right_thumb[0]:
        return FrameGestureResult(valid=False)

    # Each hand must read as an L: index above thumb in the camera image.
    if left_index[1] >= left_thumb[1] or right_index[1] >= right_thumb[1]:
        return FrameGestureResult(valid=False)

    width = max(abs(right_index[0] - left_index[0]), abs(right_thumb[0] - left_thumb[0]))
    height = max(abs(left_thumb[1] - left_index[1]), abs(right_thumb[1] - right_index[1]))

    if width < min_width_ratio * frame_width:
        return FrameGestureResult(valid=False)
    if height < min_height_ratio * frame_height:
        return FrameGestureResult(valid=False)

    aspect = width / max(height, 1e-6)
    if not (min_aspect <= aspect <= max_aspect):
        return FrameGestureResult(valid=False)

    return FrameGestureResult(
        valid=True,
        top_left=left_index,
        top_right=right_index,
        bottom_left=left_thumb,
        bottom_right=right_thumb,
        width=width,
        height=height,
    )


# --------------------------------------------------------------------------
# Countdown state machine
# --------------------------------------------------------------------------
class FrameCountdown:
    """
    Tracks how long the framing gesture has been held.

    A short "miss tolerance" absorbs the odd single-frame tracking dropout
    (completely normal with real-time hand tracking) without restarting the
    countdown from scratch — but a genuine, sustained drop of the gesture
    still cancels it, exactly as requested. Holding the gesture continuously
    through and past a capture starts the next countdown immediately —
    holding the frame is a deliberate "keep shooting" burst, one capture per
    `duration` seconds for as long as it's held.

    Elapsed time accumulates from small per-frame deltas, clamped to a sane
    maximum, rather than a raw `now - start_time` difference. A stalled
    process (a dropped frame, the OS pausing the window) could otherwise
    make a multi-second gap look like "held continuously", firing a phantom
    instant capture the moment tracking resumes.
    """

    _MAX_FRAME_DT = 0.25  # seconds; caps the damage a single stall can do

    def __init__(self, duration: float, miss_tolerance_frames: int) -> None:
        self.duration = duration
        self.miss_tolerance_frames = miss_tolerance_frames
        self.active = False
        self._elapsed = 0.0
        self._last_time: Optional[float] = None
        self._miss_streak = 0
        self.triggered = False  # True for exactly one update() call, when it completes

    def reset(self) -> None:
        self.active = False
        self._elapsed = 0.0
        self._last_time = None
        self._miss_streak = 0
        self.triggered = False

    def update(self, gesture_valid: bool, now: float) -> Optional[float]:
        """
        Call once per frame. Returns seconds remaining while the countdown
        is running, or None while idle. Sets self.triggered True for the
        one frame on which the countdown completes.
        """
        self.triggered = False

        if gesture_valid:
            self._miss_streak = 0
        elif self.active:
            self._miss_streak += 1
            if self._miss_streak > self.miss_tolerance_frames:
                self.reset()
                return None
        else:
            return None  # idle: nothing held, nothing to tick down

        if not self.active:
            self.active = True
            self._elapsed = 0.0
            self._last_time = now
            return self.duration

        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, self._MAX_FRAME_DT))
        self._last_time = now
        self._elapsed += dt

        remaining = self.duration - self._elapsed
        if remaining <= 0:
            self.triggered = True
            self._elapsed = 0.0  # immediately starts counting the next cycle if still held
            return 0.0
        return remaining


# --------------------------------------------------------------------------
# "Hold to activate" charge-up (used before air-drawing actually starts,
# shown as a filling ring around the fingertip: "DRAW MODE"). Unlike
# FrameCountdown this does not repeat — once charged it stays charged until
# the pointing gesture is lost, and must fully recharge after that.
# --------------------------------------------------------------------------
class HoldCharge:
    _MAX_FRAME_DT = 0.25

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self._elapsed = 0.0
        self._last_time: Optional[float] = None
        self.charged = False

    def reset(self) -> None:
        self._elapsed = 0.0
        self._last_time = None
        self.charged = False

    def update(self, held: bool, now: float) -> float:
        """
        Call once per frame with whether the activating pose is currently
        held. Returns progress in [0, 1] (0 when not held / just reset,
        1 once fully charged). Stays at 1 (self.charged stays True) for as
        long as `held` remains True, so the drawing doesn't stutter off for
        a single missed frame right at the charge boundary.
        """
        if not held:
            self.reset()
            return 0.0

        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, self._MAX_FRAME_DT))
        self._last_time = now
        self._elapsed = min(self.duration, self._elapsed + dt)

        progress = 1.0 if self.duration <= 0 else self._elapsed / self.duration
        if progress >= 1.0:
            self.charged = True
        return min(1.0, progress)
