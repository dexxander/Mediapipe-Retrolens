"""
RETROLENS
=========
A retro-futuristic, gesture-controlled webcam application.

Run with:
    python main.py

Native OpenCV window. No browser, no web framework — everything here is
plain Python, OpenCV, MediaPipe, and NumPy, driven straight from your
webcam. See README.md for controls and gestures.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import cv2
import numpy as np

import config
import ui
from camera import Camera, CameraError
from capture import PhotoCapture
from drawing import DrawingCanvas, composite_drawing
from effects import MODE_FILM, MODE_KONTRAS, MODE_MONO, apply_effect
from face_tracker import FaceTracker
from gesture_detector import (
    INDEX_TIP,
    FrameCountdown,
    detect_frame_gesture,
    is_drawing_gesture,
)
from hand_tracker import HandTracker
from pose_estimator import HeadPoseEstimator
from utils import FPSCounter

WINDOW_TITLE = "RETROLENS"


# --------------------------------------------------------------------------
# Startup error display — never let the user stare at a raw traceback
# --------------------------------------------------------------------------
def show_startup_error(message: str) -> None:
    print(f"[RETROLENS] ERROR: {message}")
    canvas = np.zeros((360, 720, 3), dtype=np.uint8)
    cv2.putText(canvas, "RETROLENS", (24, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, config.ACCENT_COLOR, 2, cv2.LINE_AA)
    y = 90
    for raw_line in message.split("\n"):
        line = raw_line.strip()
        if not line:
            y += 12
            continue
        cv2.putText(canvas, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        y += 22
    cv2.putText(canvas, "Press any key to exit...", (24, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.DIM_TEXT_COLOR, 1, cv2.LINE_AA)

    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_TITLE, canvas)
        cv2.waitKey(6000)
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless environment with no display — the console message above still stands


def _build_debug_lines(
    hands_count: int,
    face_detected: bool,
    fps: float,
    mode: str,
    drawing_active: bool,
    frame_gesture_valid: bool,
    countdown_remaining: Optional[float],
    roll: Optional[float],
    pitch: Optional[float],
    yaw: Optional[float],
) -> List[str]:
    return [
        f"Hands: {hands_count}",
        f"Face: {'detected' if face_detected else 'none'}",
        f"FPS: {fps:.0f}",
        f"Mode: {mode}",
        f"Drawing: {'ON' if drawing_active else 'off'}",
        f"Frame Gesture: {'TRUE' if frame_gesture_valid else 'false'}",
        f"Countdown: {countdown_remaining:.1f}" if countdown_remaining is not None else "Countdown: --",
        f"Roll: {roll:.1f}" if roll is not None else "Roll: --",
        f"Pitch: {pitch:.1f}" if pitch is not None else "Pitch: --",
        f"Yaw: {yaw:.1f}" if yaw is not None else "Yaw: --",
    ]


def main() -> None:
    print("RETROLENS")
    print("initializing...")

    # --- Phase 1: camera -----------------------------------------------
    try:
        camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    except CameraError as exc:
        show_startup_error(str(exc))
        sys.exit(1)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, camera.width, camera.height)
    splash = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    cv2.putText(splash, "RETROLENS", (40, camera.height // 2 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.4, config.ACCENT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(splash, "initializing...", (40, camera.height // 2 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.imshow(WINDOW_TITLE, splash)
    cv2.waitKey(1)

    # --- Phase 2: MediaPipe (may download model bundles on first run) --
    try:
        hand_tracker = HandTracker(
            models_dir=config.MODELS_DIR,
            max_hands=config.MAX_HANDS,
            detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            presence_confidence=config.HAND_PRESENCE_CONFIDENCE,
            tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
        )
        face_tracker = FaceTracker(
            models_dir=config.MODELS_DIR,
            max_faces=config.MAX_FACES,
            detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            presence_confidence=config.FACE_PRESENCE_CONFIDENCE,
            tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any init failure must not crash raw
        camera.release()
        cv2.destroyAllWindows()
        show_startup_error(f"MediaPipe initialization failed.\n{exc}")
        sys.exit(1)

    pose_estimator = HeadPoseEstimator(config.POSE_SMOOTHING_ALPHA, config.POSE_MAX_DEGREES)
    photo_capture = PhotoCapture(config.CAPTURES_DIR, config.FLASH_DURATION_FRAMES, config.CAPTURE_HOLD_FRAMES)
    drawing_canvas = DrawingCanvas(
        camera.width, camera.height, config.DRAWING_COLOR, config.DRAWING_THICKNESS, config.DRAWING_SMOOTHING
    )
    countdown = FrameCountdown(config.COUNTDOWN_SECONDS, config.GESTURE_MISS_TOLERANCE_FRAMES)
    fps_counter = FPSCounter(config.FPS_SMOOTHING)

    mode = MODE_MONO
    debug_mode = config.DEBUG_DEFAULT
    start_time = time.perf_counter()
    running = True

    try:
        while running:
            frame = camera.read()
            if frame is None:
                print("[RETROLENS] Camera disconnected. Exiting.")
                break

            try:
                frame = _process_frame(
                    frame=frame,
                    t=time.perf_counter() - start_time,
                    fps=fps_counter.tick(),
                    mode=mode,
                    debug_mode=debug_mode,
                    hand_tracker=hand_tracker,
                    face_tracker=face_tracker,
                    pose_estimator=pose_estimator,
                    photo_capture=photo_capture,
                    drawing_canvas=drawing_canvas,
                    countdown=countdown,
                )
            except Exception as exc:  # noqa: BLE001 - keep the live session alive across a bad frame
                print(f"[RETROLENS] Frame error (continuing): {exc}")
                cv2.imshow(WINDOW_TITLE, frame)
            else:
                cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                running = False
            elif key in (ord("c"), ord("C")):
                drawing_canvas.clear()
            elif key in (ord("m"), ord("M")):
                mode = MODE_MONO
            elif key in (ord("k"), ord("K")):
                mode = MODE_KONTRAS
            elif key in (ord("f"), ord("F")):
                mode = MODE_FILM
            elif key in (ord("d"), ord("D")):
                debug_mode = not debug_mode
            elif key in (ord("r"), ord("R")):
                drawing_canvas.clear()
                countdown.reset()
                pose_estimator.reset()
            elif key in (ord("s"), ord("S")):
                saved = photo_capture.save(ui.stamp_photo_timestamp(frame.copy(), config.WARN_COLOR))
                if saved:
                    print(f"[RETROLENS] Photo saved: {saved.name}")

            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                running = False
    finally:
        camera.release()
        hand_tracker.close()
        face_tracker.close()
        cv2.destroyAllWindows()
        print("[RETROLENS] Shut down cleanly.")


def _process_frame(
    frame: np.ndarray,
    t: float,
    fps: float,
    mode: str,
    debug_mode: bool,
    hand_tracker: HandTracker,
    face_tracker: FaceTracker,
    pose_estimator: HeadPoseEstimator,
    photo_capture: PhotoCapture,
    drawing_canvas: DrawingCanvas,
    countdown: FrameCountdown,
) -> np.ndarray:
    """One full pass: detect -> effect -> draw -> gesture -> HUD. Returns the frame to display."""
    height, width = frame.shape[:2]
    drawing_canvas.resize(width, height)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hand_tracker.process(frame_rgb, t)
    face_result = face_tracker.process(frame_rgb, t)

    hands_px = hand_tracker.get_landmarks_px(hand_result, width, height)
    face_landmarks_px = face_tracker.get_landmarks_px(face_result, width, height)

    # --- Face effect (ROI only) -----------------------------------------
    roll = pitch = yaw = None
    face_box = None
    if face_landmarks_px:
        face_box = face_tracker.get_bounding_box(face_landmarks_px, width, height, config.FACE_ROI_MARGIN)
        pose = pose_estimator.estimate(face_landmarks_px, width, height)
        if pose is not None:
            roll, pitch, yaw = pose

        x1, y1, x2, y2 = face_box
        if (x2 - x1) > config.MIN_ROI_SIZE and (y2 - y1) > config.MIN_ROI_SIZE:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                frame[y1:y2, x1:x2] = apply_effect(roi, mode, t)
    else:
        pose_estimator.reset()

    # --- Hand landmark visualization ------------------------------------
    hand_tracker.draw_landmarks(frame, hands_px)

    # --- Air drawing gesture ---------------------------------------------
    drawing_active = False
    for hand in hands_px:
        if is_drawing_gesture(hand, config.FINGER_EXTENSION_THRESHOLD, config.THUMB_EXTENSION_THRESHOLD, config.DRAWING_FOLDED_FINGERS_REQUIRED):
            tip = hand[INDEX_TIP]
            drawing_canvas.add_point(tip[0], tip[1])
            drawing_active = True
            break
    if not drawing_active:
        drawing_canvas.reset_stroke()

    frame = composite_drawing(frame, drawing_canvas.layer, config.DRAWING_GLOW_LARGE_SIGMA, config.DRAWING_GLOW_SMALL_SIGMA)

    # Snapshot for saving *before* any HUD chrome is drawn on top.
    clean_frame = frame.copy()

    # --- Two-hand framing gesture + countdown -----------------------------
    gesture_result = detect_frame_gesture(
        hands_px, width, height,
        config.MIN_FRAME_WIDTH_RATIO, config.MIN_FRAME_HEIGHT_RATIO,
        config.MIN_FRAME_ASPECT, config.MAX_FRAME_ASPECT,
    )
    remaining = countdown.update(gesture_result.valid, t)

    if countdown.triggered:
        photo_frame = clean_frame.copy()
        if config.BURN_IN_TIMESTAMP:
            photo_frame = ui.stamp_photo_timestamp(photo_frame, config.WARN_COLOR)
        saved_path = photo_capture.save(photo_frame)
        if saved_path:
            print(f"[RETROLENS] Photo saved: {saved_path.name}")

    show_capture_text = photo_capture.consume_capture_text()
    if gesture_result.valid:
        ui.draw_virtual_frame(frame, gesture_result, config.FRAME_GESTURE_COLOR)
    if remaining is not None or show_capture_text:
        ui.draw_countdown(frame, remaining, gesture_result, show_capture_text, config.FRAME_GESTURE_COLOR)

    flash_intensity = photo_capture.consume_flash()
    if flash_intensity > 0:
        frame = ui.apply_flash(frame, flash_intensity)

    # --- Static HUD chrome --------------------------------------------------
    if config.GLOBAL_VIGNETTE:
        frame = ui.apply_global_vignette(frame, config.GLOBAL_VIGNETTE_STRENGTH)

    ui.draw_crosshair(frame, config.DIM_TEXT_COLOR)
    ui.draw_rec_indicator(frame, t, config.REC_COLOR, config.TEXT_COLOR)
    ui.draw_timestamp(frame, config.TEXT_COLOR)
    if face_box is not None:
        ui.draw_roi_label(frame, face_box, mode, config.ACCENT_COLOR)
    ui.draw_bottom_hud(
        frame, config.HUD_LABEL, mode, fps, photo_capture.count,
        roll, pitch, yaw, config.ACCENT_COLOR, config.TEXT_COLOR, config.DIM_TEXT_COLOR,
    )

    if debug_mode:
        lines = _build_debug_lines(
            hands_count=len(hands_px),
            face_detected=face_landmarks_px is not None,
            fps=fps,
            mode=mode,
            drawing_active=drawing_active,
            frame_gesture_valid=gesture_result.valid,
            countdown_remaining=remaining,
            roll=roll, pitch=pitch, yaw=yaw,
        )
        ui.draw_debug_overlay(frame, lines, config.ACCENT_COLOR, config.TEXT_COLOR)

    return frame


if __name__ == "__main__":
    main()
