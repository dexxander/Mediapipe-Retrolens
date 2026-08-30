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

import math
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
    HoldCharge,
    detect_frame_gesture,
    is_drawing_gesture,
)
from hand_tracker import HandTracker
from pose_estimator import HeadPoseEstimator
from utils import FPSCounter, clamp_point

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
    fps: float,
    mode: str,
    gesture_valid: bool,
    countdown_remaining: Optional[float],
    drawing_hand_present: bool,
    charge_progress: float,
    drawing_active: bool,
    stroke_count: int,
    roll: Optional[float],
    pitch: Optional[float],
    yaw: Optional[float],
) -> List[str]:
    return [
        f"Hands: {hands_count}",
        f"FPS: {fps:.0f}",
        f"Mode: {mode}",
        f"Frame Gesture: {'TRUE' if gesture_valid else 'false'}",
        f"Countdown: {countdown_remaining:.1f}" if countdown_remaining is not None else "Countdown: --",
        f"Draw pose: {'yes' if drawing_hand_present else 'no'}  charge: {charge_progress:.2f}",
        f"Drawing: {'ON' if drawing_active else 'off'}  Goresan: {stroke_count}",
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
    photo_capture = PhotoCapture(config.CAPTURES_DIR, config.FLASH_DURATION_FRAMES, config.CAPTURE_TEXT_FRAMES)
    drawing_canvas = DrawingCanvas(
        camera.width, camera.height, config.DRAWING_COLOR, config.DRAWING_THICKNESS, config.DRAWING_SMOOTHING
    )
    countdown = FrameCountdown(config.COUNTDOWN_SECONDS, config.GESTURE_MISS_TOLERANCE_FRAMES)
    draw_charge = HoldCharge(config.DRAW_CHARGE_SECONDS)
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
                    draw_charge=draw_charge,
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
                draw_charge.reset()
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
    draw_charge: HoldCharge,
) -> np.ndarray:
    """
    One full pass: detect hands/face -> two-hand frame gesture (which
    drives the retro effect + pose readout + countdown) -> air-drawing ->
    HUD. Returns the frame to display.

    Design note: the retro face effect, its "MODE WxH" box, and the
    ROLL/PITCH/YAW readout are only active *while the two-hand L-gesture is
    engaged* — this is a "compose the shot" mode, not a permanent filter.
    Outside of it, RETROLENS is just the live color feed with hand
    tracking and the standard HUD chrome.
    """
    height, width = frame.shape[:2]
    drawing_canvas.resize(width, height)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hand_tracker.process(frame_rgb, t)
    face_result = face_tracker.process(frame_rgb, t)

    hands_px = hand_tracker.get_landmarks_px(hand_result, width, height)
    face_landmarks_px = face_tracker.get_landmarks_px(face_result, width, height)

    # --- Two-hand "L" frame gesture + countdown ---------------------------
    gesture_result = detect_frame_gesture(
        hands_px, width, height,
        config.MIN_FRAME_WIDTH_RATIO, config.MIN_FRAME_HEIGHT_RATIO,
        config.MIN_FRAME_ASPECT, config.MAX_FRAME_ASPECT,
        config.FINGER_EXTENSION_THRESHOLD, config.THUMB_EXTENSION_THRESHOLD,
    )
    remaining = countdown.update(gesture_result.valid, t)
    countdown_running = remaining is not None
    is_final_second = countdown_running and remaining <= config.COUNTDOWN_FINAL_SECOND
    accent_color = config.FINAL_SECOND_COLOR if is_final_second else config.ACCENT_COLOR

    # --- Head pose: only computed/shown while composing a shot ------------
    roll = pitch = yaw = None
    if countdown_running and face_landmarks_px:
        pose = pose_estimator.estimate(face_landmarks_px, width, height)
        if pose is not None:
            roll, pitch, yaw = pose
    if not countdown_running:
        pose_estimator.reset()

    # --- Retro effect: applied to the gesture-derived box, same window ----
    if gesture_result.valid:
        xs = (gesture_result.top_left[0], gesture_result.top_right[0], gesture_result.bottom_left[0], gesture_result.bottom_right[0])
        ys = (gesture_result.top_left[1], gesture_result.top_right[1], gesture_result.bottom_left[1], gesture_result.bottom_right[1])
        gx1, gy1 = clamp_point(int(min(xs)), int(min(ys)), width, height)
        gx2, gy2 = clamp_point(int(max(xs)), int(max(ys)), width, height)
        if (gx2 - gx1) > config.MIN_ROI_SIZE and (gy2 - gy1) > config.MIN_ROI_SIZE:
            roi = frame[gy1:gy2, gx1:gx2]
            if roi.size > 0:
                frame[gy1:gy2, gx1:gx2] = apply_effect(roi, mode, t)

    # --- Hand landmark visualization ---------------------------------------
    hand_tracker.draw_landmarks(frame, hands_px)

    # --- Air drawing: hold-to-charge, then a live neon trail ---------------
    drawing_hand = None
    for hand in hands_px:
        if is_drawing_gesture(hand, config.FINGER_EXTENSION_THRESHOLD, config.THUMB_EXTENSION_THRESHOLD, config.DRAWING_FOLDED_FINGERS_REQUIRED):
            drawing_hand = hand
            break

    charge_progress = draw_charge.update(drawing_hand is not None, t)
    drawing_active = draw_charge.charged and drawing_hand is not None
    if drawing_active:
        tip = drawing_hand[INDEX_TIP]
        drawing_canvas.add_point(tip[0], tip[1])
    else:
        drawing_canvas.reset_stroke()

    frame = composite_drawing(
        frame, drawing_canvas.layer, config.DRAWING_GLOW_LARGE_SIGMA, config.DRAWING_GLOW_SMALL_SIGMA,
        tip=drawing_canvas.current_tip if drawing_active else None,
        tip_color=config.DRAWING_TIP_COLOR,
    )

    # Snapshot for saving *before* any HUD chrome is drawn on top.
    clean_frame = frame.copy()

    if countdown.triggered:
        photo_frame = clean_frame.copy()
        if config.BURN_IN_TIMESTAMP:
            photo_frame = ui.stamp_photo_timestamp(photo_frame, config.WARN_COLOR)
        saved_path = photo_capture.save(photo_frame)
        if saved_path:
            print(f"[RETROLENS] Photo saved: {saved_path.name}")

    # --- Gesture box decoration + big countdown number ---------------------
    if gesture_result.valid:
        ui.draw_gesture_box(
            frame, gesture_result.top_left, gesture_result.top_right,
            gesture_result.bottom_left, gesture_result.bottom_right,
            mode, accent_color, config.CYAN_COLOR,
        )

    show_capture_text = photo_capture.consume_capture_text()
    if gesture_result.valid and (remaining is not None) and (remaining > 0 or show_capture_text):
        centroid = (
            (gesture_result.top_left[0] + gesture_result.top_right[0] + gesture_result.bottom_left[0] + gesture_result.bottom_right[0]) / 4,
            (gesture_result.top_left[1] + gesture_result.top_right[1] + gesture_result.bottom_left[1] + gesture_result.bottom_right[1]) / 4,
        )
        text = "CAPTURE" if show_capture_text else str(max(1, math.ceil(remaining)))
        ui.draw_countdown_number(frame, text, centroid, accent_color)

    # --- Draw-mode "hold to activate" charge ring ---------------------------
    if drawing_hand is not None and not draw_charge.charged:
        tip_pt = (int(drawing_hand[INDEX_TIP][0]), int(drawing_hand[INDEX_TIP][1]))
        ui.draw_charge_ring(frame, tip_pt, charge_progress, config.CYAN_COLOR, config.TEXT_COLOR)

    # --- Photo flash ----------------------------------------------------------
    flash_intensity = photo_capture.consume_flash()
    if flash_intensity > 0:
        frame = ui.apply_flash(frame, flash_intensity)

    # --- Static HUD chrome ------------------------------------------------------
    ui.draw_crosshair(frame, config.DIM_TEXT_COLOR)
    ui.draw_rec_indicator(frame, t, config.REC_COLOR, config.TEXT_COLOR)
    ui.draw_timestamp(frame, config.TEXT_COLOR)

    main_label = config.HUD_LABEL_DRAWING if drawing_canvas.has_content else config.HUD_LABEL_IDLE
    secondary_line = None
    if not countdown_running:
        if drawing_hand is not None:
            state_word = "GAMBAR" if drawing_active else "SIAP"
            secondary_line = f"{state_word}  GORESAN {drawing_canvas.stroke_count}"
        else:
            secondary_line = config.HINT_TEXT
    ui.draw_bottom_left(frame, main_label, mode, fps, photo_capture.count, secondary_line, config.TEXT_COLOR)

    if countdown_running:
        ui.draw_pose_readout(frame, roll, pitch, yaw, accent_color)

    if debug_mode:
        lines = _build_debug_lines(
            hands_count=len(hands_px),
            fps=fps,
            mode=mode,
            gesture_valid=gesture_result.valid,
            countdown_remaining=remaining,
            drawing_hand_present=drawing_hand is not None,
            charge_progress=charge_progress,
            drawing_active=drawing_active,
            stroke_count=drawing_canvas.stroke_count,
            roll=roll, pitch=pitch, yaw=yaw,
        )
        ui.draw_debug_overlay(frame, lines, config.ACCENT_COLOR, config.TEXT_COLOR)

    return frame


if __name__ == "__main__":
    main()
