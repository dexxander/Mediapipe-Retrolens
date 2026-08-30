# RETROLENS

A retro-futuristic, gesture-controlled webcam application. Native desktop
app — no browser, no web server. Just Python, OpenCV, MediaPipe, and NumPy,
talking straight to your webcam.

```
retro electronic camera + experimental computer vision
+ cyberpunk HUD + analog film + gesture-controlled interaction
```

## Features

- Live webcam feed in a native OpenCV window, mirrored like a selfie camera
- MediaPipe hand tracking (up to 2 hands, 21-point landmarks) with a subtle
  retro landmark overlay, always on
- A two-hand **"L" frame gesture** that puts RETROLENS into shot-composing
  mode: form an L with each hand (thumb + index extended), bring them
  together, then pull apart to size a frame. While that frame is held, the
  retro face effect, its glowing box, and the head-pose readout are all
  active — drop your hands and it's back to the plain live feed. It's a
  deliberate "compose the shot" mode, not a permanent filter.
- Three retro treatments applied inside that frame: **MONO** (CLAHE
  monochrome), **KONTRAS** (punchy LUT-based contrast), **FILM** (grain,
  flicker, scanlines, vignette)
- 3-second countdown with automatic photo capture — hold the frame through
  a capture and it keeps shooting again every 3 seconds for as long as you
  hold it; the box and countdown flash magenta for the final second as a
  "get ready" cue
- Real-time head pose estimation (ROLL / PITCH / YAW) via `solvePnP`,
  shown while the frame is held
- Air drawing: point your index finger and hold it — a small ring fills in
  around your fingertip ("MODE GAMBAR"), and once it completes, a glowing
  neon trail follows your finger with a bright hot-tip. The trail stays on
  screen until you clear it.
- Retro glitch-style HUD: REC indicator, timestamp, FPS, mode, photo
  counter, and an on-screen hint, all rendered with a cyan/magenta
  chromatic-aberration fringe, floating directly over the video (no panels)
- A sequential photo counter and auto-created `captures/` folder

## Requirements

- Python 3.10+
- A webcam
- Windows, macOS, or Linux (developed with Windows + VS Code in mind)
- An internet connection **the first time you run it**, to download two
  small MediaPipe model files (see [About the MediaPipe models](#about-the-mediapipe-models) below)

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

A native window titled **RETROLENS** opens and the webcam starts
automatically. The first launch will briefly say "Downloading required
model..." in the console while it fetches the hand and face models
(a few MB total) — every run after that is fully offline.

## Controls

| Key   | Action                                    |
|-------|--------------------------------------------|
| `ESC` | Quit                                        |
| `C`   | Clear the air-drawing canvas                |
| `M`   | Switch to MONO face effect                  |
| `K`   | Switch to KONTRAS face effect               |
| `F`   | Switch to FILM face effect                  |
| `D`   | Toggle the debug overlay                    |
| `R`   | Reset gesture / drawing / pose state        |
| `S`   | Save the current frame as a photo manually  |

You can also close the window normally (the [X] button) instead of
pressing `ESC`.

## Gestures

**"L" frame → automatic photo**

```
BENTUK 'L' DUA TANGAN, SATUKAN, LALU TARIK
(form an 'L' with both hands, bring together, then pull apart)

left index ------------------- right index      <- top edge
     |                              |
     |            FRAME             |
     |                              |
left thumb ------------------- right thumb      <- bottom edge
```

Form an L-shape with each hand — thumb and index extended, other fingers
folded — then hold both hands up so your index fingers form the top
corners and your thumbs form the bottom corners of a frame. Hold it steady
for 3 seconds: the frame lights up amber with corner markers, a countdown
(3, 2, 1) counts down inside it turning magenta on the final second, and a
photo is captured with a flash. Keep holding and it fires again every 3
seconds; drop the gesture and the countdown cancels — nothing is saved.

**Index-finger air drawing**

Extend just your index finger (other fingers folded) and hold still for
a moment — a ring fills in around your fingertip. Once it completes, move
your finger and a glowing neon trail follows it. Lift the gesture and the
trail stays put; press `C` (or `R`) to clear it.

## Photos

Saved to `captures/` as `retrolens_001.jpg`, `retrolens_002.jpg`, and so
on. The counter picks up where it left off if you restart the app, so
existing photos are never overwritten.

## Configuration

Every tunable value — camera index and resolution, detection confidence
thresholds, gesture sensitivity, drawing smoothing, effect strength, HUD
colors, and more — lives in `config.py`. Nothing else in the codebase
hard-codes these values, so it's a good first stop for tweaking the feel of
the app.

If your webcam isn't device `0`, change `CAMERA_INDEX` in `config.py`. If
the app feels sluggish on your machine, try lowering `FRAME_WIDTH` /
`FRAME_HEIGHT`.

## Project structure

```
retrolens/
├── main.py             entry point + main loop
├── config.py            all tunable constants
├── camera.py             webcam wrapper
├── hand_tracker.py       MediaPipe hand landmarks + retro visualization
├── face_tracker.py       MediaPipe face landmarks + bounding box
├── pose_estimator.py     head pose (roll/pitch/yaw) via solvePnP
├── gesture_detector.py   finger-extension, L-shape, frame gesture, countdown
├── effects.py             MONO / KONTRAS / FILM treatments (abu/ke_bgr/m_*)
├── drawing.py             air-drawing canvas + neon glow compositing
├── capture.py             photo saving, numbering, flash state
├── ui.py                  HUD rendering (glitch text, gesture box, HUD...)
├── utils.py               geometry/smoothing helpers + model downloader
├── requirements.txt
├── README.md
├── assets/                reserved for optional custom assets; also
│   └── models/            where the MediaPipe .task model files are cached
└── captures/              saved photos (created automatically)
```

## About the MediaPipe models

MediaPipe's current Python API (the "Tasks" API) loads its hand and face
models from small `.task` bundle files rather than shipping them inside the
`mediapipe` package itself. `utils.ensure_model()` downloads these
automatically into `assets/models/` the first time you run RETROLENS and
reuses the cached copy on every run after that.

If your machine has no internet access at all, download these two files
manually and place them in `assets/models/`:

- `hand_landmarker.task` —
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- `face_landmarker.task` —
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

## Troubleshooting

**"Unable to open webcam"** — make sure no other application (Zoom, Teams,
another instance of RETROLENS...) is using the camera, and that
`CAMERA_INDEX` in `config.py` matches your device (try `1` if `0` doesn't
work).

**Low FPS** — lower `FRAME_WIDTH` / `FRAME_HEIGHT` in `config.py`. RETROLENS
runs entirely on CPU by design (no GPU/CUDA required), so performance
scales with resolution and how much your machine has running alongside it.

**Frame gesture won't trigger** — each hand needs to clearly form an L
(thumb *and* index extended, other three fingers folded), index fingers
above thumbs, left hand on the left / right hand on the right, and a
reasonably sized frame. Turn on debug mode (`D`) to watch
`Frame Gesture: TRUE/false` update live while you adjust your hands.

**Drawing won't start** — hold the pointing pose (index out, others folded)
steady for a moment; the ring around your fingertip needs to fill
completely before the trail begins. This is deliberate, so a passing
finger doesn't leave accidental marks.

**Drawing trail feels laggy or jittery** — tune `DRAWING_SMOOTHING` in
`config.py`; higher values track your finger more directly, lower values
smooth out more jitter at the cost of a little lag.
