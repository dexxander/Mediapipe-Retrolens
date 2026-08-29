# assets/

Reserved for optional custom assets (e.g. a custom font for the HUD, a
splash icon, etc.) — nothing in `main.py` currently requires anything here
except the `models/` subfolder.

## models/

Where `utils.ensure_model()` caches the MediaPipe `.task` model bundles
(`hand_landmarker.task`, `face_landmarker.task`) after downloading them on
first run. Safe to delete — they'll simply be re-downloaded next launch.
