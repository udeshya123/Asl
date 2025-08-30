# web_app.py
"""
Production-grade Flask MJPEG server for the SLR pipeline.

Key improvements:
- Dynamic, percentage-based layout (adapts to canvas size).
- Bounds-safe placement helper that crops instead of crashing.
- Robust fallbacks (missing resources / camera not opened).
- Healthcheck endpoint.
- Clean shutdown handling.
"""

import os
import csv
import copy
import time
import signal
import threading
import numpy as np

import cv2 as cv
import mediapipe as mp
from dotenv import load_dotenv
from flask import Flask, Response, render_template_string, jsonify

# ---- Project imports (unchanged from your app) ----
from slr.model.classifier import KeyPointClassifier
from slr.utils.args import get_args
from slr.utils.cvfpscalc import CvFpsCalc
from slr.utils.landmarks import draw_landmarks
from slr.utils.draw_debug import (
    get_result_image, get_fps_log_image,
    draw_bounding_rect, draw_hand_label,
    show_fps_log, show_result
)
from slr.utils.pre_process import (
    calc_bounding_rect, calc_landmark_list, pre_process_landmark
)
from slr.utils.logging import (
    log_keypoints, get_dict_form_list, get_mode
)

# ---------- Flask setup ----------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Sign Language Recognition</title>
    <style>
      :root { color-scheme: dark; }
      * { box-sizing: border-box; }
      html, body { height: 100%; margin: 0; }
      body {
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        background: #0f1222;
        color: #e6e6e6;
        display: flex;
        flex-direction: column;
      }
      header {
        padding: 12px 16px;
        background: #131738;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25);
      }
      .row { display: flex; gap: 12px; align-items: center; }
      .pill { background: #2a346b; padding: 4px 10px; border-radius: 999px; font-size: 12px; letter-spacing: .8px; }
      main {
        flex: 1;
        display: grid;
        place-items: center;
        padding: 18px;
      }
      .frame {
        width: min(96vw, 1600px);
        aspect-ratio: 16 / 9;   /* matches default canvas 1600x900 */
        background: #191f45;
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        display: grid;
        place-items: center;
        overflow: hidden;
      }
      .frame img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        user-select: none;
        -webkit-user-drag: none;
      }
      footer {
        text-align:center;
        font-size: 12px;
        color: #9aa5d1;
        padding: 8px 0 16px;
      }
      a { color: #9aa5d1; }
    </style>
  </head>
  <body>
    <header>
      <div class="row">
        <div class="pill">LIVE</div>
        <div>Sign Language Recognition (MJPEG)</div>
      </div>
    </header>
    <main>
      <div class="frame">
        <img src="/stream" alt="Sign Language Recognition Stream"/>
      </div>
    </main>
    <footer>
      <a href="/health" target="_blank" rel="noopener">Healthcheck</a>
    </footer>
  </body>
</html>
"""

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def place_region_safely(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> None:
    """
    Paste fg into bg at (x, y), cropping if needed so we never go OOB.
    Modifies bg in-place. If no overlap, it's a no-op.
    """
    if bg is None or fg is None:
        return
    H, W = bg.shape[:2]
    h, w = fg.shape[:2]

    # Compute destination slice (clamped)
    x1 = clamp(x, 0, W)
    y1 = clamp(y, 0, H)
    x2 = clamp(x + w, 0, W)
    y2 = clamp(y + h, 0, H)

    # If nothing visible, skip
    if x2 <= x1 or y2 <= y1:
        return

    # Compute corresponding source crop
    src_x1 = x1 - x
    src_y1 = y1 - y
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    bg[y1:y2, x1:x2] = fg[src_y1:src_y2, src_x1:src_x2]


class SLRProcessor:
    """
    Encapsulates camera, MediaPipe pipeline, and frame composition.
    Designed to be robust and bounds-safe.
    """
    def __init__(self):
        print("INFO: Initializing System (web mode)")
        load_dotenv()

        # Args (reuse your CLI defaults)
        args = get_args()
        self.CAP_DEVICE = 0
        self.CAP_WIDTH = args.width
        self.CAP_HEIGHT = args.height

        # Modes / flags
        self.USE_STATIC_IMAGE_MODE = True
        self.MAX_NUM_HANDS = args.max_num_hands
        self.MIN_DETECTION_CONFIDENCE = args.min_detection_confidence
        self.MIN_TRACKING_CONFIDENCE = args.min_tracking_confidence
        self.USE_BRECT = args.use_brect
        self.MODE = args.mode
        self.DEBUG = int(os.environ.get("DEBUG", "0")) == 1

        # Counters/labels
        keypoint_file = "slr/model/keypoint.csv"
        self.counter_obj = get_dict_form_list(keypoint_file)

        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.USE_STATIC_IMAGE_MODE,
            max_num_hands=self.MAX_NUM_HANDS,
            min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
        )

        # Classifier + labels
        self.keypoint_classifier = KeyPointClassifier()
        keypoint_labels_file = "slr/model/label.csv"
        with open(keypoint_labels_file, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in reader]

        # FPS calc
        self.cv_fps = CvFpsCalc(buffer_len=10)

        # Canvas (can be tuned via env)
        self.CANVAS_W = int(os.environ.get("CANVAS_W", "1600"))
        self.CANVAS_H = int(os.environ.get("CANVAS_H", "900"))

        # Themed background or solid fallback
        bg = cv.imread("resources/background.png")
        if bg is None:
            self.background_image = np.zeros((self.CANVAS_H, self.CANVAS_W, 3), dtype=np.uint8)
            self.background_image[:] = (25, 31, 69)  # dark indigo
        else:
            self.background_image = cv.resize(bg, (self.CANVAS_W, self.CANVAS_H), interpolation=cv.INTER_AREA)

        # Camera
        print("INFO: Opening Camera")
        self.cap = cv.VideoCapture(self.CAP_DEVICE, cv.CAP_DSHOW)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.CAP_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAP_HEIGHT)

        # Control
        self._stop = threading.Event()
        self._last_ok = False

        print("INFO: System is up & running")

    def stop(self):
        self._stop.set()

    def _compose(self, debug_image, result_image, fps_log_image) -> np.ndarray:
        """
        Compose the UI onto a canvas using dynamic, percentage-based layout.
        All placements are bounds-safe.
        """
        bg = self.background_image.copy()
        W, H = bg.shape[1], bg.shape[0]

        # Layout parameters (relative to canvas)
        MARGIN = int(0.025 * W)   # ~2.5% margin
        GAP = int(0.0125 * W)     # ~1.25% gaps
        MAIN_W = int(0.80 * W)    # main view ~80% canvas width
        MAIN_H = int(MAIN_W * 9 / 16)  # keep 16:9
        MAIN_X = MARGIN
        MAIN_Y = int(0.08 * H)    # ~8% from top

        # Ensure main fits vertically; if not, shrink it
        if MAIN_Y + MAIN_H + MARGIN > H:
            MAIN_H = max(100, H - MAIN_Y - MARGIN)
            MAIN_W = int(MAIN_H * 16 / 9)

        # Resize overlays with safe interpolation
        main = cv.resize(debug_image, (MAIN_W, MAIN_H), interpolation=cv.INTER_LINEAR)
        place_region_safely(bg, main, MAIN_X, MAIN_Y)

        # Result panel: target ~16% width, fixed height from your assets
        RES_W_TARGET = int(0.16 * W)
        # Keep within canvas to the right of main; clamp width
        space_right = W - (MAIN_X + MAIN_W) - MARGIN
        res_w = clamp(RES_W_TARGET, 120, max(120, space_right))
        res_h = 127  # from your asset height; can be scaled if needed
        if res_w < 120:
            res_w = 120
        res = cv.resize(result_image, (res_w, res_h), interpolation=cv.INTER_LINEAR)
        res_x = MAIN_X + MAIN_W + GAP
        res_y = max(int(0.15 * H), MARGIN)
        place_region_safely(bg, res, res_x, res_y)

        # FPS bar: width = main width, height fixed
        fps_w, fps_h = MAIN_W, 30
        fpsbar = cv.resize(fps_log_image, (fps_w, fps_h), interpolation=cv.INTER_LINEAR)
        fps_x = MAIN_X
        fps_y = MAIN_Y + MAIN_H + GAP
        place_region_safely(bg, fpsbar, fps_x, fps_y)

        return bg

    def _process_frame(self, frame_bgr, key=255) -> np.ndarray:
        """
        Run the SLR pipeline on a single frame and return composed canvas.
        """
        fps = self.cv_fps.get()

        debug_image = copy.deepcopy(frame_bgr)
        result_image = get_result_image()
        fps_log_image = get_fps_log_image()

        # Hands detection (RGB for mediapipe)
        img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.hands.process(img_rgb)
        img_rgb.flags.writeable = True

        if self.DEBUG:
            self.MODE = get_mode(key, self.MODE)
            fps_log_image = show_fps_log(fps_log_image, fps)

        if results and results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)

                if self.MODE == 0:
                    hand_sign_id = self.keypoint_classifier(pre_processed)
                    hand_sign_text = "" if hand_sign_id == 25 else self.keypoint_classifier_labels[hand_sign_id]
                    result_image = show_result(result_image, handedness, hand_sign_text)
                elif self.MODE == 1:
                    log_keypoints(key, pre_processed, self.counter_obj, data_limit=1000)

                debug_image = draw_bounding_rect(debug_image, True, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)

        # Compose safely
        return self._compose(debug_image, result_image, fps_log_image)

    def _placeholder_frame(self, message: str) -> np.ndarray:
        """
        Generate a readable placeholder canvas with a message.
        """
        canvas = np.zeros((self.CANVAS_H, self.CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (25, 31, 69)
        cv.putText(canvas, message, (40, 160), cv.FONT_HERSHEY_SIMPLEX, 1.0, (200, 220, 255), 2, cv.LINE_AA)
        cv.putText(canvas, "Check camera connection and restart the server.", (40, 210),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (180, 200, 240), 2, cv.LINE_AA)
        return canvas

    def frames(self):
        """
        Generator that yields JPEG frames for MJPEG.
        Always returns a frame (fallbacks if camera fails).
        """
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                self._last_ok = ok
                if ok:
                    # Resize to capture size and mirror (match original)
                    frame = cv.resize(frame, (self.CAP_WIDTH, self.CAP_HEIGHT), interpolation=cv.INTER_AREA)
                    frame = cv.flip(frame, 1)
                    composed = self._process_frame(frame)
                else:
                    # Camera not available; show placeholder
                    composed = self._placeholder_frame("Camera frame not available.")

                # Ensure output width equals canvas width
                if composed.shape[1] != self.CANVAS_W or composed.shape[0] != self.CANVAS_H:
                    composed = cv.resize(composed, (self.CANVAS_W, self.CANVAS_H), interpolation=cv.INTER_AREA)

                # Encode JPEG (quality can be tuned)
                ok_jpg, jpg = cv.imencode(".jpg", composed, [cv.IMWRITE_JPEG_QUALITY, 80])
                if not ok_jpg:
                    # In the rare case encoding fails, continue with a fresh placeholder
                    composed = self._placeholder_frame("Encoding error; recovering...")
                    ok_jpg, jpg = cv.imencode(".jpg", composed, [cv.IMWRITE_JPEG_QUALITY, 80])
                    if not ok_jpg:
                        time.sleep(0.02)
                        continue

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        except GeneratorExit:
            # Client disconnected
            return
        except Exception as e:
            # On unexpected error, keep streaming a readable frame instead of crashing the server
            msg = f"Runtime error: {type(e).__name__}"
            placeholder = self._placeholder_frame(msg)
            ok_jpg, jpg = cv.imencode(".jpg", placeholder, [cv.IMWRITE_JPEG_QUALITY, 80])
            if ok_jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(0.2)

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            cv.destroyAllWindows()
        except Exception:
            pass


processor = None

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/stream")
def stream():
    def generate():
        for chunk in processor.frames():
            yield chunk
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/health")
def health():
    info = {
        "status": "ok",
        "camera_ok": bool(processor and processor._last_ok),
        "canvas": {"width": getattr(processor, "CANVAS_W", None),
                   "height": getattr(processor, "CANVAS_H", None)}
    }
    return jsonify(info), 200


def main():
    global processor

    # Handle Ctrl+C gracefully on Windows, too
    def handle_sigint(signum, frame):
        if processor:
            processor.stop()
            processor.release()
        # Let Flask/Werkzeug exit
        os._exit(0)

    try:
        signal.signal(signal.SIGINT, handle_sigint)
    except Exception:
        # Not all platforms allow signal setup the same way
        pass

    processor = SLRProcessor()
    try:
        # Flask dev server on localhost
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        if processor:
            processor.stop()
            processor.release()


if __name__ == "__main__":
    main()
