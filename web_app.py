# web_app.py
"""
Flask MJPEG server for SLR with multi-page UI:
- Navbar: Home / ASL / About Us
- ASL page shows live server stream + Start/Stop Recording
- Bounds-safe layout, healthcheck, clean shutdown
"""

import os
import csv
import copy
import time
import signal
import threading
import datetime
import numpy as np

import cv2 as cv
import mediapipe as mp
from dotenv import load_dotenv
from flask import (
    Flask, Response, render_template_string, jsonify, request, redirect, url_for
)

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

# ---------- Shared HTML (Navbar + styles) ----------
BASE_CSS = """
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    background: #0f1222; color: #e6e6e6;
    display: flex; flex-direction: column;
  }
  header {
    background: #131738; position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
  }
  .nav {
    display: flex; gap: 16px; align-items: center; padding: 12px 16px;
    max-width: 1200px; margin: 0 auto;
  }
  .brand { font-weight: 700; }
  .nav a {
    color: #c9d1ff; text-decoration: none; padding: 6px 10px; border-radius: 8px;
  }
  .nav a.active, .nav a:hover { background: #2a346b; }
  main { flex: 1; display: grid; place-items: center; padding: 20px; }
  .frame {
    width: min(96vw, 1600px); aspect-ratio: 16/9;
    background: #191f45; border-radius: 16px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    display: grid; place-items: center; overflow: hidden;
  }
  .frame img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .toolbar {
    max-width: 1600px; width: 96vw; margin: 8px auto 0;
    display: flex; gap: 10px; align-items: center; justify-content: space-between;
  }
  .btn {
    background: #2a346b; color: #e6e6e6; border: none;
    padding: 8px 12px; border-radius: 10px; cursor: pointer;
  }
  .btn:hover { filter: brightness(1.1); }
  .pill { background: #2a346b; padding: 4px 10px; border-radius: 999px; font-size: 12px; letter-spacing: .6px; }
  .content { max-width: 1000px; margin: 0 auto; line-height: 1.6; color: #dce2ff; text-align:center; }
  footer { text-align:center; font-size: 12px; color: #9aa5d1; padding: 8px 0 16px; }
  code { background:#121633; padding:2px 6px; border-radius:6px; }
</style>
"""

NAVBAR = """
<header>
  <nav class="nav">
    <div class="brand">SLR Demo</div>
    <a href="/" class="{{ 'active' if active=='home' else '' }}">Home</a>
    <a href="/asl" class="{{ 'active' if active=='asl' else '' }}">ASL</a>
    <a href="/about" class="{{ 'active' if active=='about' else '' }}">About Us</a>
    <a href="/health" target="_blank">Health</a>
  </nav>
</header>
"""

HOME_HTML = f"""
<!doctype html><html><head><meta charset="utf-8"/><title>Home</title>{BASE_CSS}</head>
<body>
  {NAVBAR}
  <main>
    <div class="content">
      <h2>Welcome</h2>
      <p>This is the landing page for the Sign Language Recognition demo.</p>
      <p><a class="btn" href="/asl">Go to ASL Live</a></p>
    </div>
  </main>
  <footer>© SLR Demo</footer>
</body></html>
"""

ASL_HTML = f"""
<!doctype html><html><head><meta charset="utf-8"/><title>ASL • Live</title>{BASE_CSS}</head>
<body>
  {NAVBAR}
  <main>
    <div>
      <div class="toolbar">
        <div class="pill">LIVE (Server Stream)</div>
        <div style="display:flex; gap:10px;">
          <button id="recBtn" class="btn">Start Recording</button>
          <a class="btn" href="/recordings" target="_blank">Recordings</a>
        </div>
      </div>
      <div class="frame" style="margin-top:10px;">
        <img src="/stream" alt="Sign Language Recognition Stream"/>
      </div>
    </div>
  </main>
  <footer>Server stream uses the camera on the machine running Flask.</footer>
  <script>
    async function syncRecordBtn() {{
      try {{
        const r = await fetch('/record/status'); const s = await r.json();
        document.getElementById('recBtn').textContent =
          s.recording ? 'Stop Recording' : 'Start Recording';
      }} catch {{}}
    }}
    async function toggleRecord() {{
      const btn = document.getElementById('recBtn');
      const stopping = btn.textContent.includes('Stop');
      btn.disabled = true;
      try {{
        const res = await fetch(stopping ? '/record/stop' : '/record/start', {{ method: 'POST' }});
        await res.json(); await syncRecordBtn();
      }} finally {{ btn.disabled = false; }}
    }}
    document.getElementById('recBtn').addEventListener('click', toggleRecord);
    syncRecordBtn();
  </script>
</body></html>
"""

ABOUT_HTML = f"""
<!doctype html><html><head><meta charset="utf-8"/><title>About Us</title>{BASE_CSS}</head>
<body>
  {NAVBAR}
  <main>
    <div class="content" style="text-align:left;">
      <h2>About This Demo</h2>
      <p>This app runs a server-side Sign Language Recognition pipeline using OpenCV + MediaPipe, streams a composed canvas to your browser as MJPEG, and can record MP4s on the server.</p>
      <ul>
        <li>Live ASL page: <code>/asl</code></li>
        <li>Stream endpoint: <code>/stream</code></li>
        <li>Recording controls on the ASL page; files saved under <code>recordings/</code></li>
        <li>Healthcheck: <code>/health</code></li>
      </ul>
    </div>
  </main>
  <footer>© SLR Demo</footer>
</body></html>
"""

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def place_region_safely(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> None:
    """
    Paste fg into bg at (x, y), cropping if needed so we never go out-of-bounds.
    """
    if bg is None or fg is None:
        return
    H, W = bg.shape[:2]
    h, w = fg.shape[:2]
    x1 = clamp(x, 0, W); y1 = clamp(y, 0, H)
    x2 = clamp(x + w, 0, W); y2 = clamp(y + h, 0, H)
    if x2 <= x1 or y2 <= y1: return
    src_x1 = x1 - x; src_y1 = y1 - y
    src_x2 = src_x1 + (x2 - x1); src_y2 = src_y1 + (y2 - y1)
    bg[y1:y2, x1:x2] = fg[src_y1:src_y2, src_x1:src_x2]


class SLRProcessor:
    """Encapsulates camera, MediaPipe pipeline, composition, and recording."""
    def __init__(self):
        print("INFO: Initializing System (web mode)")
        load_dotenv()

        args = get_args()
        self.CAP_DEVICE = 0
        self.CAP_WIDTH = args.width
        self.CAP_HEIGHT = args.height

        self.USE_STATIC_IMAGE_MODE = True
        self.MAX_NUM_HANDS = args.max_num_hands
        self.MIN_DETECTION_CONFIDENCE = args.min_detection_confIDENCE = args.min_detection_confidence
        self.MIN_TRACKING_CONFIDENCE = args.min_tracking_confidence
        self.USE_BRECT = args.use_brect
        self.MODE = args.mode
        self.DEBUG = int(os.environ.get("DEBUG", "0")) == 1

        keypoint_file = "slr/model/keypoint.csv"
        self.counter_obj = get_dict_form_list(keypoint_file)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.USE_STATIC_IMAGE_MODE,
            max_num_hands=self.MAX_NUM_HANDS,
            min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
        )

        self.keypoint_classifier = KeyPointClassifier()
        keypoint_labels_file = "slr/model/label.csv"
        with open(keypoint_labels_file, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in reader]

        self.cv_fps = CvFpsCalc(buffer_len=10)

        self.CANVAS_W = int(os.environ.get("CANVAS_W", "1600"))
        self.CANVAS_H = int(os.environ.get("CANVAS_H", "900"))

        bg = cv.imread("resources/background.png")
        if bg is None:
            self.background_image = np.zeros((self.CANVAS_H, self.CANVAS_W, 3), dtype=np.uint8)
            self.background_image[:] = (25, 31, 69)
        else:
            self.background_image = cv.resize(bg, (self.CANVAS_W, self.CANVAS_H), interpolation=cv.INTER_AREA)

        print("INFO: Opening Camera")
        self.cap = cv.VideoCapture(self.CAP_DEVICE, cv.CAP_DSHOW)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.CAP_WIDTH)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.CAP_HEIGHT)

        # Recording state
        self.recording = False
        self.writer = None
        self.record_dir = os.path.abspath("recordings")
        os.makedirs(self.record_dir, exist_ok=True)
        self.record_path = None

        self._last_ok = False
        self._stop = threading.Event()

        print("INFO: System is up & running")

    def stop(self):
        self._stop.set()

    def _compose(self, debug_image, result_image, fps_log_image) -> np.ndarray:
        bg = self.background_image.copy()
        W, H = bg.shape[1], bg.shape[0]
        MARGIN = int(0.025 * W)
        GAP = int(0.0125 * W)
        MAIN_W = int(0.80 * W)
        MAIN_H = int(MAIN_W * 9 / 16)
        MAIN_X = MARGIN
        MAIN_Y = int(0.08 * H)
        if MAIN_Y + MAIN_H + MARGIN > H:
            MAIN_H = max(100, H - MAIN_Y - MARGIN)
            MAIN_W = int(MAIN_H * 16 / 9)

        main = cv.resize(debug_image, (MAIN_W, MAIN_H), interpolation=cv.INTER_LINEAR)
        place_region_safely(bg, main, MAIN_X, MAIN_Y)

        RES_W_TARGET = int(0.16 * W)
        space_right = W - (MAIN_X + MAIN_W) - MARGIN
        res_w = clamp(RES_W_TARGET, 120, max(120, space_right))
        res_h = 127
        if res_w < 120: res_w = 120
        res = cv.resize(result_image, (res_w, res_h), interpolation=cv.INTER_LINEAR)
        res_x = MAIN_X + MAIN_W + GAP
        res_y = max(int(0.15 * H), MARGIN)
        place_region_safely(bg, res, res_x, res_y)

        fps_w, fps_h = MAIN_W, 30
        fpsbar = cv.resize(fps_log_image, (fps_w, fps_h), interpolation=cv.INTER_LINEAR)
        fps_x = MAIN_X
        fps_y = MAIN_Y + MAIN_H + GAP
        place_region_safely(bg, fpsbar, fps_x, fps_y)
        return bg

    def _process_frame(self, frame_bgr) -> np.ndarray:
        fps = self.cv_fps.get()
        debug_image = copy.deepcopy(frame_bgr)
        result_image = get_result_image()
        fps_log_image = get_fps_log_image()

        img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.hands.process(img_rgb)
        img_rgb.flags.writeable = True

        if self.DEBUG:
            self.MODE = get_mode(255, self.MODE)
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
                    log_keypoints(255, pre_processed, self.counter_obj, data_limit=1000)

                debug_image = draw_bounding_rect(debug_image, True, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)

        return self._compose(debug_image, result_image, fps_log_image)

    def _placeholder_frame(self, message: str) -> np.ndarray:
        canvas = np.zeros((self.CANVAS_H, self.CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (25, 31, 69)
        cv.putText(canvas, message, (40, 160), cv.FONT_HERSHEY_SIMPLEX, 1.0, (200, 220, 255), 2, cv.LINE_AA)
        cv.putText(canvas, "Check camera connection and restart the server.", (40, 210),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (180, 200, 240), 2, cv.LINE_AA)
        return canvas

    def _ensure_writer(self):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out_w, out_h = self.CANVAS_W, self.CANVAS_H
        fps = self.cap.get(cv.CAP_PROP_FPS)
        if not fps or fps <= 0: fps = 20.0
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.record_dir, f"slr_{ts}.mp4")
        self.writer = cv.VideoWriter(path, fourcc, fps, (out_w, out_h))
        self.record_path = path

    def start_record(self):
        if self.recording:
            return True, self.record_path
        self._ensure_writer()
        self.recording = True
        return True, self.record_path

    def stop_record(self):
        if self.writer is not None:
            try: self.writer.release()
            except Exception: pass
        self.writer = None
        self.recording = False
        return True, self.record_path

    def frames(self):
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                self._last_ok = ok
                if ok:
                    frame = cv.resize(frame, (self.CAP_WIDTH, self.CAP_HEIGHT), interpolation=cv.INTER_AREA)
                    frame = cv.flip(frame, 1)
                    composed = self._process_frame(frame)
                else:
                    composed = self._placeholder_frame("Camera frame not available.")

                # Ensure output matches canvas size
                if composed.shape[1] != self.CANVAS_W or composed.shape[0] != self.CANVAS_H:
                    composed = cv.resize(composed, (self.CANVAS_W, self.CANVAS_H), interpolation=cv.INTER_AREA)

                # Write frame if recording
                if self.recording:
                    if self.writer is None:
                        self._ensure_writer()
                    try:
                        self.writer.write(composed)
                    except Exception:
                        self.stop_record()

                ok_jpg, jpg = cv.imencode(".jpg", composed, [cv.IMWRITE_JPEG_QUALITY, 80])
                if not ok_jpg:
                    composed = self._placeholder_frame("Encoding error; recovering...")
                    ok_jpg, jpg = cv.imencode(".jpg", composed, [cv.IMWRITE_JPEG_QUALITY, 80])
                    if not ok_jpg:
                        time.sleep(0.02); continue

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        except GeneratorExit:
            return
        except Exception as e:
            placeholder = self._placeholder_frame(f"Runtime error: {type(e).__name__}")
            ok_jpg, jpg = cv.imencode(".jpg", placeholder, [cv.IMWRITE_JPEG_QUALITY, 80])
            if ok_jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(0.2)

    def release(self):
        try:
            if self.writer is not None: self.writer.release()
        except Exception: pass
        try:
            if self.cap is not None: self.cap.release()
        except Exception: pass
        try:
            cv.destroyAllWindows()
        except Exception: pass


processor = None

# ---------- Routes ----------
@app.route("/")
def home():
    # Use landing page; if you want direct ASL, return redirect(url_for('asl'))
    return render_template_string(HOME_HTML, active="home")

@app.route("/asl")
def asl():
    return render_template_string(ASL_HTML, active="asl")

@app.route("/about")
def about():
    return render_template_string(ABOUT_HTML, active="about")

@app.route("/stream")
def stream():
    def generate():
        for chunk in processor.frames():
            yield chunk
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/record/start", methods=["POST"])
def record_start():
    ok, path = processor.start_record()
    return jsonify({"ok": ok, "recording": processor.recording, "filepath": path})

@app.route("/record/stop", methods=["POST"])
def record_stop():
    ok, path = processor.stop_record()
    return jsonify({"ok": ok, "recording": processor.recording, "filepath": path})

@app.route("/record/status")
def record_status():
    return jsonify({"recording": processor.recording, "filepath": processor.record_path})

@app.route("/recordings")
def list_recordings():
    folder = processor.record_dir
    files = []
    try:
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".mp4"):
                files.append(name)
    except Exception:
        pass

    links = []
    for n in files:
        filepath = os.path.join(folder, n).replace("\\", "/")
        links.append(f"<li><a href='file:///{filepath}' target='_blank'>{n}</a></li>")

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Recordings</title>{BASE_CSS}</head>
    <body>{NAVBAR}
    <main><div class='content' style='text-align:left;'>
      <h2>Recordings</h2>
      <ul>
        {''.join(links) or "<li>No recordings yet</li>"}
      </ul>
      <p>Folder: <code>{folder}</code></p>
    </div></main><footer>Open MP4s with your default player.</footer></body></html>"""
    return html


@app.route("/health")
def health():
    info = {
        "status": "ok",
        "camera_ok": bool(processor and processor._last_ok),
        "recording": bool(processor and processor.recording),
        "canvas": {"width": getattr(processor, "CANVAS_W", None),
                   "height": getattr(processor, "CANVAS_H", None)}
    }
    return jsonify(info), 200

# ---------- Bootstrap ----------
def main():
    global processor

    def handle_sigint(signum, frame):
        if processor:
            processor.stop()
            processor.release()
        os._exit(0)

    try:
        signal.signal(signal.SIGINT, handle_sigint)
    except Exception:
        pass

    processor = SLRProcessor()
    try:
        # Local only; to allow other devices on LAN, set host="0.0.0.0"
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        if processor:
            processor.stop()
            processor.release()

if __name__ == "__main__":
    main()
