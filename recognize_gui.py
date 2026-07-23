import cv2
import numpy as np
import pickle
import time
import mediapipe as mp

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH      = r"D:\sign_language_project\model\sign_model.pkl"
LANDMARKER_PATH = r"D:\sign_language_project\hand_landmarker.task"

with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
model   = data['model']
classes = data['classes']

# ── MediaPipe setup (new Tasks API) ─────────────────────────────────────────
BaseOptions          = mp.tasks.BaseOptions
HandLandmarker       = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions= mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

# ── Colors ───────────────────────────────────────────────────────────────────
BG_COLOR         = (15, 15, 25)
PANEL_COLOR      = (25, 25, 40)
ACCENT           = (0, 200, 255)
GREEN            = (0, 220, 100)
WHITE            = (255, 255, 255)
GRAY             = (150, 150, 160)
RED              = (60, 60, 200)

# ── Stability filter settings ────────────────────────────────────────────────
STABILITY_TIME   = 1.2   # seconds to hold a sign before confirming
MIN_CONFIDENCE   = 0.60  # minimum confidence to consider a prediction

# ── State ────────────────────────────────────────────────────────────────────
sentence         = ""
current_letter   = ""
confidence_val   = 0.0
stable_letter    = ""
stable_start     = None
progress         = 0.0

def draw_landmarks(frame, hand_landmarks, w, h):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (80, 80, 120), 1)
    for i, (x, y) in enumerate(pts):
        color = ACCENT if i == 0 else (200, 80, 80)
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.circle(frame, (x, y), 5, WHITE, 1)

def draw_confidence_bar(frame, x, y, w, h, value, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), PANEL_COLOR, -1)
    cv2.rectangle(frame, (x, y), (x + int(w * value), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)

# ── Main loop ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Sign Language Recognition...")
print("Controls: SPACE = space | BACKSPACE = delete | ESC = quit")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]

        # ── Run MediaPipe ────────────────────────────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result  = landmarker.detect(mp_img)

        detected = False
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks, w, h)

            row = []
            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z])

            features   = np.array(row).reshape(1, -1)
            probs      = model.predict_proba(features)[0]
            pred_idx   = np.argmax(probs)
            pred_label = classes[pred_idx]
            pred_conf  = probs[pred_idx]

            current_letter = pred_label
            confidence_val = pred_conf
            detected       = True

            # ── Stability filter ─────────────────────────────────────────────
            if pred_conf >= MIN_CONFIDENCE:
                if pred_label == stable_letter:
                    elapsed  = time.time() - stable_start
                    progress = min(elapsed / STABILITY_TIME, 1.0)
                    if elapsed >= STABILITY_TIME:
                        sentence    += pred_label
                        stable_letter = ""
                        stable_start  = None
                        progress      = 0.0
                else:
                    stable_letter = pred_label
                    stable_start  = time.time()
                    progress      = 0.0
            else:
                stable_letter = ""
                stable_start  = None
                progress      = 0.0
        else:
            current_letter = ""
            confidence_val = 0.0
            stable_letter  = ""
            stable_start   = None
            progress       = 0.0

        # ── Draw right panel background ──────────────────────────────────────
        panel_x = w
        full_w  = w + 280
        canvas  = np.zeros((h, full_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = PANEL_COLOR

        # ── Panel: title ─────────────────────────────────────────────────────
        cv2.putText(canvas, "SIGN LANGUAGE", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1)
        cv2.putText(canvas, "TO TEXT", (w + 10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1)
        cv2.line(canvas, (w + 10, 60), (full_w - 10, 60), ACCENT, 1)

        # ── Panel: current letter ─────────────────────────────────────────────
        cv2.putText(canvas, "CURRENT SIGN", (w + 10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
        letter_display = current_letter if detected else "-"
        cv2.putText(canvas, letter_display, (w + 90, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, ACCENT, 5)

        # ── Panel: confidence bar ─────────────────────────────────────────────
        cv2.putText(canvas, "CONFIDENCE", (w + 10, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
        draw_confidence_bar(canvas, w + 10, 192, 255, 14, confidence_val, GREEN)
        cv2.putText(canvas, f"{confidence_val*100:.0f}%", (w + 220, 204),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)

        # ── Panel: stability progress bar ─────────────────────────────────────
        cv2.putText(canvas, "HOLD PROGRESS", (w + 10, 228),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
        draw_confidence_bar(canvas, w + 10, 235, 255, 14, progress,
                            ACCENT if progress < 1.0 else GREEN)

        # ── Panel: sentence ───────────────────────────────────────────────────
        cv2.line(canvas, (w + 10, 265), (full_w - 10, 265), GRAY, 1)
        cv2.putText(canvas, "SENTENCE", (w + 10, 285),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

        display_sentence = sentence[-18:] if len(sentence) > 18 else sentence
        cv2.putText(canvas, display_sentence, (w + 10, 315),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)

        # ── Panel: controls ───────────────────────────────────────────────────
        cv2.line(canvas, (w + 10, 340), (full_w - 10, 340), GRAY, 1)
        cv2.putText(canvas, "SPACE  = add space", (w + 10, 362),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)
        cv2.putText(canvas, "BKSP   = delete", (w + 10, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)
        cv2.putText(canvas, "ESC    = quit", (w + 10, 398),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)

        # ── Camera active indicator ───────────────────────────────────────────
        cv2.circle(canvas, (full_w - 20, 15), 6, GREEN if detected else RED, -1)
        cv2.putText(canvas, "CAM", (full_w - 55, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)

        cv2.imshow("Sign Language to Text", canvas)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:       # ESC
            break
        elif key == 32:     # SPACE
            sentence += " "
        elif key == 8:      # BACKSPACE
            sentence = sentence[:-1]

cap.release()
cv2.destroyAllWindows()
print("Closed.")