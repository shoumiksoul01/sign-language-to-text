import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe with new API
model_path = None
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download the hand landmarker model
import urllib.request
MODEL_PATH = r"D:\sign_language_project\hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded.")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3
)

DATASET_PATH = r"D:\sign_language_project\data\asl_alphabet_train"
OUTPUT_CSV   = r"D:\sign_language_project\data\landmarks.csv"

LABELS = [chr(i) for i in range(ord('A'), ord('Z')+1)]

all_rows = []
skipped  = 0
total    = 0

with HandLandmarker.create_from_options(options) as landmarker:
    for label in LABELS:
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder):
            print(f"[SKIP] Folder not found: {folder}")
            continue

        files = os.listdir(folder)
        print(f"Processing {label} — {len(files)} images...")

        for filename in files:
            filepath = os.path.join(folder, filename)
            image_bgr = cv2.imread(filepath)
            if image_bgr is None:
                skipped += 1
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)
            total += 1

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                row = []
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(label)
                all_rows.append(row)
            else:
                skipped += 1

print(f"\nDone! Collected {len(all_rows)} samples. Skipped {skipped}/{total} images.")

cols = []
for i in range(21):
    cols += [f"x{i}", f"y{i}", f"z{i}"]
cols.append("label")

df = pd.DataFrame(all_rows, columns=cols)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")