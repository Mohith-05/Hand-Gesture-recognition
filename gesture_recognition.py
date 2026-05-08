"""
Hand Gesture Recognition - Real-Time Inference
===============================================
Project : Hand Gesture Recognition for Hearing Impaired and Aphonic People
Method  : Binarization → Contour Detection → SIFT → CNN Classification

Team    : Mohith V, Vibish M, Sree Kumaran — Second Year CSE-C

Mac Note: cv2.imshow() requires the Terminal / VS Code process to have
          Camera + Screen Recording permissions in System Settings.

Usage:
    python gesture_recognition.py

Controls:
    Q  – Quit
    B  – Toggle binary/contour debug panel
    S  – Save snapshot
"""

import cv2
import numpy as np
import tensorflow as tf
import os, time

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH           = "gesture_cnn_model.h5"
CLASSES_PATH         = "gesture_classes.npy"
IMG_SIZE             = 64
CONFIDENCE_THRESHOLD = 0.30

ROI_X, ROI_Y = 100, 100
ROI_W, ROI_H = 300, 300


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING  (identical to training pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def binarize_image(image):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return binary


def detect_and_crop_contour(binary, original):
    contours, _ = cv2.findContours(binary,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if not contours:
        return original, debug

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(debug, contours, -1, (0, 255, 0), 1)
    cv2.drawContours(debug, [largest], -1, (255, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(largest)
    pad = 10
    x = max(0, x - pad);  y = max(0, y - pad)
    w = min(original.shape[1] - x, w + 2*pad)
    h = min(original.shape[0] - y, h + 2*pad)

    cv2.rectangle(debug, (x,y), (x+w, y+h), (0,0,255), 2)
    return original[y:y+h, x:x+w], debug


def preprocess_for_cnn(image):
    binary        = binarize_image(image)
    cropped, debug = detect_and_crop_contour(binary, image)
    gray          = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized       = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized    = (resized / 255.0).astype(np.float32)
    return normalized[np.newaxis, ..., np.newaxis], debug


# ─────────────────────────────────────────────────────────────────────────────
#  UI DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=10):
    x1,y1 = pt1;  x2,y2 = pt2
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
    for cx,cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),
                  (x1+radius,y2-radius),(x2-radius,y2-radius)]:
        cv2.circle(img, (cx,cy), radius, color, -1)


def draw_ui(frame, label, score, fps, debug_mode):
    h, w = frame.shape[:2]

    # ROI box
    cv2.rectangle(frame, (ROI_X,ROI_Y),
                  (ROI_X+ROI_W, ROI_Y+ROI_H), (65,105,225), 2)
    cv2.putText(frame, "Place Hand Here",
                (ROI_X, ROI_Y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (65,105,225), 1)

    # Gesture label
    if label and score >= CONFIDENCE_THRESHOLD:
        text     = label.replace("_", " ")
        fs, th   = 1.6, 3
        (tw,tht),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        bx1,by1  = 10, h-130
        bx2,by2  = 10+tw+20, h-130+tht+20
        draw_rounded_rect(frame, (bx1,by1), (bx2,by2), (20,20,20))
        cv2.putText(frame, text, (bx1+10, by2-10),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (255,255,255), th)
        cv2.putText(frame, f"(score = {score:.5f})",
                    (bx1+10, by2+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    elif label:
        cv2.putText(frame, "Low confidence...",
                    (10, h-90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (100,100,255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,100), 2)

    # Debug badge
    if debug_mode:
        cv2.putText(frame, "[DEBUG MODE]", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

    # Controls hint
    cv2.putText(frame, "Q=Quit  B=Debug  S=Snapshot",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150,150,150), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RECOGNITION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_recognition():
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: '{MODEL_PATH}' not found.")
        print("Run  python train_model.py  first.\n")
        return

    print("[*] Loading model ...")
    model   = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    print(f"[*] Gestures: {classes.tolist()}\n")

    # Mac built-in FaceTime camera = index 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        print("System Settings → Privacy & Security → Camera → allow Terminal/VS Code")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs("snapshots", exist_ok=True)
    debug_mode   = False
    snap_counter = 0
    prev_time    = time.time()

    print("[*] Recognition running — Q=Quit | B=Debug | S=Snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame read failed.")
            break

        frame = cv2.flip(frame, 1)

        roi_frame               = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W].copy()
        input_arr, debug_frame  = preprocess_for_cnn(roi_frame)
        predictions             = model.predict(input_arr, verbose=0)[0]

        best_idx   = int(np.argmax(predictions))
        best_score = float(predictions[best_idx])
        best_label = classes[best_idx] if best_score >= CONFIDENCE_THRESHOLD else None

        curr_time = time.time()
        fps       = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        display = draw_ui(frame.copy(), best_label, best_score, fps, debug_mode)

        if debug_mode:
            debug_resized = cv2.resize(debug_frame, (ROI_W, ROI_H))
            h = display.shape[0]
            # Paste debug panel to the right of ROI
            rx2 = ROI_X + ROI_W
            if rx2 + ROI_W <= display.shape[1]:
                display[ROI_Y:ROI_Y+ROI_H, rx2:rx2+ROI_W] = debug_resized

        cv2.imshow("Hand Gesture Recognition", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            print("[*] Quitting ...")
            break
        elif key in (ord('b'), ord('B')):
            debug_mode = not debug_mode
            print(f"[*] Debug: {'ON' if debug_mode else 'OFF'}")
        elif key in (ord('s'), ord('S')):
            snap = f"snapshots/snapshot_{snap_counter:04d}.jpg"
            cv2.imwrite(snap, display)
            print(f"[*] Snapshot saved: {snap}")
            snap_counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    run_recognition()
