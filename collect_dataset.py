"""
Hand Gesture Recognition - Smart Dataset Collection Tool
=========================================================
Real-time validator that tells you if lighting and hand
position are good BEFORE saving photos.
"""

import cv2
import os
import time
import numpy as np

SAVE_DIR     = "dataset"
ROI_X, ROI_Y = 100, 100
ROI_W, ROI_H = 300, 300
TARGET_COUNT = 200


def binarize(image):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    return binary


def validate_frame(roi):
    """
    Returns (is_valid, score, list_of_issues)
    Checks lighting, hand presence, contrast, hand size.
    """
    issues  = []
    score   = 100

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary  = binarize(roi)

    # ── Check 1: Brightness (lighting) ──────────────────────────────────
    brightness = gray.mean()
    if brightness < 60:
        issues.append("Too dark — turn on more light")
        score -= 40
    elif brightness > 220:
        issues.append("Too bright — reduce light or move back")
        score -= 30
    elif brightness < 90:
        issues.append("Slightly dark — add more light")
        score -= 15

    # ── Check 2: White pixel ratio (hand presence) ───────────────────────
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio < 0.05:
        issues.append("No hand detected — put hand in the box")
        score -= 40
    elif white_ratio > 0.85:
        issues.append("Too much white — use darker background")
        score -= 35
    elif white_ratio < 0.10:
        issues.append("Hand too small — move closer to camera")
        score -= 20

    # ── Check 3: Hand size (contour area) ───────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_area = max(cv2.contourArea(c) for c in contours)
        frame_area   = ROI_W * ROI_H
        hand_ratio   = largest_area / frame_area
        if hand_ratio < 0.08:
            issues.append("Hand too far — move closer to camera")
            score -= 25
        elif hand_ratio > 0.90:
            issues.append("Hand too close — move slightly back")
            score -= 15
    else:
        issues.append("No hand contour found — adjust position")
        score -= 30

    # ── Check 4: Contrast (background vs hand) ───────────────────────────
    contrast = gray.std()
    if contrast < 30:
        issues.append("Low contrast — use darker background")
        score -= 20

    score   = max(0, score)
    is_valid = score >= 65 and len(issues) == 0

    return is_valid, score, issues


def draw_validator_ui(frame, roi, is_valid, score, issues, count, gesture, mode):
    h, w = frame.shape[:2]

    # ── Binary preview ───────────────────────────────────────────────────
    binary         = binarize(roi)
    binary_color   = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_resized = cv2.resize(binary_color, (ROI_W, ROI_H))

    panel_x = ROI_X + ROI_W + 10
    if panel_x + ROI_W <= w:
        frame[ROI_Y:ROI_Y+ROI_H, panel_x:panel_x+ROI_W] = binary_resized
        cv2.putText(frame, "What model sees",
                    (panel_x, ROI_Y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,255), 1)

    # ── ROI box color based on validity ──────────────────────────────────
    box_color = (0, 255, 0) if is_valid else (0, 0, 255)
    cv2.rectangle(frame, (ROI_X, ROI_Y),
                  (ROI_X+ROI_W, ROI_Y+ROI_H), box_color, 3)

    # ── Score bar ────────────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = ROI_X, ROI_Y + ROI_H + 10, ROI_W, 18
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (60,60,60), -1)
    fill_w  = int(bar_w * score / 100)
    fill_col = (0,200,0) if score>=65 else (0,165,255) if score>=40 else (0,0,220)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_h), fill_col, -1)
    cv2.putText(frame, f"Quality: {score}%",
                (bar_x + 5, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # ── Big status badge ─────────────────────────────────────────────────
    if is_valid:
        badge_col  = (0, 180, 0)
        badge_text = "GOOD  — capturing!"
    elif score >= 40:
        badge_col  = (0, 130, 220)
        badge_text = "ALMOST GOOD — fix issues below"
    else:
        badge_col  = (0, 0, 200)
        badge_text = "NOT VALID — fix issues below"

    cv2.rectangle(frame, (ROI_X, bar_y + 25),
                  (ROI_X + ROI_W, bar_y + 50), badge_col, -1)
    cv2.putText(frame, badge_text,
                (ROI_X + 8, bar_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    # ── Issues list ──────────────────────────────────────────────────────
    if issues:
        cv2.putText(frame, "Fix these:",
                    (ROI_X, bar_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
        for i, issue in enumerate(issues[:3]):
            cv2.putText(frame, f"  {i+1}. {issue}",
                        (ROI_X, bar_y + 92 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80,200,255), 1)
    else:
        cv2.putText(frame, "All checks passed!",
                    (ROI_X, bar_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,230,0), 1)

    # ── Top info bar ─────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 35), (30,30,30), -1)
    mode_txt = "AUTO" if mode else "MANUAL"
    cv2.putText(frame,
                f"Gesture: {gesture}  |  Saved: {count}/{TARGET_COUNT}  |  Mode: {mode_txt}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

    # ── Bottom controls bar ───────────────────────────────────────────────
    cv2.rectangle(frame, (0, h-28), (w, h), (30,30,30), -1)
    cv2.putText(frame,
                "A=auto capture  |  SPACE=manual  |  N=next gesture  |  Q=quit",
                (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    return frame


def collect(gesture_name=None):
    os.makedirs(SAVE_DIR, exist_ok=True)

    if gesture_name is None:
        print("\n" + "="*50)
        print("  Gestures to collect:")
        print("  -> PEACE      (index + middle up)")
        print("  -> LOSER      (index + pinky + thumb)")
        print("  -> HANG_LOOSE (thumb + pinky)")
        print("  -> POWER      (full fist)")
        print("  -> SUPER      (thumb up)")
        print("="*50)
        gesture_name = input("\nEnter gesture name: ").strip().upper()

    if not gesture_name:
        return

    save_path = os.path.join(SAVE_DIR, gesture_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\nERROR: Cannot open camera.")
        print("System Settings -> Privacy & Security -> Camera -> allow Terminal/VS Code")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count        = len([f for f in os.listdir(save_path)
                        if f.lower().endswith(".jpg")])
    auto_capture = False
    last_auto    = time.time()
    saved_flash  = 0

    print(f"\n  Ready to collect '{gesture_name}' | Existing: {count}")
    print("  Watch the QUALITY BAR and STATUS BADGE on screen!")
    print("  Only collect when badge shows GREEN 'GOOD'\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        roi   = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W].copy()

        # Validate this frame
        is_valid, score, issues = validate_frame(roi)

        # Draw full UI
        frame = draw_validator_ui(frame, roi, is_valid, score,
                                  issues, count, gesture_name, auto_capture)

        # Flash effect when saved
        if saved_flash > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (ROI_X, ROI_Y),
                          (ROI_X+ROI_W, ROI_Y+ROI_H), (0,255,0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, f"SAVED! {count}",
                        (ROI_X+60, ROI_Y+160),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2)
            saved_flash -= 1

        cv2.imshow("Smart Dataset Collection", frame)

        # Auto capture — only saves VALID frames
        if auto_capture and is_valid and (time.time() - last_auto) >= 0.2:
            fname = os.path.join(save_path, f"{gesture_name}_{count:04d}.jpg")
            cv2.imwrite(fname, roi)
            count    += 1
            last_auto = time.time()
            saved_flash = 3
            print(f"  Auto-saved valid frame: {count}", end="\r")

            if count >= TARGET_COUNT:
                print(f"\n  Target {TARGET_COUNT} reached for '{gesture_name}'!")
                auto_capture = False

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if is_valid:
                fname = os.path.join(save_path, f"{gesture_name}_{count:04d}.jpg")
                cv2.imwrite(fname, roi)
                count       += 1
                saved_flash  = 3
                print(f"  Saved: {fname} ({count})")
            else:
                print(f"  NOT saved — quality too low ({score}%). Fix: {issues[0] if issues else 'unknown'}")

        elif key in (ord('a'), ord('A')):
            auto_capture = not auto_capture
            if auto_capture:
                print("\n  Auto-capture ON — will only save VALID frames automatically!")
            else:
                print("\n  Auto-capture OFF")

        elif key in (ord('n'), ord('N')):
            print(f"\n  Done '{gesture_name}' — {count} images saved.")
            cap.release()
            cv2.destroyAllWindows()
            collect()
            return

        elif key in (ord('q'), ord('Q')):
            print(f"\n  Finished. {count} images saved for '{gesture_name}'.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect()