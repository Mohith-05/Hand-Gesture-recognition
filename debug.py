import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH   = "gesture_cnn_model.h5"
CLASSES_PATH = "gesture_classes.npy"
IMG_SIZE     = 64

model   = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ROI_X, ROI_Y, ROI_W, ROI_H = 100, 100, 300, 300

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    roi   = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

    gray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized   = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = (resized / 255.0).astype(np.float32)
    inp       = normalized[np.newaxis, ..., np.newaxis]

    preds = model.predict(inp, verbose=0)[0]

    # Print ALL gesture scores
    print("\n--- Scores ---")
    for cls, score in zip(classes, preds):
        bar = "█" * int(score * 30)
        print(f"  {cls:<15} {score:.4f}  {bar}")

    cv2.rectangle(frame, (ROI_X, ROI_Y),
                  (ROI_X+ROI_W, ROI_Y+ROI_H), (0,255,0), 2)
    cv2.imshow("Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()