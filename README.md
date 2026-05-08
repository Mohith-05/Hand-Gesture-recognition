# Hand Gesture Recognition — Mac Setup Guide
**For Hearing Impaired and Aphonic People using CNN**

 Team: Mohith V, Vibish M, Sree Kumaran | Second Year CSE-F

---

## Quick Start (Mac)

### Step 1 — Open project in VS Code
```
File → Open Folder → select HandGestureRecognition
```

### Step 2 — Run setup in VS Code Terminal
```bash
chmod +x setup.sh
./setup.sh
```
This creates a virtual environment and installs all packages including
`tensorflow-macos` and `tensorflow-metal` for Apple Silicon GPU support.

### Step 3 — Activate venv (every new terminal session)
```bash
source venv/bin/activate
```

### Step 4 — Grant Camera Permission
Go to: **System Settings → Privacy & Security → Camera**
Enable access for **Terminal** and/or **Visual Studio Code**

---

## Running the Project (in order)

### 1. Collect Dataset
```bash
python collect_dataset.py
```
- Type a gesture name: `PEACE`, `GOOD_LUCK`, `LOSER`, `HANG_LOOSE`, `POWER`, `NOTHING`
- Put your hand in the **green box** on screen
- Press `A` for auto-capture (recommended — 150 images per gesture)
- Press `N` to move to next gesture

### 2. Train the CNN Model
```bash
python train_model.py
```
Produces `gesture_cnn_model.h5` and `gesture_classes.npy`

### 3. Real-Time Recognition
```bash
python gesture_recognition.py
```

| Key | Action |
|-----|--------|
| Q | Quit |
| B | Toggle debug view (binary + contours) |
| S | Save snapshot to `snapshots/` folder |

---

## Project Pipeline (from Research Paper)

```
Webcam Frame
     │
     ▼
Binarization (Otsu Thresholding)
     │
     ▼
Contour Detection (findContours)
     │
     ▼
SIFT Feature Extraction
     │
     ▼
CNN Classification
  ├─ Convolutional Layers
  ├─ Pooling Layers
  └─ Dense Layers
     │
     ▼
Text Output (Gesture Label + Score)
```

---

## Folder Structure
```
HandGestureRecognition/
├── .vscode/
│   ├── launch.json       ← F5 run configs
│   └── settings.json     ← Mac Python interpreter path
├── dataset/              ← Your gesture images go here
├── snapshots/            ← Recognition screenshots saved here
├── collect_dataset.py    ← Step 1
├── train_model.py        ← Step 2
├── gesture_recognition.py← Step 3
├── requirements.txt
├── setup.sh              ← One-time Mac setup
└── README.md
```

---

## Troubleshooting (Mac)

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: cv2` | Run `source venv/bin/activate` first |
| Camera won't open | System Settings → Privacy → Camera → allow Terminal/VS Code |
| `tensorflow` install fails | Use `pip install tensorflow-macos tensorflow-metal` |
| Window doesn't appear | Make sure you're running from Terminal, not a background task |
# Hand-Gesture-recognition
