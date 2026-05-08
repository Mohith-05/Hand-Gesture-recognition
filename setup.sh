#!/bin/bash
echo "============================================"
echo "  Hand Gesture Recognition - Mac Setup"
echo "============================================"
echo ""

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found."
    echo "Install it from https://www.python.org/downloads/"
    exit 1
fi

echo "[1] Creating virtual environment..."
python3 -m venv venv

echo "[2] Activating virtual environment..."
source venv/bin/activate

echo "[3] Upgrading pip..."
pip install --upgrade pip

echo "[4] Installing dependencies (Mac / Apple Silicon)..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  Setup Complete!"
echo ""
echo "  NEXT STEPS in VS Code Terminal:"
echo "  source venv/bin/activate"
echo ""
echo "  Then run in order:"
echo "  python collect_dataset.py   # Step 1"
echo "  python train_model.py       # Step 2"
echo "  python gesture_recognition.py  # Step 3"
echo "============================================"
