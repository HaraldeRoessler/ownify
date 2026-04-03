#!/bin/bash
# ownify — vast.ai auto-start script
# Paste this into "On-start script" when renting a new instance.
# Training runs automatically. Download adapters/openclaw-mlx-v3.zip when done.

set -e
LOG=/workspace/training.log
exec > >(tee -a $LOG) 2>&1

echo "=== ownify training started at $(date) ==="

cd /workspace

echo "--- Installing dependencies ---"
pip install unsloth trl datasets transformers accelerate -q

echo "--- Cloning repo ---"
git clone https://github.com/HaraldeRoessler/ownify.git
cd ownify

echo "--- Starting training ---"
python train_vastai.py

echo "--- Zipping adapter ---"
zip -r /workspace/openclaw-mlx-v3.zip adapters/openclaw-mlx-v3/

echo "=== Training complete at $(date) ==="
echo "Download /workspace/openclaw-mlx-v3.zip via Jupyter file browser"
