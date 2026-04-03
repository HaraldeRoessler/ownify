#!/bin/bash
# Run this once on the vast.ai machine to install dependencies
pip install unsloth trl datasets transformers accelerate -q
echo "Setup done. Now run: python train_vastai.py"
