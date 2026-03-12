#!/bin/bash
# RunPod setup + training script for Hand Magic Transformer
# Run this on the RunPod instance after uploading the project

set -e

echo "=== Installing dependencies ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib tqdm fastapi uvicorn python-dotenv

echo "=== Starting training ==="
python train_transformer.py \
  --epochs 300 \
  --batch_size 128 \
  --max_stroke_len 1000 \
  --checkpoint_dir checkpoints/transformer/ \
  --deepwriting_path deepwriting_dataset/ \
  --tqdm
