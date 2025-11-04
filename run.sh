#!/usr/bin/env bash
set -e

# 1) Train classifier A (128x128)
python -m src.train_A \
  --data ./data \
  --out  ./artifacts/A \
  --base MobileNetV2 --epochs 10

# 2) Train SRGAN (LR 32 -> HR 128)
python -m src.train_srgan \
  --data    ./data \
  --ckpts   ./artifacts/SRGAN/ckpts \
  --samples ./artifacts/SRGAN/samples \
  --epochs  150 \
  --batch   32

# 3) Quant. SR eval (PSNR / SSIM)
python -m src.eval_sr_metrics \
  --data ./data \
  --gen  ./artifacts/SRGAN/generator_final.keras \
  --out  ./artifacts/SRGAN/eval

# 4) Make SR training GIF (visual progression)
python tools/make_progress_gif.py \
  --samples ./artifacts/SRGAN/samples \
  --out     ./artifacts/SRGAN/sr_progress.gif \
  --fps     3

# 5) Train classifier B on SR images
python -m src.train_B \
  --data  ./data \
  --gen   ./artifacts/SRGAN/generator_final.keras \
  --out   ./artifacts/B \
  --base  MobileNetV2 \
  --epochs 10

# 6) Compare A vs B (Accuracy, F1, AUC)
python -m src.eval_cls_compare \
  --data ./data \
  --modelA ./artifacts/A/modelA_final.keras \
  --modelB ./artifacts/B/modelB_final.keras \
  --out   ./artifacts/CLS
