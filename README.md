# Applied AI Midterm — SRGAN + Transfer Learning

Applied_AI_midterm/
│
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   ├── models_cls.py
│   ├── models_srgan.py
│   ├── train_A.py
│   ├── train_B.py
│   ├── eval_sr_metrics.py
│   ├── eval_cls_compare.py
│
├── artifacts/
│   ├── CLS/
│   │   ├── cm_A.png
│   │   ├── cm_B.png
│   │   ├── compare.csv
│   │   └── reports/
│   │       ├── A_report.txt
│   │       └── B_report.txt
│   ├── SRGAN/
│   │   ├── sr_progress.gif
│   │   ├── preview_augmented.png
│   │   └── eval/
│   │       └── sr_metrics.csv
│
├── data/                # (can be empty or contain a readme)
│
├── README.md            # your documentation (see below)
├── .gitignore
└── requirements.txt     # (optional but good practice)


## Dataset split
- 70% train / 30% test handled with `image_dataset_from_directory` seeds=42.

## Steps (as required)
1. **Train binary classifier A (128×128, transfer learning)**  
   `python -m src.train_A --data ./data --out ./artifacts/A --base MobileNetV2 --epochs 10`

2. **Train SRGAN to generate 128×128 from 32×32 LR**  
   `python -m src.train_srgan --data ./data --ckpts ./artifacts/SRGAN/ckpts --samples ./artifacts/SRGAN/samples --epochs 150 --batch 32`

3. **Show scaled examples in notebook**  
   See `notebooks/01_data_preview.ipynb` (LR/HR previews & augmentations).

4. **Use SRGAN outputs to train classifier B**  
   `python -m src.train_B --data ./data --gen ./artifacts/SRGAN/generator_final.keras --out ./artifacts/B --base MobileNetV2 --epochs 10`

5. **Train SRGAN for ≥150 epochs** ✔️ done.

6. **Normalization & transforms**  
   Implemented in `src/datasets.py` with `rescale=[-1,1]`, LR (32×32) ↔ HR (128×128), and data aug demos in notebook.

7. **Compare A vs B (F1, Accuracy, AUC)**  
   `python -m src.eval_cls_compare --data ./data --modelA ./artifacts/A/modelA_final.keras --modelB ./artifacts/B/modelB_final.keras --out ./artifacts/CLS`

8. **Checkpointing**  
   SRGAN saves every *n* epochs via callback; A/B save final models. (We do not push large files to Git; see `artifacts/README.md`.)

## Reproduce end-to-end
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-min.txt
bash run.sh
