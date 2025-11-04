# src/eval_cls_compare.py
import argparse, csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def ds128(dirpath, split, batch=64):
    return tf.keras.utils.image_dataset_from_directory(
        dirpath,
        validation_split=0.30,
        subset=split, seed=42,
        image_size=(128,128),
        batch_size=batch,
        label_mode="binary"
    )

def probs_from_output(p):
    """Return positive-class probability from model predictions."""
    p = np.array(p)
    if p.ndim == 2 and p.shape[1] == 2:     # softmax over 2 classes
        return p[:, 1]
    if p.ndim == 2 and p.shape[1] == 1:     # sigmoid single-unit
        return p[:, 0]
    if p.ndim == 1:                         # already a vector
        return p
    # Fallback: squeeze last dim
    return np.squeeze(p)

def eval_model(model, ds):
    y_true = []
    y_prob = []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true.append(y.numpy().ravel())
        y_prob.append(probs_from_output(p).ravel())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    # probabilities → labels
    y_pred = (y_prob >= 0.5).astype(np.int32)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    # ROC-AUC only if both classes are present
    try:
        if np.unique(y_true).size == 2:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    cm   = confusion_matrix(y_true, y_pred)
    rpt  = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return acc, f1, auc, cm, rpt, (y_true, y_prob)

def plot_cm(cm, outpng, title):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpng, dpi=160); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--modelA", required=True)
    ap.add_argument("--modelB", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Build two fresh (identical) validation datasets so each eval gets a full pass
    ds_val_A = ds128(args.data, "validation", batch=args.batch).prefetch(tf.data.AUTOTUNE)
    ds_val_B = ds128(args.data, "validation", batch=args.batch).prefetch(tf.data.AUTOTUNE)

    A = keras.models.load_model(args.modelA, compile=False)
    B = keras.models.load_model(args.modelB, compile=False)

    a_acc,a_f1,a_auc,a_cm,a_rpt,_ = eval_model(A, ds_val_A)
    b_acc,b_f1,b_auc,b_cm,b_rpt,_ = eval_model(B, ds_val_B)

    # Save CSV
    with open(outdir/"compare.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["model","accuracy","f1","roc_auc"])
        w.writerow(["A_baseline",f"{a_acc:.4f}",f"{a_f1:.4f}",f"{a_auc:.4f}"])
        w.writerow(["B_SR",      f"{b_acc:.4f}",f"{b_f1:.4f}",f"{b_auc:.4f}"])

    # Save CMs + reports
    (outdir/"reports").mkdir(exist_ok=True)
    plot_cm(a_cm, outdir/"cm_A.png", "Confusion Matrix — A")
    plot_cm(b_cm, outdir/"cm_B.png", "Confusion Matrix — B")
    (outdir/"reports"/"A_report.txt").write_text(a_rpt)
    (outdir/"reports"/"B_report.txt").write_text(b_rpt)

    print("[A] acc={:.4f} f1={:.4f} auc={:.4f}".format(a_acc,a_f1,a_auc))
    print("[B] acc={:.4f} f1={:.4f} auc={:.4f}".format(b_acc,b_f1,b_auc))
    print("Wrote:", outdir/"compare.csv", outdir/"cm_A.png", outdir/"cm_B.png")
    print("Reports in:", outdir/"reports")
if __name__ == "__main__":
    main()
