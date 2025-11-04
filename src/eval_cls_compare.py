# src/eval_cls_compare.py
import argparse, numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from src.datasets import load_classifier_ds, make_sr_classifier_ds

def eval_model(model, ds, name, class_names):
    y_true, y_prob = [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0)
        y_prob.append(p); y_true.append(yb.numpy())
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    if len(class_names)==2:
        print("AUC:", roc_auc_score(y_true, y_prob[:,1]))
    else:
        print("AUC (ovr):", roc_auc_score(y_true, y_prob, multi_class='ovr'))
    print("F1 (macro):", f1_score(y_true, y_pred, average="macro"))
    print("Confusion:\n", confusion_matrix(y_true, y_pred))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--A",    required=True)
    ap.add_argument("--B",    required=True)
    ap.add_argument("--gen",  required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    train_cls, val_cls, classes = load_classifier_ds(args.data, batch=args.batch)
    model_A = keras.models.load_model(args.A, compile=False)
    model_B = keras.models.load_model(args.B, compile=False)
    G       = keras.models.load_model(args.gen, compile=False)
    _, val_B = make_sr_classifier_ds(args.data, G, batch=args.batch)

    eval_model(model_A, val_cls, "Model A on real HR", classes)
    eval_model(model_B, val_B,   "Model B on SR",      classes)
    # Optional domain-gap check:
    eval_model(model_B, val_cls, "Model B on real HR", classes)

if __name__ == "__main__":
    main()
