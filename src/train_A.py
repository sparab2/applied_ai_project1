# src/train_A.py
import os
# Make TF quieter + disable XLA JIT (prevents UnsortedSegmentSum bug)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from src.datasets import load_classifier_ds
from src.models_cls import build_classifier

class SaveEveryN(keras.callbacks.Callback):
    def __init__(self, outdir, prefix, n=5):
        super().__init__()
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.n = n; self.prefix = prefix
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.n == 0:
            self.model.save(self.outdir / f"{self.prefix}_epoch{epoch+1}.keras")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--base", default="MobileNetV2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    # Optional mixed precision (works fine for MobileNetV2 on 4090)
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

    # Best-effort allow-growth (ignore if already initialized)
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("[warn] memory_growth skipped:", e)

    # Also explicitly disable JIT at runtime
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

    train_cls, val_cls, classes = load_classifier_ds(args.data, batch=args.batch)

    # Build 2-class classifier (softmax); train with accuracy only.
    model = build_classifier(len(classes), input_shape=(128,128,3), base=args.base)
    # recompile to ensure simple metrics and no jit
    model.compile(optimizer=model.optimizer,
                  loss=model.loss,
                  metrics=["accuracy"],
                  jit_compile=False)

    model.fit(
        train_cls,
        validation_data=val_cls,
        epochs=args.epochs,
        callbacks=[SaveEveryN(args.out, "modelA", n=5)],
    )
    model.save(Path(args.out) / "modelA_final.keras")
    print("Saved:", Path(args.out) / "modelA_final.keras")

if __name__ == "__main__":
    main()
