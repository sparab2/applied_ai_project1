# src/train_B.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# XLA can slow small dynamic graphs; disabling it here
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from src.datasets import make_sr_classifier_ds
from src.models_cls import build_classifier
from src.models_srgan import InstanceNorm, PixelShuffle

AUTOTUNE = tf.data.AUTOTUNE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gen",  required=True, help="path to trained generator .keras")
    ap.add_argument("--out",  required=True)
    ap.add_argument("--base", default="MobileNetV2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    # Mixed precision can help on 4090
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

    # GPU mem growth
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("[warn] memory_growth skipped:", e)

    # ---- Load generator ON GPU and warm it up (keeps it placed on GPU) ----
    with tf.device("/GPU:0"):
        G = keras.models.load_model(
            args.gen, compile=False,
            custom_objects={"InstanceNorm": InstanceNorm, "PixelShuffle": PixelShuffle},
        )
        G.trainable = False
        # warmup: alloc kernels & persistent buffers on GPU
        _ = G(tf.zeros([1, 32, 32, 3], dtype=tf.float16))

    # Build LR→SR classification datasets (LR in [-1,1], G returns SR in [-1,1])
    train_B, val_B = make_sr_classifier_ds(args.data, G, batch=args.batch)

    # Make sure the SR step runs on GPU and the pipeline is fast
    @tf.function(jit_compile=False)
    def to_float32(x, y):
        # classifier expects float32; keep cast here (cheap)
        return tf.cast(x, tf.float32), y

    # Turn on TF data optimizations
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    opts.experimental_optimization.apply_default_optimizations = True
    # opts.experimental_optimization.map_parallelization = True
    # opts.experimental_optimization.map_vectorization = True
    eo = opts.experimental_optimization
    if hasattr(eo, "map_parallelization"):
        eo.map_parallelization = True
    if hasattr(eo, "map_vectorization"):
        eo.map_vectorization = True

    train_B = (
        train_B
        .map(to_float32, num_parallel_calls=AUTOTUNE)
        .cache()  # speeds up reuse between epochs
        .with_options(opts)
        .prefetch(AUTOTUNE)  # prefetch must be last
    )

    val_B = (
        val_B
        .map(to_float32, num_parallel_calls=AUTOTUNE)
        .cache()
        .with_options(opts)
        .prefetch(AUTOTUNE)
    )

    # class names for classifier head size
    classes = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.30, subset="training",
        seed=42, image_size=(128, 128), batch_size=1
    ).class_names

    model = build_classifier(len(classes), input_shape=(128, 128, 3), base=args.base)

    class SaveEveryN(keras.callbacks.Callback):
        def __init__(self, outdir, n=5):
            super().__init__()
            self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
            self.n = n
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.n == 0:
                self.model.save(self.outdir / f"modelB_epoch{epoch+1}.keras")

    model.fit(
        train_B,
        validation_data=val_B,
        epochs=args.epochs,
        callbacks=[SaveEveryN(args.out, 5)]
    )
    model.save(Path(args.out) / "modelB_final.keras")
    print("Saved:", Path(args.out) / "modelB_final.keras")

if __name__ == "__main__":
    main()
