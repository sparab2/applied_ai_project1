import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"   # avoid unexpected XLA fusions with mixed precision

import argparse
import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.datasets import load_sr_pairs, IMG_HR
from src.models_srgan import (
    build_generator, build_discriminator, build_vgg_feature_extractor, SRGAN
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to classless image root used for SR pairs")
    ap.add_argument("--ckpts", required=True, help="Directory to write checkpoints")
    ap.add_argument("--samples", required=True, help="Directory to write sample SR images")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    # --- dirs ---
    ckpt_dir = Path(args.ckpts); ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples); samples_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = ckpt_dir.parent / "logs" / f"srgan_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- mixed precision policy (safe on recent TF) ---
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

    # --- GPU memory growth ---
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("[warn] memory_growth skipped:", e)

    # --- data ---
    train_sr, val_sr = load_sr_pairs(args.data, batch=args.batch)

    # --- models ---
    G = build_generator()
    D = build_discriminator()
    V = build_vgg_feature_extractor()

    gan = SRGAN(G, D, V, lam_content=1.0, lam_adv=1e-3)
    gan.compile(
        g_opt=keras.optimizers.Adam(2e-4, beta_1=0.9, beta_2=0.999),
        d_opt=keras.optimizers.Adam(2e-4, beta_1=0.9, beta_2=0.999),
    )

    # --- checkpoints ---
    ckpt = tf.train.Checkpoint(g=G, d=D, g_opt=gan.g_opt, d_opt=gan.d_opt)
    manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=5)

    # --- tensorboard ---
    tb = keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        write_graph=False,
        histogram_freq=0,
        profile_batch=0,
    )

    class SaveGAN(keras.callbacks.Callback):
        """Save checkpoints every n epochs and log an LR|SR|HR grid to disk + TensorBoard."""
        def __init__(self, n=5, val_ds=val_sr):
            super().__init__()
            self.n = n
            self.fixed = next(iter(val_ds.take(1)))  # one held-out batch
            self.writer = tf.summary.create_file_writer(str(log_dir))

        def on_epoch_end(self, epoch, logs=None):
            # periodic checkpoint
            if (epoch + 1) % self.n == 0:
                manager.save()

            lr, hr = self.fixed

            # forward pass (may be float16 under mixed precision)
            sr = self.model.G(lr, training=False)

            # cast everything to float32 for visualization/summaries
            lr32 = tf.cast(lr, tf.float32)
            sr32 = tf.cast(sr, tf.float32)
            hr32 = tf.cast(hr, tf.float32)

            # back to [0,1]
            lr01 = (lr32 + 1.0) * 0.5
            sr01 = (sr32 + 1.0) * 0.5
            hr01 = (hr32 + 1.0) * 0.5

            # side-by-side grid: LR↑ | SR | HR
            lr_up = tf.image.resize(lr01, (IMG_HR, IMG_HR), method="nearest")
            grid = tf.concat([lr_up, sr01, hr01], axis=2)
            img = tf.concat(tf.unstack(grid[:4], axis=0), axis=0)
            img = tf.clip_by_value(img, 0.0, 1.0)

            # save PNG
            out_path = samples_dir / f"epoch_{epoch + 1:04d}.png"
            keras.utils.save_img(str(out_path), img)

            # write TB scalars + image
            with self.writer.as_default():
                tf.summary.image("LR|SR|HR_grid", tf.expand_dims(img, 0), step=epoch + 1)
                if logs:
                    for k, v in logs.items():
                        try:
                            tf.summary.scalar(k, float(v), step=epoch + 1)
                        except Exception:
                            pass
            self.writer.flush()

    # --- train ---
    gan.fit(
        train_sr,
        validation_data=val_sr,
        epochs=args.epochs,
        callbacks=[tb, SaveGAN(5)],
    )

    # --- final generator export ---
    final_path = ckpt_dir.parent / "generator_final.keras"
    G.save(final_path)
    print("Saved:", final_path)


if __name__ == "__main__":
    main()
