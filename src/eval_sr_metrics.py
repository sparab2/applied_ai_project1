# src/eval_sr_metrics.py
import argparse, csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.datasets import load_sr_pairs, IMG_HR
from src.models_srgan import InstanceNorm, PixelShuffle


def to01(x):  # [-1,1] -> [0,1] in float32
    x = tf.cast(x, tf.float32)
    return tf.clip_by_value((x + 1.0) * 0.5, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gen", required=True, help="path to generator_final.keras")
    ap.add_argument("--out", required=True, help="dir to write CSV and TensorBoard")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load generator (register custom layers)
    G = keras.models.load_model(
        args.gen,
        compile=False,
        custom_objects={"InstanceNorm": InstanceNorm, "PixelShuffle": PixelShuffle},
    )

    # We only need the validation SR pairs here
    _, val_sr = load_sr_pairs(args.data, batch=args.batch)

    # TensorBoard writer
    tb_dir = outdir / "tb"
    writer = tf.summary.create_file_writer(str(tb_dir))

    rows = [("idx", "psnr", "ssim")]
    psnrs, ssims = [], []
    idx = 0

    # Optional preview grid from first batch
    first = True

    for lr, hr in val_sr:
        sr = G(lr, training=False)

        lr01 = to01(lr)
        hr01 = to01(hr)
        sr01 = to01(sr)

        if first:
            # Make side-by-side grid: LR↑ | SR | HR
            lr_up = tf.image.resize(lr01, (IMG_HR, IMG_HR), method="nearest")
            grid = tf.concat([lr_up, sr01, hr01], axis=2)  # concat along width
            img = tf.concat(tf.unstack(grid[:4], axis=0), axis=0)
            keras.utils.save_img(str(outdir / "preview_grid.png"), img)
            first = False

        # Per-image metrics
        b = sr01.shape[0]
        for i in range(b):
            p = tf.image.psnr(sr01[i], hr01[i], max_val=1.0).numpy().item()
            s = tf.image.ssim(sr01[i], hr01[i], max_val=1.0).numpy().item()
            rows.append((idx, p, s))
            psnrs.append(p)
            ssims.append(s)
            idx += 1

    # CSV
    csv_path = outdir / "sr_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # TB scalars
    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    with writer.as_default():
        tf.summary.scalar("SR/val_psnr_mean", float(psnrs.mean()), step=0)
        tf.summary.scalar("SR/val_ssim_mean", float(ssims.mean()), step=0)

    print(f"[eval] Wrote {csv_path}")
    print(f"[eval] Mean PSNR={psnrs.mean():.3f}  Mean SSIM={ssims.mean():.4f}")
    print(f"[eval] TensorBoard logs: {tb_dir}")
    print(f"[eval] Saved preview image: {outdir/'preview_grid.png'}")


if __name__ == "__main__":
    main()
