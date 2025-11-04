# src/datasets.py
"""
Dataset utilities for Applied AI Midterm (SRGAN + Classification A/B)
Author: sparab2

Handles:
  - Binary classification dataset loading (cats/dogs)
  - HR (128×128) and LR (32×32) pairs for SRGAN
  - On-the-fly SR generation for classifier B
"""

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_HR = 128
IMG_LR = 32

def to_m11(x):
    return tf.cast(x, tf.float32) / 127.5 - 1.0

def hr_to_lr(hr):
    x01 = (hr + 1.0) * 0.5
    lr = tf.image.resize(x01, (IMG_LR, IMG_LR), method="bicubic")
    return lr * 2.0 - 1.0

augment_cls = keras.Sequential(
    [layers.RandomFlip("horizontal"),
     layers.RandomRotation(0.05),
     layers.RandomZoom(0.1)],
    name="aug_cls"
)

def load_classifier_ds(data_dir, batch=32, seed=42):
    train_hr = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode="int", validation_split=0.30, subset="training",
        seed=seed, image_size=(IMG_HR, IMG_HR), batch_size=batch, shuffle=True
    )
    val_hr = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode="int", validation_split=0.30, subset="validation",
        seed=seed, image_size=(IMG_HR, IMG_HR), batch_size=batch, shuffle=False
    )
    classes = train_hr.class_names
    def prep_train(x, y): return augment_cls(to_m11(x)), y
    def prep_eval(x, y):  return to_m11(x), y
    return (train_hr.map(prep_train).prefetch(tf.data.AUTOTUNE),
            val_hr.map(prep_eval).prefetch(tf.data.AUTOTUNE),
            classes)

def load_sr_pairs(data_dir, batch=32, seed=42):
    train_hr = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode=None, validation_split=0.30, subset="training",
        seed=seed, image_size=(IMG_HR, IMG_HR), batch_size=batch, shuffle=True
    )
    val_hr = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode=None, validation_split=0.30, subset="validation",
        seed=seed, image_size=(IMG_HR, IMG_HR), batch_size=batch, shuffle=False
    )
    def pair(x):
        hr = to_m11(x)
        lr = hr_to_lr(hr)
        return lr, hr
    return (train_hr.map(pair).prefetch(tf.data.AUTOTUNE),
            val_hr.map(pair).prefetch(tf.data.AUTOTUNE))

def make_sr_classifier_ds(data_dir, generator, batch=32, seed=42):
    train_hr = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode="int", validation_split=0.30, subset="training",
        seed=seed, image_size=(IMG_HR, IMG_HR), batch_size=batch, shuffle=True
    )
    val_hr = tf.keras.utils.image_dataset_from_directory(
        data_dir, label_mode="int", validation_split=0.30, subset="validation",
        seed=seed, image_size=(IMG_HR, IMG_HR), batch_size=batch, shuffle=False
    )
    def to_sr(x, y):
        hr = to_m11(x)
        lr = hr_to_lr(hr)
        sr = generator(lr, training=False)
        return sr, y
    return (train_hr.map(to_sr).prefetch(tf.data.AUTOTUNE),
            val_hr.map(to_sr).prefetch(tf.data.AUTOTUNE))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save_preview", default="")
    args = ap.parse_args()

    print("[datasets] Loading classifier ds from:", args.data)
    train_ds, val_ds, classes = load_classifier_ds(args.data, batch=args.batch)
    print("[datasets] Classes:", classes)
    xb, yb = next(iter(train_ds.take(1)))
    print("[datasets] Train batch:", xb.shape, yb.shape, xb.dtype)

    train_sr, val_sr = load_sr_pairs(args.data, batch=args.batch)
    lr, hr = next(iter(train_sr.take(1)))
    print("[datasets] SR pair shapes: LR", lr.shape, "HR", hr.shape)

    if args.save_preview:
        x = (xb[:16] + 1.0) * 0.5
        rows = []
        for i in range(0, x.shape[0], 4):
            rows.append(tf.concat(list(x[i:i+4]), axis=1))
        grid = tf.concat(rows, axis=0)
        keras.utils.save_img(args.save_preview, grid)
        print("[datasets] Wrote preview to:", args.save_preview)
