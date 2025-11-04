# src/models_cls.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_classifier(num_classes: int, input_shape=(128, 128, 3), base="MobileNetV2"):
    """
    For binary problems (num_classes == 2) we use a 1-unit sigmoid head and
    binary_crossentropy. For >2 classes we use softmax and categorical losses.
    """
    # pick the backbone
    base = base.lower()
    if base == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess = keras.applications.mobilenet_v2.preprocess_input
    elif base == "resnet50":
        backbone = keras.applications.ResNet50(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess = keras.applications.resnet.preprocess_input
    else:
        raise ValueError(f"Unknown base: {base}")

    backbone.trainable = False  # fine-tune later if you want

    inp = keras.Input(shape=input_shape)
    x = preprocess(inp)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="acc"),
                keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        out = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
                keras.metrics.AUC(name="auc", multi_label=True),
            ],
        )

    return model
