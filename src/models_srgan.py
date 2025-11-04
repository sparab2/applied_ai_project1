# src/models_srgan.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_HR = 128
IMG_LR = 32
BCE = keras.losses.BinaryCrossentropy(from_logits=False)


class InstanceNorm(layers.Layer):
    def __init__(self, eps=1e-5, **kw):
        super().__init__(**kw)
        self.eps = eps

    def build(self, input_shape):
        c = int(input_shape[-1])
        # Force FP32 variables regardless of global mixed-precision policy
        self.gamma = self.add_weight(
            name="gamma",
            shape=(c,),
            dtype=tf.float32,          # <-- force float32 (use tf.dtype, not a string)
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(c,),
            dtype=tf.float32,          # <-- force float32
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        # Do all norm math in float32, then cast back to input dtype
        x32 = tf.cast(x, tf.float32)
        mean, var = tf.nn.moments(x32, axes=[1, 2], keepdims=True)
        y32 = (x32 - mean) / tf.sqrt(var + self.eps)
        # Ensure variables are also float32 in case policy tries to autocast
        y32 = tf.cast(self.gamma, tf.float32) * y32 + tf.cast(self.beta, tf.float32)
        return tf.cast(y32, x.dtype)


class PixelShuffle(layers.Layer):
    def __init__(self, scale=2, **kw):
        super().__init__(**kw)
        self.scale = scale
    def call(self, x):
        # do the op in a layer (legal for KerasTensor)
        return tf.nn.depth_to_space(x, block_size=self.scale)


def residual_block(x, f=64):
    s = x
    x = layers.Conv2D(f,3,padding="same")(x); x = InstanceNorm()(x); x = layers.PReLU(shared_axes=[1,2])(x)
    x = layers.Conv2D(f,3,padding="same")(x); x = InstanceNorm()(x)
    return layers.Add()([s,x])

def upsample_ps(x, scale=2, f=64):
    x = layers.Conv2D(f * (scale ** 2), 3, padding="same")(x)
    x = PixelShuffle(scale)(x)
    return layers.PReLU(shared_axes=[1, 2])(x)


def build_generator():
    inp = layers.Input((IMG_LR,IMG_LR,3))
    x = layers.Conv2D(64,9,padding="same")(inp); x = x1 = layers.PReLU(shared_axes=[1,2])(x)
    for _ in range(8): x = residual_block(x,64)
    x = layers.Conv2D(64,3,padding="same")(x); x = InstanceNorm()(x)
    x = layers.Add()([x, x1])
    x = upsample_ps(x, 2, 64); x = upsample_ps(x, 2, 64)
    out = layers.Conv2D(3,9,activation="tanh",padding="same")(x)
    return keras.Model(inp, out, name="G")

def disc_block(x,f,s=1,bn=True):
    x = layers.Conv2D(f,3,strides=s,padding="same")(x)
    if bn: x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.2)(x)

def build_discriminator():
    inp = layers.Input((IMG_HR,IMG_HR,3))
    x = disc_block(inp,64,1,False); x = disc_block(x,64,2)
    x = disc_block(x,128,1); x = disc_block(x,128,2)
    x = disc_block(x,256,1); x = disc_block(x,256,2)
    x = disc_block(x,512,1); x = disc_block(x,512,2)
    x = layers.Flatten()(x); x = layers.Dense(1024)(x); x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="D")

def build_vgg_feature_extractor():
    vgg = keras.applications.VGG19(include_top=False, weights="imagenet",
                                   input_shape=(IMG_HR,IMG_HR,3))
    fe = keras.Model(vgg.input, vgg.get_layer("block5_conv4").output, name="VGG19_FE")
    fe.trainable = False
    return fe

class SRGAN(keras.Model):
    def __init__(self, G, D, VGG, lam_content=1.0, lam_adv=1e-3):
        super().__init__()
        self.G = G
        self.D = D
        self.VGG = VGG
        self.lc = lam_content
        self.la = lam_adv
        self.mse = keras.losses.MeanSquaredError()
        self.bce = BCE
        self.dacc = keras.metrics.BinaryAccuracy(name="d_acc")

    # Let Keras know how to do a forward pass (used in eval/summary/etc.)
    def call(self, inputs, training=False):
        # inputs is LR; return SR
        return self.G(inputs, training=training)

    def compile(self, g_opt, d_opt):
        super().compile()
        self.g_opt = g_opt
        self.d_opt = d_opt

    def train_step(self, data):
        lr, hr = data

        # ----- Discriminator step -----
        with tf.GradientTape() as t:
            sr = self.G(lr, training=True)
            r  = self.D(hr, training=True)
            f  = self.D(sr, training=True)
            d_real = self.bce(tf.ones_like(r), r)
            d_fake = self.bce(tf.zeros_like(f), f)
            d_loss = 0.5 * (d_real + d_fake)
        dgr = t.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(dgr, self.D.trainable_variables))

        # ----- Generator step -----
        with tf.GradientTape() as t:
            sr = self.G(lr, training=True)
            f  = self.D(sr, training=True)
            # Perceptual/content loss via VGG19 (expects [0,255] preprocessed)
            sr01 = (sr + 1.0) * 0.5
            hr01 = (hr + 1.0) * 0.5
            srF = self.VGG(keras.applications.vgg19.preprocess_input(sr01 * 255.0), training=False)
            hrF = self.VGG(keras.applications.vgg19.preprocess_input(hr01 * 255.0), training=False)
            content = self.mse(hrF, srF)
            adv = self.bce(tf.ones_like(f), f)
            g_loss = self.lc * content + self.la * adv
        ggr = t.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(ggr, self.G.trainable_variables))

        # Metrics
        self.dacc.update_state(tf.concat([tf.ones_like(r), tf.zeros_like(f)], axis=0),
                               tf.concat([r, f], axis=0))
        return {"d_loss": d_loss, "g_loss": g_loss, "d_acc": self.dacc.result()}

    # Tell Keras how to run validation
    def test_step(self, data):
        lr, hr = data
        # We won’t update weights; just compute the same scalar logs
        sr = self.G(lr, training=False)
        r  = self.D(hr, training=False)
        f  = self.D(sr, training=False)

        d_real = self.bce(tf.ones_like(r), r)
        d_fake = self.bce(tf.zeros_like(f), f)
        d_loss = 0.5 * (d_real + d_fake)

        sr01 = (sr + 1.0) * 0.5
        hr01 = (hr + 1.0) * 0.5
        srF = self.VGG(keras.applications.vgg19.preprocess_input(sr01 * 255.0), training=False)
        hrF = self.VGG(keras.applications.vgg19.preprocess_input(hr01 * 255.0), training=False)
        content = self.mse(hrF, srF)
        adv = self.bce(tf.ones_like(f), f)
        g_loss = self.lc * content + self.la * adv

        self.dacc.update_state(tf.concat([tf.ones_like(r), tf.zeros_like(f)], axis=0),
                               tf.concat([r, f], axis=0))
        return {"val_d_loss": d_loss, "val_g_loss": g_loss, "val_d_acc": self.dacc.result()}
