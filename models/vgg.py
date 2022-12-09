import tensorflow as tf
from tensorflow import keras
from layers import upsampling
from models.TCSVT import *


def VGG(shape=(256, 512, 3)):
    inputs = keras.layers.Input(shape=shape)

    x = keras.layers.Conv2D(64, 3, padding="same")(inputs)
    x1 = keras.layers.Conv2D(64, 3, padding="same")(x)
    x = keras.layers.MaxPool2D()(x1)
    x = keras.layers.Conv2D(128, 3, padding="same")(x)
    x2 = keras.layers.Conv2D(128, 3, padding="same")(x)
    x = keras.layers.MaxPool2D()(x2)
    x = keras.layers.Conv2D(256, 3, padding="same")(x)
    x3 = keras.layers.Conv2D(256, 3, padding="same")(x)
    x = keras.layers.MaxPool2D()(x3)
    x = keras.layers.Conv2D(512, 3, padding="same")(x)
    x4 = keras.layers.Conv2D(512, 3, padding="same")(x)
    x = keras.layers.MaxPool2D()(x4)
    x = keras.layers.Conv2D(512, 3, padding="same")(x)
    x = keras.layers.Conv2D(512, 3, padding="same")(x)
    x = keras.layers.Conv2D(512, 3, padding="same")(x)

    x = Scene_Understanding(512, 256)(x)

    bneck = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding="same")(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(bneck)
    x = upsampling(x, 256, x4)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 128, x3)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 64, x2)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 32, x1)
    x = keras.layers.Conv2D(
        filters=1, activation="sigmoid", kernel_size=(3, 3), padding="same"
    )(x)

    model = keras.Model(inputs=inputs, outputs=x)
    return model
