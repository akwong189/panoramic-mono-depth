import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPool2D,
    Dropout,
    BatchNormalization,
)
from keras.layers import LeakyReLU, concatenate, Concatenate, Input
from keras import Model
from utils import upsampling
from models.TCSVT import *


def VGG(shape=(256, 512, 3)):
    inputs = Input(shape=shape)

    x = Conv2D(64, 3, padding="same")(inputs)
    x1 = Conv2D(64, 3, padding="same")(x)
    x = MaxPool2D()(x1)
    x = Conv2D(128, 3, padding="same")(x)
    x2 = Conv2D(128, 3, padding="same")(x)
    x = MaxPool2D()(x2)
    x = Conv2D(256, 3, padding="same")(x)
    x3 = Conv2D(256, 3, padding="same")(x)
    x = MaxPool2D()(x3)
    x = Conv2D(512, 3, padding="same")(x)
    x4 = Conv2D(512, 3, padding="same")(x)
    x = MaxPool2D()(x4)
    x = Conv2D(512, 3, padding="same")(x)
    x = Conv2D(512, 3, padding="same")(x)
    x = Conv2D(512, 3, padding="same")(x)

    x = Scene_Understanding(512, 256)(x)

    bneck = Conv2D(filters=512, kernel_size=(1, 1), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(bneck)
    x = upsampling(x, 256, x4)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 128, x3)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 64, x2)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 32, x1)
    x = Conv2D(filters=1, activation="sigmoid", kernel_size=(3, 3), padding="same")(x)

    model = Model(inputs=inputs, outputs=x)
    return model
