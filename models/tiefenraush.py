import tensorflow as tf
from tensorflow import keras

def t_block(x, out_channels, s_u, s_d):
    x = keras.layers.Conv2D(filters=x.shape[2], kernel_size=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.UpSampling2D(size=())(x)
    x = keras.

def tiefenraush(input_shape=()):
    input_layer = keras.layers.Input(input_shape)

    x = keras.layers.Conv2D(filters=384, kernel_size=(), padding="same")

