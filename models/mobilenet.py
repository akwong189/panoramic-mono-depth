import tensorflow as tf
from tensorflow import keras
from utils import upsampling


def MobileNet(shape=(256, 512, 3)):
    base_model = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=shape
    )
    for layer in base_model.layers:
        layer.trainable = True

    inputs = base_model.input
    x = base_model.get_layer("block_13_expand_relu").output

    names = [
        "input_6",
        "block_1_expand_relu",
        "block_3_expand_relu",
        "block_6_expand_relu",
    ][::-1]
    bneck = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding="same")(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(bneck)
    x = upsampling(x, 256, base_model.get_layer(names[0]).output)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 128, base_model.get_layer(names[1]).output)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 64, base_model.get_layer(names[2]).output)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 32, base_model.layers[0].output)
    x = keras.layers.Conv2D(filters=1, activation="sigmoid", kernel_size=(3, 3), padding="same")(x)

    model = keras.Model(inputs=inputs, outputs=x)
    return model
