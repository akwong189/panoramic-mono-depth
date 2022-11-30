# Model
import tensorflow as tf
from tensorflow import keras
from layers import upsampling


def EfficientUNet(*kargs):
    base_model = keras.applications.efficientnet.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(256, 512, 3)
    )
    for layer in base_model.layers:
        layer.trainable = True

    inputs = base_model.input
    x = base_model.output

    names = [
        "block1a_activation",
        "block2b_activation",
        "block3a_activation",
        "block4a_activation",
    ][::-1]
    bneck = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding="same")(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(bneck)
    x = upsampling(bneck, 256, base_model.get_layer(names[0]).output)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 128, base_model.get_layer(names[1]).output)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 64, base_model.get_layer(names[2]).output)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 32, base_model.get_layer(names[3]).output)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 16, base_model.layers[0].output)
    x = keras.layers.Conv2D(
        filters=1, activation="sigmoid", kernel_size=(3, 3), padding="same"
    )(x)

    model = keras.Model(inputs=inputs, outputs=x)
    return model
