import tensorflow as tf
from tensorflow import keras

def upsampling(input_tensor, n_filters, concat_layer, concat=True):
    """
    Block of Decoder
    """
    # Bilinear 2x upsampling layer
    x = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(input_tensor)
    # concatenation with encoder block
    if concat:
        x = keras.layers.concatenate([x, concat_layer])

    # decreasing the depth filters by half
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    return x