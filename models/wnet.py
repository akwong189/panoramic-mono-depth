from glob import glob
import tensorflow as tf
from tensorflow import keras

from models.shufflenet import conv_bn_relu, shuffle_block

groups = [(116, 4), (232, 8), (464, 4), (1024, 1)]

def upsample_custom(x, out_channels, concat_layer=None):
    x = keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)
    x = keras.layers.Concatenate(axis=3)([concat_layer, x])
    x = keras.layers.Conv2D(out_channels, kernel_size=1, padding="same")(x)
    return x

def scene_understanding(x, out_channels):
    global_understanding = keras.layers.GlobalAveragePooling2D()(x)
    global_understanding = keras.layers.Dense(out_channels // 2)(global_understanding)
    global_understanding = keras.layers.Reshape((1, 1, global_understanding.shape[1]))(global_understanding)
    global_understanding = keras.backend.tile(global_understanding, [1, x.shape[1], x.shape[2], global_understanding.shape[1]])

    pixel_transform = keras.layers.Conv2D(out_channels // 2, kernel_size=1, padding="same")(x)

    aspp1 = keras.layers.Conv2D(out_channels // 4, kernel_size=3, padding="same")(x)
    aspp2 = keras.layers.Conv2D(out_channels // 4, kernel_size=3, dilation_rate=(1, 2), padding="same")(x)
    aspp3 = keras.layers.Conv2D(out_channels // 4, kernel_size=3, dilation_rate=(1, 4), padding="same")(x)
    aspp4 = keras.layers.Conv2D(out_channels // 4, kernel_size=3, dilation_rate=(2, 1), padding="same")(x)

    cat = keras.layers.Concatenate(axis=3)([global_understanding, pixel_transform, aspp1, aspp2, aspp3, aspp4])
    return keras.layers.Conv2D(out_channels * 2, kernel_size=1, padding="same")(cat)

def shufflenet_encoder(input_layer):
    skip_connections = []

    x = conv_bn_relu(input_layer, 24, 3, 2)
    skip_connections.append(x)
    x = keras.layers.MaxPooling2D()(x)
    skip_connections.append(x)

    for g in groups[:-1]:
        out_channel, repeats = g
        x = shuffle_block(x, out_channel, 3, 2, 2)

        for _ in range(repeats - 1):
            x = shuffle_block(x, out_channel, 3, shuffle_group=2)
        skip_connections.append(x)

    x = conv_bn_relu(x, groups[-1][0], 1)
    return x, skip_connections

def skip_connection_understanding(x, output_channels):
    # x = keras.layers.Conv2D(output_channels // 2, kernel_size=3, padding="same")(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(output_channels, kernel_size=3, padding="same")(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.ReLU()(x)
    return x

def upsample(x, concat, out_channels, skip_process=True, alpha=0.2):
    if skip_process:
        concat = skip_connection_understanding(concat, out_channels // 4)
        concat = keras.layers.LeakyReLU(alpha=alpha)(concat) # leaky ReLU could be slower than using ReLU
    x = keras.layers.LeakyReLU(alpha=alpha)(x)
    x = upsample_custom(x, out_channels, concat)
    return x

def wnet(input_shape=(256, 640, 3), w_scene=True, w_skip=True):
    input_layer = keras.layers.Input(input_shape)
    x, skip_connections = shufflenet_encoder(input_layer=input_layer)

    if w_scene:
        x = keras.layers.Conv2D(64, 1, padding="same")(x)
        x = scene_understanding(x, 128)

    x = upsample(x, skip_connections[-2], 128, skip_process=w_skip)
    x = upsample(x, skip_connections[-3], 64, skip_process=w_skip)
    x = upsample(x, skip_connections[-4], 32, skip_process=w_skip)
    x = upsample(x, skip_connections[-5], 16, skip_process=w_skip)

    # output resulting image
    x = keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)
    output = keras.layers.Conv2D(filters=1, activation="sigmoid", kernel_size=(3, 3), padding="same")(x) # can check for sigmoid

    model = keras.Model(input_layer, output)
    return model
