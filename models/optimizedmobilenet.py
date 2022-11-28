import tensorflow as tf
from tensorflow import keras

# from utils import downsampling, UpSampling
from models.TCSVT import *

# https://medium.com/analytics-vidhya/creating-mobilenetsv2-with-tensorflow-from-scratch-c85eb8605342
def expansion_block(x, t, filters, block_id):
    prefix = "block_{}_".format(block_id)
    total_filters = t * filters
    x = keras.layers.Conv2D(
        total_filters, 1, padding="same", use_bias=False, name=prefix + "expand"
    )(x)
    x = keras.layers.BatchNormalization(name=prefix + "expand_bn")(x)
    x = keras.layers.ReLU(6, name=prefix + "expand_relu")(x)
    return x


def depthwise_block(x, stride, block_id):
    prefix = "block_{}_".format(block_id)
    x = DownSampling(x.shape[-1], x.shape[-1], stride=stride)(x)
    #     x = keras.layers.DepthwiseConv2D(3,strides=(stride,stride),padding ='same', use_bias = False, name = prefix + 'depthwise_conv')(x)
    x = keras.layers.BatchNormalization(name=prefix + "dw_bn")(x)
    x = keras.layers.ReLU(6, name=prefix + "dw_relu")(x)
    return x


def projection_block(x, out_channels, block_id):
    prefix = "block_{}_".format(block_id)
    x = keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "compress",
    )(x)

    x = keras.layers.BatchNormalization(name=prefix + "compress_bn")(x)
    return x


def Bottleneck(x, t, filters, out_channels, stride, block_id, expand=False):
    y = expansion_block(x, t, filters, block_id)
    if expand:
        return y

    y = depthwise_block(y, stride, block_id)
    y = projection_block(y, out_channels, block_id)
    if y.shape[-1] == x.shape[-1]:
        y = keras.layers.Add()([x, y])
    return y


def MobileNetV2(input_shape=(224, 224, 3)):
    input_layer = keras.layers.Input(input_shape)

    x = keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(
        input_layer
    )
    x = keras.layers.BatchNormalization(name="conv1_bn")(x)
    x = keras.layers.ReLU(6, name="conv1_relu")(x)

    # 13 Bottlenecks
    x = depthwise_block(x, stride=1, block_id=0)
    x = projection_block(x, out_channels=16, block_id=0)
    x1 = Bottleneck(x, t=6, filters=x.shape[-1], out_channels=24, stride=2, block_id=1)
    x2 = Bottleneck(
        x1, t=6, filters=x1.shape[-1], out_channels=24, stride=1, block_id=2
    )
    x3 = Bottleneck(
        x2, t=6, filters=x2.shape[-1], out_channels=32, stride=2, block_id=3
    )
    x4 = Bottleneck(
        x3, t=6, filters=x3.shape[-1], out_channels=32, stride=1, block_id=4
    )
    x5 = Bottleneck(
        x4, t=6, filters=x4.shape[-1], out_channels=32, stride=1, block_id=5
    )
    x6 = Bottleneck(
        x5, t=6, filters=x5.shape[-1], out_channels=64, stride=2, block_id=6
    )
    x7 = Bottleneck(
        x6, t=6, filters=x6.shape[-1], out_channels=64, stride=1, block_id=7
    )
    x8 = Bottleneck(
        x7, t=6, filters=x7.shape[-1], out_channels=64, stride=1, block_id=8
    )
    x9 = Bottleneck(
        x8, t=6, filters=x8.shape[-1], out_channels=64, stride=1, block_id=9
    )
    x10 = Bottleneck(
        x9, t=6, filters=x9.shape[-1], out_channels=96, stride=1, block_id=10
    )
    x11 = Bottleneck(
        x10, t=6, filters=x10.shape[-1], out_channels=96, stride=1, block_id=11
    )
    x12 = Bottleneck(
        x10, t=6, filters=x10.shape[-1], out_channels=96, stride=1, block_id=12
    )
    x13_expand = Bottleneck(
        x11,
        t=6,
        filters=x11.shape[-1],
        out_channels=96,
        stride=1,
        block_id=13,
        expand=True,
    )

    model = keras.Model(input_layer, x13_expand)
    return model


def OptimizedUNet(input_shape=(256, 512, 3)):
    base_model = MobileNetV2(input_shape)
    input_layer = base_model.input

    layer_names = ["block_6_expand_relu", "block_3_expand_relu", "block_1_expand_relu"]

    # Up Scaling
    x = base_model.output
    x = keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding="same")(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = UpSampling(x.shape[-1], 256)(
        x, concat=base_model.get_layer(layer_names[0]).output
    )
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = UpSampling(x.shape[-1], 128)(
        x, concat=base_model.get_layer(layer_names[1]).output
    )
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = UpSampling(x.shape[-1], 64)(
        x, concat=base_model.get_layer(layer_names[2]).output
    )
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = UpSampling(x.shape[-1], 32)(x, concat=base_model.layers[0].output)
    output = keras.layers.Conv2D(
        filters=1, activation="sigmoid", kernel_size=(3, 3), padding="same"
    )(x)

    model = keras.Model(input_layer, output)
    return model


def OptimizedUNet_Scene(input_shape=(256, 512, 3)):
    base_model = MobileNetV2(input_shape)
    input_layer = base_model.input

    layer_names = ["block_6_expand_relu", "block_3_expand_relu", "block_1_expand_relu"]

    # Up Scaling
    x = base_model.output
    x = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding="same")(x)

    x = Scene_Understanding(64, 128)(x)

    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = UpSampling(x.shape[-1], 128)(
        x, concat=base_model.get_layer(layer_names[0]).output
    )
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = UpSampling(x.shape[-1], 64)(
        x, concat=base_model.get_layer(layer_names[1]).output
    )
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = UpSampling(x.shape[-1], 32)(
        x, concat=base_model.get_layer(layer_names[2]).output
    )
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = UpSampling(x.shape[-1], 16)(x, concat=base_model.layers[0].output)
    output = keras.layers.Conv2D(
        filters=1, activation="sigmoid", kernel_size=(3, 3), padding="same"
    )(x)

    model = keras.Model(input_layer, output)
    return model
