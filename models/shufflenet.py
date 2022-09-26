import tensorflow as tf
from tensorflow import keras

# https://github.com/timctho/shufflenet-v2-tensorflow/blob/master/module.py
def shuffle_unit(x, groups):
    n, h, w, c = x.get_shape()
    x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
    x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
    x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
    return x

def conv_bn_relu(x, out, kernel, stride=1, dilation=1):
    x = keras.layers.Conv2D(out, kernel, strides=stride, dilation_rate=dilation, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x

def conv_bn_relu(x, out, kernel, stride=1, dilation=1):
    x = keras.layers.Conv2D(out, kernel, strides=stride, dilation_rate=dilation, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    return x

def depthwise_conv_bn(x, kernel, stride=1, dilation=1):
    x = keras.layers.DepthwiseConv2D(kernel, strides=stride, dilation_rate=dilation, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    return x

def stages():
    pass

def shuffle_block(x, out, kernel, stride=1, dilation=1, shuffle_group=2):
    if stride == 1: # downsampling
        top, bot = tf.split(x, num_or_size_splits=2, axis=3)
        half_channel = out // 2
        
        top = conv_bn_relu(top, half_channel, 1)
        top = depthwise_conv_bn(top, kernel, stride, dilation)
        top = conv_bn_relu(top, half_channel, 1)

        out = tf.concat([top, bot], axis=3)
        out = shuffle_unit(out, shuffle_group)
    else: # spatial downsampling
        half_channel = out // 2
        b0 = conv_bn_relu(x, half_channel, 1)
        b0 = depthwise_conv_bn(b0, kernel, stride, dilation)
        b0 = conv_bn_relu(b0, half_channel, 1)

        b1 = depthwise_conv_bn(x, kernel, stride, dilation)
        b1 = conv_bn_relu(b1, half_channel, 1)

        out = tf.concat([b0, b1], axis=3)
        out = shuffle_unit(out, shuffle_group)
    return out

def shufflenet(shape=(224, 224, 3)):
    groups = [(116, 4), (232, 8), (464, 4), (1024, 1)]
    input_layer = keras.layers.Input(shape)

    x = conv_bn_relu(input_layer, 24, 3, 2)
    x = keras.layers.MaxPooling2D()(x)

    for g in groups[:-1]:
        out_channel, repeats = g
        x = shuffle_block(x, out_channel, 3, 2, 2)

        for i in range(repeats - 1):
            x = shuffle_block(x, out_channel, 3, shuffle_group=2)
    
    x = conv_bn_relu(x, groups[-1][0], 1)
    model = keras.Model(input_layer, x)
    return model