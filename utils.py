from re import I
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys


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

# DenseDepth
def loss_function(y_true, y_pred):

    K1 = 0.01  # 0.01
    K2 = 0.03  # 0.03
    _SSIM = 2  # 1
    _EDGES = 1.5  # 1
    _DEPTH = 0.2  # 0.1

    # Cosine distance loss
    l_depth = keras.backend.mean(keras.backend.abs(y_pred - y_true), axis=-1)

    # edge loss for sharp edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    # l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
    l_edges = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

    # structural similarity loss
    # l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)
    l_ssim = tf.reduce_mean(
        1
        - tf.image.ssim(
            y_true, y_pred, max_val=1, filter_size=7, k1=K1**2, k2=K2**2
        )
    )

    return (_SSIM * l_ssim) + (_EDGES * keras.backend.mean(l_edges)) + (_DEPTH * keras.backend.mean(l_depth))

# based on the results of this paper: https://www.sciencedirect.com/science/article/pii/S2666827021001092
def berhu_loss(y_true, y_pred, threshold=0.2):
    bounds = threshold * keras.backend.max(keras.backend.abs(y_pred - y_true))
    l1 = keras.backend.abs(y_pred - y_true)
    l2 = (keras.backend.square(y_pred - y_true) + (bounds ** 2)) / (2 * bounds)
    l1_mask = tf.cast((l1 <= bounds), tf.float32)
    res = l1 * l1_mask + l2 * tf.cast(l1_mask == 0, tf.float32)
    return keras.backend.mean(res)

def ssim_loss(y_true, y_pred, sharpen=True):
    if sharpen:
        edge_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        edge_kernel = np.expand_dims(edge_kernel, 2)
        edge_kernel = np.expand_dims(edge_kernel, 3)
        edge_kernel = tf.constant(edge_kernel, dtype=tf.float32)
        sharp_img = tf.nn.conv2d(y_true, edge_kernel, strides=[1,1,1,1], padding="SAME")
        y_true = sharp_img
    l_ssim = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
    return l_ssim

def sobel_loss(y_true, y_pred):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_x = np.expand_dims(sobel_x, 2)
    sobel_x = np.expand_dims(sobel_x, 3)
    sobel_x = tf.constant(sobel_x, dtype=tf.float32)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.expand_dims(sobel_y, 2)
    sobel_y = np.expand_dims(sobel_y, 3)
    sobel_y = tf.constant(sobel_y, dtype=tf.float32)

    y_true_x = tf.nn.conv2d(y_true, sobel_x, strides=[1,1,1,1], padding="SAME")
    y_true_y = tf.nn.conv2d(y_true, sobel_y, strides=[1,1,1,1], padding="SAME")
    y_pred_x = tf.nn.conv2d(y_pred, sobel_x, strides=[1,1,1,1], padding="SAME")
    y_pred_y = tf.nn.conv2d(y_pred, sobel_y, strides=[1,1,1,1], padding="SAME")

    diff_x = tf.math.abs(y_pred_x - y_true_x)
    diff_y = tf.math.abs(y_pred_y - y_true_y)

    return keras.backend.mean(diff_x + diff_y)

def new_loss_function(y_true, y_pred, l1=1.5, l2=0.5, l3=1):
    return l1 * berhu_loss(y_true, y_pred) + l2 * ssim_loss(y_true, y_pred) + l3 * sobel_loss(y_true, y_pred)

def check_image_not_nan(img, name):
    check_nan = tf.math.is_nan(img)
    check_inf = tf.math.is_inf(img)
    if tf.math.reduce_any(check_nan):
        tf.print(f"\n{name} has nan...", output_stream=sys.stderr)
        return False
    if tf.math.reduce_any(check_inf):
        tf.print(f"\n{name} has inf...", output_stream=sys.stderr)
        return False
    return True

def loss_nan_inf(name, loss):
    if tf.math.is_nan(loss):
        # tf.print(f"{name} is nan", output_stream=sys.stderr)
        return tf.convert_to_tensor(0.0, dtype=tf.float32)
    if tf.experimental.numpy.isneginf(loss):
        # tf.print(f"{name} is -inf", output_stream=sys.stderr)
        return tf.convert_to_tensor(0.0, dtype=tf.float32)
    if tf.experimental.numpy.isposinf(loss):
        # tf.print(f"{name} is inf", output_stream=sys.stderr)
        return tf.convert_to_tensor(2.0, dtype=tf.float32)
    return loss

def replace_nan_inf(losses):
    nans = tf.math.is_nan(losses)
    infs = tf.math.is_inf(losses)
    zeros = tf.zeros_like(losses)

    losses = tf.where(nans, zeros, losses)
    losses = tf.where(infs, zeros, losses)

    return losses


def new_new_loss(target, pred, debug=False):
    # Edges
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
        abs(smoothness_y)
    )

    # Edges
    dy_true, dx_true = tf.image.image_gradients(target)
    dy_pred, dx_pred = tf.image.image_gradients(pred)
    l_edges = tf.keras.backend.mean(tf.keras.backend.abs(dy_pred - dy_true) + tf.keras.backend.abs(dx_pred - dx_true), axis=-1)
    l_edges = tf.reduce_mean(l_edges)

    # Structural similarity (SSIM) index
    ssim_loss = (
        1
        - tf.image.ssim(
            target, pred, max_val=1.0# , filter_size=11, k1=0.01 ** 2, k2=0.03 ** 2
        )
    )
    ssim_loss = tf.reduce_mean(ssim_loss)

    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(target - pred))

    berhu = berhu_loss(target, pred)
    sobel = sobel_loss(target, pred)

    loss = (
        (1 * ssim_loss) # 0.95
        + (1 * l1_loss)
        + (1 * l_edges)
        + (1 * depth_smoothness_loss) # 1.1
        # + (1 * berhu) # 0.35
        + (1 * sobel) # 0.75
    )

    return loss

# accuracy function
def accuracy_function(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

# save model frequently for later use.
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "./checkpoints", save_best_only=True, verbose=1
)

def learning_decay(epoch, lr):
    # if epoch >= 80:
        # return 0.0000001
    # if epoch >= 25:
        # return 0.000001
    # if epoch >= 15:
        # return 0.00001
    # if epoch >= 5:
        # return 0.0001
    # return 0.001

    # test 1
    # if epoch >= 35:
    #     return 1e-7
    # if epoch >= 5:
    #     return 1e-6
    # return 1e-5

    # test 2
    # if epoch >= 20:
    #     return 1e-6
    # if epoch >= 8:
    #     return 1e-5
    # if epoch >= 4:
    #     return 1e-4
    # if epoch >= 2:
    #     return 1e-3
    # if epoch >= 1:
    #     return 1e-2
    # return 1e-1

    if epoch >= 45:
        return 1e-5
    return 1e-4