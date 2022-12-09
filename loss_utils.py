import tensorflow as tf
from tensorflow import keras
import numpy as np
from loss import *
from options import loss2func

#!!! Set Loss Functions Here !!!
def set_loss_function(selected_loss):
    """Returns a loss function based on the selected losses"""
    def loss_function(y_true, y_pred):
        loss = 0
        for l in selected_loss:
            loss += 1 * loss2func[l](y_true, y_pred)
        return loss

    return loss_function

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

def new_loss_function(y_true, y_pred, l1=1.5, l2=0.5, l3=1):
    return l1 * berhu_loss(y_true, y_pred) + l2 * ssim_loss(y_true, y_pred) + l3 * sobel_loss(y_true, y_pred)

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