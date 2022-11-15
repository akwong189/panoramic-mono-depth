import tensorflow as tf
from tensorflow import keras
import numpy as np

#!!!ADD LOSS MODULES HERE !!!

# based on the results of this paper: https://www.sciencedirect.com/science/article/pii/S2666827021001092
def berhu_loss(y_true, y_pred, threshold=0.2):
    bounds = threshold * keras.backend.max(keras.backend.abs(y_pred - y_true))
    l1 = keras.backend.abs(y_pred - y_true)
    l2 = (keras.backend.square(y_pred - y_true) + (bounds ** 2)) / (2 * bounds)
    l1_mask = tf.cast((l1 <= bounds), tf.float32)
    res = l1 * l1_mask + l2 * tf.cast(l1_mask == 0, tf.float32)
    return keras.backend.mean(res)

def ssim_loss(y_true, y_pred, sharpen=False):
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

def edge_loss(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = tf.keras.backend.mean(tf.keras.backend.abs(dy_pred - dy_true) + tf.keras.backend.abs(dx_pred - dx_true), axis=-1)
    l_edges = tf.reduce_mean(l_edges)
    return l_edges

def depth_smoothness_loss(y_true, y_pred):
    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y

    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
        abs(smoothness_y)
    )
    return depth_smoothness_loss

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true, y_pred))