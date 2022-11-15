from re import I
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
from losses import *

loss2func = {
    "ssim": ssim_loss,
    "l1": l1_loss,
    "berhu": berhu_loss,
    "sobel": sobel_loss,
    "smooth": depth_smoothness_loss,
    "edges": edge_loss
}

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

"""Returns a loss function based on the selected losses"""
def set_loss_function(selected_loss):
    def loss_function(y_true, y_pred):
        loss = 0
        for l in selected_loss:
            loss += 1 * loss2func[l](y_true, y_pred)
        return loss

    return loss_function

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