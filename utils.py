import tensorflow as tf
import sys

def check_image_not_nan(img, name):
    """Determines if an image has a nan/inf value"""
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
    """Determines if the loss value is nan/inf and replace the value"""
    if tf.math.is_nan(loss):
        return tf.convert_to_tensor(0.0, dtype=tf.float32)
    if tf.experimental.numpy.isneginf(loss):
        return tf.convert_to_tensor(0.0, dtype=tf.float32)
    if tf.experimental.numpy.isposinf(loss):
        return tf.convert_to_tensor(2.0, dtype=tf.float32)
    return loss

def replace_nan_inf(losses):
    """replace nan and inf with zeros"""
    nans = tf.math.is_nan(losses)
    infs = tf.math.is_inf(losses)
    zeros = tf.zeros_like(losses)

    losses = tf.where(nans, zeros, losses)
    losses = tf.where(infs, zeros, losses)

    return losses

# accuracy function
def accuracy_function(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

# save model frequently for later use.
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "./checkpoints", save_best_only=True, verbose=1
)

# learning decay
def learning_decay(epoch, lr):
    if epoch >= 45:
        return 1e-5
    return 1e-4