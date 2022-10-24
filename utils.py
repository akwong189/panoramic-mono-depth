from re import I
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import numpy as np
import cv2
import sys


def depthwise_seperable_conv(inputs, nin, nout):
    x = keras.layers.Conv2D(nin, kernel_size=3, padding="same", groups=nin)(inputs)
    x = keras.layers.Conv2D(nout, kernel_size=1, padding="same")(x)
    return x


def LIST(inputs, in_channel, out_channel, k=4, nb=2):
    x = keras.layers.Conv2D(in_channel // k, kernel_size=(1, 1), padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x1 = keras.layers.Conv2D(out_channel // nb, kernel_size=1, padding="same")(x)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = keras.activations.relu(x1)

    x2 = depthwise_seperable_conv(x, in_channel // k, out_channel // nb)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.activations.relu(x2)

    return keras.layers.Concatenate(axis=3)([x1, x2])


class Group_Conv_Dilated(keras.layers.Layer):
    def __init__(self, in_channel, groups, dilation_factor, **kwargs):
        super(Group_Conv_Dilated, self).__init__(**kwargs)
        self.in_channel = in_channel
        self.num_groups = groups
        self.dilation_factor = dilation_factor

        self.partial = in_channel // groups

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "in_channel": self.in_channel,
                "groups": self.groups,
                "dilation_factor": self.dilation_factor,
            }
        )
        return config

    def call(self, inputs):
        groups = tf.split(inputs, self.num_groups, axis=3)
        #         print([group.shape for group in groups])
        dil_conv = []

        for group in groups:
            dil_conv.append(
                keras.layers.Conv2D(
                    self.partial,
                    kernel_size=(3, 3),
                    dilation_rate=self.dilation_factor,
                    padding="same",
                )(group)
            )

        return keras.layers.Concatenate(axis=3)(dil_conv)


class Channel_Shuffle(keras.layers.Layer):
    def __init__(self, groups, **kwargs):
        super(Channel_Shuffle, self).__init__(**kwargs)
        self.groups = groups

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "groups": self.groups,
            }
        )
        return config

    def call(self, inputs):
        shape = inputs.shape[1:]

        x = keras.layers.Reshape(
            [shape[0], shape[1], self.groups, shape[2] // self.groups]
        )(inputs)
        x = keras.layers.Permute([1, 2, 4, 3])(x)
        x = keras.layers.Reshape([shape[0], shape[1], shape[2]])(x)
        return x


class Group_Conv_Normal(keras.layers.Layer):
    def __init__(self, in_channel, groups, **kwargs):
        super(Group_Conv_Normal, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.num_groups = groups

        self.partial = in_channel // groups

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "in_channel": self.in_channel,
                "num_groups": self.num_groups,
            }
        )
        return config

    def call(self, inputs):
        groups = tf.split(inputs, self.num_groups, axis=3)
        #         print([group.shape for group in groups])
        dil_conv = []

        for group in groups:
            dil_conv.append(
                keras.layers.Conv2D(self.partial, kernel_size=(1, 1), padding="same")(
                    group
                )
            )

        return keras.layers.Concatenate(axis=3)(dil_conv)


class GSAT(keras.layers.Layer):
    def __init__(self, in_channel, num_groups=8, dilation_factor=2, **kwargs):
        super(GSAT, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.num_groups = num_groups
        self.dilation_factor = dilation_factor

        self.dil_conv = Group_Conv_Dilated(in_channel, num_groups, dilation_factor)
        self.shuffle = Channel_Shuffle(num_groups)
        self.norm_conv = Group_Conv_Normal(in_channel, num_groups)
        self.bn = keras.layers.BatchNormalization()
        self.add = keras.layers.Add()

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "in_channel": self.in_channel,
                "num_groups": self.num_groups,
                "dilation_factor": self.dilation_factor,
            }
        )
        return config

    def call(self, inputs):
        skip = inputs

        x = self.dil_conv(inputs)
        x = self.shuffle(x)
        x = self.norm_conv(x)
        x = self.bn(x)
        x = self.add([skip, x])

        return keras.activations.relu(x)


def new_upsampling(inputs, in_channel, out_channel, stride=2, concat=None):
    shape = inputs.shape
    x = tf.image.resize(inputs, [shape[1] * stride, shape[2] * stride])

    if concat is not None:
        # print("Concatenating")
        x = keras.layers.Concatenate(axis=3)([concat, x])

    x = LIST(x, in_channel, out_channel)
    return x


def downsampling(inputs, in_channel, out_channel, stride=2):
    shape = inputs.shape
    x = tf.image.resize(inputs, [shape[1] // stride, shape[2] // stride])
    return x


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
        # y_true = tfa.image.sharpness(y_true, 1)
        edge_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        edge_kernel = np.expand_dims(edge_kernel, 2)
        edge_kernel = np.expand_dims(edge_kernel, 3)
        edge_kernel = tf.constant(edge_kernel, dtype=tf.float32)
        # sharp_img = cv2.filter2D(y_true[0], ddepth=-1, kernel=edge_kernel)
        sharp_img = tf.nn.conv2d(y_true, edge_kernel, strides=[1,1,1,1], padding="SAME")
        y_true = sharp_img
        # sharp_img = tf.convert_to_tensor(sharp_img)
        # y_true = tf.expand_dims(sharp_img, 0)
        # y_true = tf.expand_dims(y_true, 3)
    l_ssim = 1 - tf.image.ssim(y_true, y_pred, max_val=640)
    return l_ssim
    # return keras.backend.clip(l_ssim * 0.5, 0, 1)

def sobel_loss(y_true, y_pred):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_x = np.expand_dims(sobel_x, 2)
    sobel_x = np.expand_dims(sobel_x, 3)
    sobel_x = tf.constant(sobel_x, dtype=tf.float32)

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.expand_dims(sobel_y, 2)
    sobel_y = np.expand_dims(sobel_y, 3)
    sobel_y = tf.constant(sobel_y, dtype=tf.float32)

    # y_true_x = keras.backend.conv2d(y_true, sobel_x, padding="same", data_format="channels_last")
    # y_true_y = keras.backend.conv2d(y_true, sobel_y, padding="same", data_format="channels_last")
    # y_pred_x = keras.backend.conv3d(y_pred, sobel_x, padding="same", dilation_rate=(1,1,1,1), strides=(1,1,1,1))
    # y_pred_y = keras.backend.conv3d(y_pred, sobel_y, padding="same", dilation_rate=(1,1,1,1), strides=(1,1,1,1))

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
    # ones = losses ** 0
    nans = tf.math.is_nan(losses)
    infs = tf.math.is_inf(losses)
    zeros = tf.zeros_like(losses)

    losses = tf.where(nans, zeros, losses)
    losses = tf.where(infs, zeros, losses)

    # avg = ones * tf.reduce_mean(losses)
    # losses = tf.where(nans, avg, losses)
    # losses = tf.where(infs, avg, losses)

    return losses


def new_new_loss(target, pred, debug=False):
    # Edges
    check_image_not_nan(pred, "predicted")
    check_image_not_nan(target, "target")

    # pred = tf.math.divide_no_nan(
    #     tf.math.subtract(
    #         pred,
    #         tf.math.reduce_min(pred)
    #     ),
    #     tf.math.subtract(
    #         tf.math.reduce_max(pred),
    #         tf.math.reduce_min(pred)
    #     )
    # )

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
    # loss_nan_inf("smoothness", depth_smoothness_loss)

    # Structural similarity (SSIM) index
    ssim_loss = (
        1
        - tf.image.ssim(
            target, pred, max_val=1.0, filter_size=11, k1=0.01 ** 2, k2=0.03 ** 2
        )
    )
    # tf.print(ssim_loss, summarize=-1)
    if tf.math.reduce_any(tf.math.is_nan(ssim_loss)) or tf.math.reduce_any(tf.math.is_inf(ssim_loss)):
        tf.print(ssim_loss, summarize=-1)
        ssim_loss = replace_nan_inf(ssim_loss)
        tf.print(ssim_loss, summarize=-1)
    ssim_loss = tf.reduce_mean(ssim_loss)

    # ssim_loss = loss_nan_inf("SSIM", ssim_loss)
    # if tf.math.is_nan(ssim_loss) or tf.math.is_inf(ssim_loss):
        # tf.print("SSIM is still NaN or Inf", output_stream=sys.stderr)
    # ssim_loss = tf.convert_to_tensor(ssim_l, dtype=tf.float32)

    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(target - pred))

    berhu = berhu_loss(target, pred)
    # loss_nan_inf("smoothness", berhu)
    sobel = sobel_loss(target, pred)
    # loss_nan_inf("smoothness", sobel)

    # if tf.math.is_nan(depth_smoothness_loss) or tf.math.is_inf(depth_smoothness_loss):
        # tf.print("Smoothness is nan or inf", output_stream=sys.stderr)
    # if tf.math.is_nan(berhu) or tf.math.is_inf(berhu):
        # tf.print("Berhu is nan or inf", output_stream=sys.stderr)
    # if tf.math.is_nan(sobel) or tf.math.is_inf(berhu):
        # tf.print("Sobel is nan or inf", output_stream=sys.stderr)

    loss = (
        (1.3 * ssim_loss) # 0.95
        + (1.1 * depth_smoothness_loss) # 1.1
        + (3.35 * berhu) # 0.35
        + (3.7 * sobel) # 0.75
    )
    # if tf.math.is_nan(loss) or tf.math.is_inf(loss):
        # tf.print("WARNING LOSS IS NOT VALID")
    # else:
        # tf.print(loss)

    if debug or tf.math.is_nan(ssim_loss):
        keras.backend.print_tensor(pred)
        keras.backend.print_tensor(tf.math.reduce_min(pred))
        keras.backend.print_tensor(tf.math.reduce_max(pred))
        keras.backend.print_tensor(target)
        keras.backend.print_tensor(tf.math.reduce_min(target))
        keras.backend.print_tensor(tf.math.reduce_max(target))
        keras.backend.print_tensor(ssim_loss)


    return loss

# accuracy function
def accuracy_function(y_true, y_pred):
    # return keras.backend.mean(
        # tf.less_equal(
            # tf.math.abs(tf.math.subtract(y_true, y_pred)), tf.constant([0.0625])
        # )
    # )
    # return K.mean(K.constant(tf.experimental.numpy.isclose(y_pred, y_true, 1e-6, 0.0625)))
    # return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


# save model frequently for later use.
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "./checkpoints", save_best_only=True, verbose=1
)
# Learning rate scheduler
def polynomial_decay(epoch, lr):
    max_epochs = 100
    base_lr = 0.0001
    power = 1.0
    lr = base_lr * (1 - (epoch / float(max_epochs))) ** power
    return lr

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
    if epoch >= 35:
        return 1e-8
    if epoch >= 25:
        return 1e-7
    if epoch >= 10:
        return 1e-6
    return 1e-5

# optimizer
# opt = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-6, amsgrad=True)
opt = keras.optimizers.Adam(learning_rate=0.00001)
