import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import keras.backend as K
from keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPool2D,
    Dropout,
    BatchNormalization,
)
from keras.layers import LeakyReLU, concatenate, Concatenate, Input


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
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(input_tensor)
    # concatenation with encoder block
    if concat:
        x = concatenate([x, concat_layer])

    # decreasing the depth filters by half
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    return x


def loss_function(y_true, y_pred):
    
    K1 = 0.01 # 0.01
    K2 = 0.03 # 0.03
    _SSIM = 2 # 1
    _EDGES = 1.5 # 1
    _DEPTH = 0.2 # 0.1

    # Cosine distance loss
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

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
            y_true, y_pred, max_val=640, filter_size=7, k1=K1 ** 2, k2=K2 ** 2
        )
    )
    
    return (_SSIM * l_ssim) + (_EDGES * K.mean(l_edges)) + (_DEPTH * K.mean(l_depth))


# accuracy function
def accuracy_function(y_true, y_pred):
    return K.mean(
        tf.less_equal(
            tf.math.abs(tf.math.subtract(y_true, y_pred)), tf.constant([0.0625])
        )
    )
    # return K.mean(K.constant(tf.experimental.numpy.isclose(y_pred, y_true, 1e-6, 0.0625)))
    # return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


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


# optimizer
# opt = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-6, amsgrad=True)
opt = keras.optimizers.Adam(learning_rate=0.0005)
