import tensorflow as tf
from tensorflow import keras


class Depthwise_Seperable_Conv(keras.layers.Layer):
    def __init__(self, nin, nout, name="", **kwargs):
        super(Depthwise_Seperable_Conv, self).__init__(**kwargs)

        self.nin = nin
        self.nout = nout

        self.depthwise = keras.layers.Conv2D(
            nin, kernel_size=3, padding="same", groups=nin
        )
        self.pointwise = keras.layers.Conv2D(nout, kernel_size=1, padding="same")

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "nin": self.nin,
                "nout": self.nout,
            }
        )
        return config

    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        return x


class LIST(keras.layers.Layer):
    def __init__(self, in_channel, out_channel, k=4, nb=2, **kwargs):
        super(LIST, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k = k
        self.nb = nb

        self.squeeze = keras.layers.Conv2D(
            in_channel // k, kernel_size=(1, 1), padding="same"
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.stream1 = keras.layers.Conv2D(
            out_channel // nb, kernel_size=1, padding="same"
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.stream2 = Depthwise_Seperable_Conv(in_channel // k, out_channel // nb)
        self.bn3 = keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "in_channel": self.in_channel,
                "out_channel": self.out_channel,
                "k": self.k,
                "nb": self.nb,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.bn1(x)
        x = keras.activations.relu(x)

        # stream 1
        x1 = self.stream1(x)
        x1 = self.bn2(x1)
        x1 = keras.activations.relu(x1)

        # stream 2
        x2 = self.stream2(x)
        x2 = self.bn3(x2)
        x2 = keras.activations.relu(x2)

        return keras.layers.Concatenate(axis=3)([x1, x2])


class Group_Conv_Dilated(keras.layers.Layer):
    def __init__(self, in_channel, groups, dilation_factor, **kwargs):
        super(Group_Conv_Dilated, self).__init__(**kwargs)
        self.in_channel = in_channel
        self.groups = groups
        self.dilation_factor = dilation_factor

        self.partial = in_channel // groups
        self.layers = []

        for i in range(groups):
            self.layers.append(
                keras.layers.Conv2D(
                    self.partial,
                    kernel_size=(3, 3),
                    dilation_rate=self.dilation_factor,
                    padding="same",
                )
            )

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
        groups = tf.split(inputs, self.groups, axis=3)
        #         print([group.shape for group in groups])
        dil_conv = []

        for i in range(self.groups):
            dil_conv.append(self.layers[i](groups[i]))

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
        self.groups = groups

        self.partial = in_channel // groups
        self.layers = []

        for i in range(groups):
            self.layers.append(
                keras.layers.Conv2D(self.partial, kernel_size=(1, 1), padding="same")
            )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "in_channel": self.in_channel,
                "groups": self.groups,
            }
        )
        return config

    def call(self, inputs):
        groups = tf.split(inputs, self.groups, axis=3)
        #         print([group.shape for group in groups])
        dil_conv = []

        for i in range(self.groups):
            dil_conv.append(self.layers[i](groups[i]))

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


class UpSampling(keras.layers.Layer):
    def __init__(self, in_channel, out_channel, stride=2, **kwargs):
        super(UpSampling, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.list = LIST(in_channel, out_channel)
        self.stride = stride
        #         self.trans_conv = keras.layers.Conv2DTranspose(in_channel, kernel_size=(1,1), strides=stride, padding='same')
        self.up = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "in_channel": self.in_channel,
                "out_channel": self.out_channel,
                "stride": self.stride,
            }
        )
        return config

    def call(self, inputs, concat=None):
        #         shape = inputs.shape
        #         x = tf.image.resize(
        #             inputs,
        #             [shape[1] * self.stride, shape[2] * self.stride]
        #         )
        x = self.up(inputs)
        #         x = self.trans_conv(inputs)
        # print(inputs.shape, concat.shape, x.shape)
        if concat is not None:
            # print("Concatenating")
            x = keras.layers.Concatenate(axis=3)([concat, x])

        x = self.list(x)
        return x


class DownSampling(keras.layers.Layer):
    def __init__(self, in_channel, out_channel, stride=2, **kwargs):
        super(DownSampling, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.list = LIST(in_channel, out_channel)
        self.stride = stride
        self.pool = keras.layers.MaxPooling2D(pool_size=stride)

    def get_config(self):
        config = super(DownSampling, self).get_config()
        config.update(
            {
                "in_channel": self.in_channel,
                "out_channel": self.out_channel,
                "stride": self.stride,
            }
        )
        return config

    def call(self, inputs):
        #         shape = inputs.shape
        #         x = tf.image.resize(
        #             inputs,
        #             [shape[1] // self.stride, shape[2] // self.stride]
        #         )
        x = self.pool(inputs)
        x = self.list(x)
        return x


class Scene_Understanding(keras.layers.Layer):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(Scene_Understanding, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.global_pool = keras.layers.GlobalAveragePooling2D()
        self.dense = keras.layers.Dense(out_channel // 2)

        self.p_transform = keras.layers.Conv2D(
            out_channel // 2, kernel_size=(1, 1), padding="same"
        )

        self.aspp1 = LIST(in_channel, out_channel // 4)
        self.aspp2 = GSAT(in_channel, out_channel // 4, dilation_factor=(1, 2))
        self.aspp3 = GSAT(in_channel, out_channel // 4, dilation_factor=(1, 4))
        self.aspp4 = GSAT(in_channel, out_channel // 4, dilation_factor=(2, 1))

        self.final = keras.layers.Conv2D(
            out_channel * 2, kernel_size=(1, 1), padding="same"
        )

    def get_config(self):
        config = super(Scene_Understanding, self).get_config()
        config.update(
            {
                "in_channel": self.in_channel,
                "out_channel": self.out_channel,
            }
        )
        return config

    def call(self, inputs):
        u1 = self.global_pool(inputs)
        u1 = self.dense(u1)

        u1 = keras.layers.Reshape((1, 1, u1.shape[1]))(u1)
        u1 = keras.backend.tile(u1, [1, inputs.shape[1], inputs.shape[2], u1.shape[1]])
        #         print(u1.shape)

        #         u1 = tf.reshape(u1, [tf.shape(u1)[0], 1, 1, u1.shape[1]])
        #         u1 = tf.tile(u1, [tf.shape(u1)[0], inputs.shape[1],inputs.shape[2], u1.shape[1]])

        u2 = self.p_transform(inputs)

        u3 = self.aspp1(inputs)
        u4 = self.aspp2(inputs)
        u5 = self.aspp3(inputs)
        u6 = self.aspp4(inputs)

        #         print(u1.shape, u2.shape, u3.shape, u4.shape, u5.shape, u6.shape)

        cat = keras.layers.Concatenate(axis=3)([u1, u2, u3, u4, u5, u6])

        return self.final(cat)
