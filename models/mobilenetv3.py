import tensorflow as tf
from tensorflow import keras

def scene_understanding(x, out_channels):
    global_understanding = keras.layers.GlobalAveragePooling2D()(x)
    global_understanding = keras.layers.Dense(out_channels // 2)(global_understanding)
    global_understanding = keras.layers.Reshape((1, 1, global_understanding.shape[1]))(global_understanding)
    global_understanding = keras.backend.tile(global_understanding, [1, x.shape[1], x.shape[2], global_understanding.shape[1]])

    pixel_transform = keras.layers.Conv2D(out_channels // 2, kernel_size=1, padding="same")(x)

    aspp1 = keras.layers.Conv2D(out_channels // 4, kernel_size=3, padding="same")(x)
    aspp2 = keras.layers.Conv2D(out_channels // 4, kernel_size=3, dilation_rate=(1, 2), padding="same")(x)
    aspp3 = keras.layers.Conv2D(out_channels // 4, kernel_size=3, dilation_rate=(1, 4), padding="same")(x)
    aspp4 = keras.layers.Conv2D(out_channels // 4, kernel_size=3, dilation_rate=(2, 1), padding="same")(x)

    cat = keras.layers.Concatenate(axis=3)([global_understanding, pixel_transform, aspp1, aspp2, aspp3, aspp4])
    return keras.layers.Conv2D(out_channels, kernel_size=1, padding="same")(cat)

def skip_connection_understanding(x, output_channels):
    x = keras.layers.Conv2D(output_channels, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(output_channels, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(rate=0.1)(x)
    return x

def upsample_custom(x, out_channels, concat_layer=None):
    x = keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear")(x)
    x = keras.layers.Concatenate(axis=3)([concat_layer, x])
    x = keras.layers.Conv2D(out_channels, kernel_size=1, padding="same")(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Dropout(rate=0.1)(x)
    return x

def upsample(x, concat, out_channels, skip_process=True, alpha=0.2):
    if skip_process:
        concat = skip_connection_understanding(concat, out_channels // 4)
        concat = keras.layers.LeakyReLU(alpha=alpha)(concat) # leaky ReLU could be slower than using ReLU
    x = keras.layers.LeakyReLU(alpha=alpha)(x)
    x = keras.layers.Dropout(rate=0.1)(x)
    x = upsample_custom(x, out_channels, concat)
    return x

def MobileNetv3(shape=(256, 512, 3), w_scene=True, w_skip=True):
    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=True,
        weights="imagenet",
        input_shape=shape,
        minimalistic=True,
    )

    for layer in base_model.layers:
        layer.trainable = True

    inputs = base_model.input
    x = base_model.get_layer("re_lu_22").output # 8

    names = [
        "re_lu", # 128
        "re_lu_2", # 64
        "re_lu_6", # 32
        "re_lu_16" # 16
    ][::-1]

    # names = [
    #     "rescaling", # 256
    #     "re_lu", # 128
    #     "re_lu_3", # 64
    #     "re_lu_7", # 32
    #     "re_lu_22" # 16
    # ][::-1]

    if w_scene:
        x = keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(x)
        x = scene_understanding(x, 128)

    # x = upsample(x, 256, base_model.get_layer(names[0]).output)
    x = upsample(x, base_model.get_layer(names[0]).output, 128, skip_process=w_skip)
    x = upsample(x, base_model.get_layer(names[1]).output, 64, skip_process=w_skip)
    x = upsample(x, base_model.get_layer(names[2]).output, 32, skip_process=w_skip)
    x = upsample(x, base_model.get_layer(names[3]).output, 16, skip_process=w_skip)
    x = upsample(x, base_model.layers[0].output, 16, skip_process=w_skip)
    x = keras.layers.Conv2D(
        filters=1, activation="sigmoid", kernel_size=(3, 3), padding="same"
    )(x)

    model = keras.Model(inputs=inputs, outputs=x)
    return model
