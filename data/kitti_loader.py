import tensorflow as tf
import numpy as np
import pandas as pd


# loads kitit rgb images
def preprocess_color(
    filename, image, flip=False, shape=None, rand_shape=None, **kwargs
):
    if flip:
        image = tf.image.flip_left_right(image)

    if shape:
        orig_shape = image.shape
        random_height, random_width = rand_shape
        to_height = random_height + shape[0]
        to_width = random_width + shape[1]
        image = image[random_height:to_height, random_width:to_width, :]
        # assert image.shape[0] == shape[0] and image.shape[1] == shape[1], f"Train - original size {filename}: {orig_shape} | image size: {image.shape} | expected shape: {shape} | random location: {rand_shape} - ({to_height}, {to_width})"

    if type(image) is np.ndarray:
        return image
    return image.numpy()


def load_color_image(filename, **kwargs):
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    return image


def load_color(
    filename,
    path="/data3/awong/kitti_raw/",
    flip=False,
    shape=(256, 640),
    rand_shape=(0, 0),
    **kwargs
):
    img = load_color_image(path + filename)
    return preprocess_color(filename, img, flip, shape, rand_shape)


# loads kitti depth images
def preprocess_depth(image, flip=False, shape=None, rand_shape=(0, 0), **kwargs):
    if flip:
        image = tf.image.flip_left_right(image)

    if shape:
        random_height, random_width = rand_shape
        to_height = random_height + shape[0]
        to_width = random_width + shape[1]
        image = image[random_height:to_height, random_width:to_width, :]

    if type(image) is np.ndarray:
        return image
    return image.numpy()


def load_depth_image(filename, **kwargs):
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image = tf.cast(image, tf.float32)
    return image


def load_depth(
    filename,
    path="/data3/awong/kitti/",
    flip=False,
    shape=(256, 640),
    rand_shape=(0, 0),
    **kwargs
):
    img = load_depth_image(path + filename)
    return preprocess_depth(img, flip, shape, rand_shape)


def generate_dataframe(filename, type):
    df = pd.read_csv(filename, index_col=0)
    df["depth"] = type + "/" + df["depth"]
    return df.sample(frac=1).reset_index(drop=True)
