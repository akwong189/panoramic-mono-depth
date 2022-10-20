import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

IMG_SIZE = (1024, 768)
MIN_VAL = 0.5

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
    path="/data3/awong/diode/",
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
        # print(image.shape)
        random_height, random_width = rand_shape
        to_height = random_height + shape[0]
        to_width = random_width + shape[1]
        image = image[random_height:to_height, random_width:to_width, :]

    if type(image) is np.ndarray:
        return image
    return image.numpy()


def load_depth_image(depth_map, mask, **kwargs):
    depth_map = np.load(depth_map).squeeze()
    mask = np.load(mask)

    mask = mask > 0

    max_depth = min(300, np.percentile(depth_map, 99))

    depth_map = np.clip(depth_map, MIN_VAL, max_depth)
    depth_map = np.nan_to_num(depth_map, nan=0.5, posinf=max_depth, neginf=0.5)
    depth_map = np.log(depth_map + 1e-7, where=mask)

    depth_map = np.ma.masked_where(~mask, depth_map)

    depth_map[depth_map < 0.5] = 0.5
    # print(depth_map.min(), depth_map.max(), np.any(np.isnan(depth_map)))
    depth_map = np.clip(depth_map > 0, MIN_VAL, np.log(max_depth, where=max_depth > 0))
    #     depth_map = cv2.resize(depth_map, IMG_SIZE)
    depth_map = np.expand_dims(depth_map, axis=2)
    depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)
    return depth_map


def load_depth(
    depth_map,
    mask,
    path="/data3/awong/diode/",
    flip=False,
    shape=(256, 640),
    rand_shape=(0, 0),
    **kwargs
):
    img = load_depth_image(path + depth_map, path + mask)
    return preprocess_depth(img, flip, shape, rand_shape)


def generate_dataframe(filename):
    df = pd.read_csv(filename, index_col=0)
    return df.sample(frac=1).reset_index(drop=True)
