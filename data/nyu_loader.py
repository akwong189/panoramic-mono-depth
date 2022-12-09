import tensorflow as tf
import numpy as np
import cv2
import pandas as pd


def preprocess_depth(depth, flip=False, swap=False, shift=0, wrap=0, **kwargs):
    if shift > 0:
        h, w = depth.shape

        shifted = int(w * shift)

        depth_left = depth[:, :shifted]
        depth_right = depth[:, shifted:]

        depth = np.hstack((depth_right, depth_left))

    if swap:
        h, w = depth.shape
        depth_left = depth[:, : w // 2]
        depth_right = depth[:, -w // 2 :]

        depth = np.hstack((depth_right, depth_left))

    if wrap > 0:
        temp = cv2.flip(depth, 1)
        depth_left = temp[:, :wrap]
        depth_right = temp[:, -wrap:]

        depth_left = cv2.flip(depth_left, 1)
        depth_right = cv2.flip(depth_right, 1)

        depth = np.hstack((depth, depth_right))
        depth = np.hstack((depth_left, depth))

    if flip:
        depth = cv2.flip(depth, 1)

    depth = np.expand_dims(depth, axis=-1)

    return depth


def load_depth_image(filename, size=(256, 128), **kwargs):
    depth = cv2.imread(filename, 0)
    return depth


def load_depth(filename, flip=False, swap=False, size=(256, 128), shift=0, wrap=0):
    depth = load_depth_image(filename, size)
    return preprocess_depth(depth, flip, swap, shift, wrap)


def preprocess_color(image, flip=False, swap=False, shift=0, wrap=0, **kwargs):
    if shift > 0:
        h, w, c = image.shape

        shifted = int(w * shift)

        image_left = image[:, :shifted]
        image_right = image[:, shifted:]

        image = np.hstack((image_right, image_left))

    if swap:
        h, w, c = image.numpy().shape
        image_left = image[:, : w // 2]
        image_right = image[:, -w // 2 :]

        image = np.hstack((image_right, image_left))

    if wrap > 0:
        temp = np.flip(image, axis=1)
        image_left = temp[:, :wrap]
        image_right = temp[:, -wrap:]

        image_left = np.flip(image_left, 1)
        image_right = np.flip(image_right, 1)

        image = np.hstack((image, image_right))
        image = np.hstack((image_left, image))

    if flip:
        image = tf.image.flip_left_right(image)

    if type(image) is np.ndarray:
        return image
    return image.numpy()


def load_color_image(filename, **kwargs):
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    return image


def load_color(filename, flip=False, swap=False, shift=0, wrap=0, **kwargs):
    img = load_color_image(filename)
    return preprocess_color(img, flip, swap, shift, wrap)


def generate_nyu_dataframe(filename: str, path: str = "/data3/awong"):
    df = pd.read_csv(filename, header=None)
    df = df[0]

    img_paths = []
    depth_paths = []

    for record in df:
        temp = path + "/" + record
        img_paths.append(temp)
        depth_paths.append(temp.replace("jpg", "png"))

    return {"images": img_paths, "depth": depth_paths}
