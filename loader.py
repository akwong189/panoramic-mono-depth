import types
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
from yaml import safe_load

def load_depth(filename, flip, size=(256,128), max_depth=8.0, **kwargs):
    if 'position' in kwargs:
        filename = filename.replace('center', kwargs['position'])
    depth_filename = filename.replace('emission', 'depth').replace('.png', '.exr')
    if 'filmic' in depth_filename.split(os.sep)[-3]:
        depth_filename = depth_filename.replace('_filmic', '')

    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    # depth = cv2.resize(depth, size)

    if flip:
        depth = cv2.flip(depth, 1)

    depth = np.expand_dims(depth, axis=-1)

    #NOTE: add a micro meter to allow for thresholding to extact the valid mask
    depth[depth > max_depth] = max_depth + 1e-6

    return depth

def load_color(filename, flip, **kwargs):
    if 'position' in kwargs:
        filename = filename.replace('center', kwargs['position'])
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels=3, expand_animations = False)
    image = tf.cast(image, tf.float32)

    if flip:
        image = tf.image.flip_left_right(image)

    return image.numpy()

def generate_dataframe(filename: str, path: str="./") -> pd.DataFrame:
    with open(filename) as file:
        df = pd.json_normalize(safe_load(file.read()))
        img_paths = []
        depth_paths = []

        for cols in df.columns:
            temp = path + cols + "/" + pd.DataFrame(df[cols][0])
            img_paths += temp.values.tolist()
            depth_paths = temp.replace(".png", ".exr")

        img_paths = np.squeeze(img_paths)
        depth_paths = np.squeeze(img_paths)

        return {'images': img_paths, 'depth': depth_paths}
