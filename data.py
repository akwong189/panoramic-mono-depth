import tensorflow as tf
from tensorflow import keras
import numpy as np
import loader
import nyu_loader as nyl
import kitti_loader as kloader
import diode_loader as dloader

KITTI_SHAPE = (320, 1150)


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=True,
        wrap=0,
        datatype="pano",
        shape=(256, 640),
    ):
        self.images = dataset["images"]
        self.depth = dataset["depth"]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        self.wrap = wrap
        self.datatype = datatype
        self.shape = shape

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        flip = np.random.choice([True, False])
        shift = np.random.choice([i * 0.1 for i in range(10)])

        images = [self.images[k] for k in indexes]
        depth = [self.depth[k] for k in indexes]

        # for kitti
        max_height = 320 - self.shape[0]
        max_width = 1150 - self.shape[1]

        random_height = np.random.randint(0, max_height)
        random_width = np.random.randint(0, max_width)

        p_img = self.__preprocess_images(
            images, flip, shift, (random_height, random_width)
        )
        p_depth = self.__preprocess_depth(
            depth, flip, shift, (random_height, random_width)
        )

        return p_img, p_depth

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __preprocess_images(self, images, flip, shift, random):
        result = []
        for img in images:
            if self.datatype == "nyu":
                image = nyl.load_color(img, flip=flip)
            elif self.datatype == "pano":
                image = loader.load_color(img, flip=flip, shift=shift, wrap=self.wrap)
            else:
                image = kloader.load_color(img, flip=flip, rand_shape=random)

            scaled_img = (image - image.min()) / (image.max() - image.min())
            result.append(scaled_img)
        return np.array(result)

    def __preprocess_depth(self, images, flip, shift, random):
        result = []
        for img in images:
            if self.datatype == "nyu":
                image = nyl.load_depth(img, flip=flip)
            elif self.datatype == "pano":
                image = loader.load_depth(img, flip=flip, shift=shift, wrap=self.wrap)
            else:
                image = kloader.load_depth(img, flip=flip, rand_shape=random)
            scaled_img = (image - image.min()) / (image.max() - image.min())
            result.append(scaled_img)
        return np.array(result)


def prepare_nyu(datapoint):
    image = datapoint["image"]
    depth = datapoint["depth"]

    img_max = tf.math.reduce_max(image)
    img_min = tf.math.reduce_min(image)
    scaled_img = (image - img_min) / (img_max - img_min)

    depth = tf.reshape(depth, (480, 640, 1))
    depth_max = tf.math.reduce_max(depth)
    depth_min = tf.math.reduce_min(depth)
    scaled_depth = (depth - depth_min) / (depth_max - depth_min)

    return scaled_img, scaled_depth


def prepare_validation_data(datapoint):
    image = datapoint["image"]
    depth = datapoint["depth"]

    return image, depth
