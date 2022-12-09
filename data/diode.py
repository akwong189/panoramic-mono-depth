import tensorflow as tf
import data.diode_loader as dloader
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        data,
        batch_size=32,
        dim=(768, 1024),
        out_shape=(256, 640),
        n_channels=3,
        shuffle=True,
    ):
        """
        Initialization
        """
        self.images = data["images"]
        self.depth = data["depth"]
        self.mask = data["depth_mask"]
        self.indexes = np.arange(len(self.images))

        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.out_shape = out_shape
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        flip = np.random.choice([True, False])

        # randomly take a shape location
        max_height = self.dim[0] - self.out_shape[0]
        max_width = self.dim[1] - self.out_shape[1]

        random_height = np.random.randint(0, max_height)
        random_width = np.random.randint(0, max_width)

        # Find list of IDs
        images = [self.images[k] for k in indexes]
        depth = [self.depth[k] for k in indexes]
        mask = [self.mask[k] for k in indexes]

        p_img = self._preprocess_images(images, flip, (random_height, random_width))
        p_depth = self._preprocess_depth(
            depth, mask, flip, (random_height, random_width)
        )
        assert not (np.any(np.isnan(p_depth)) or np.any(np.isinf(p_depth)))
        assert not (np.any(np.isnan(p_img)) or np.any(np.isinf(p_img)))

        return p_img, p_depth

    def on_epoch_end(self):
        self.index = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.index)

    def _preprocess_images(self, images, flip, random_shape):
        result = []
        for img in images:
            image = dloader.load_color(img, flip=flip, rand_shape=random_shape)
            assert not np.any(np.isnan(image))
            image = image / 255.0
            # image = (image - image.min()) / (image.max() - image.min())
            result.append(image)
        return np.array(result)

    def _preprocess_depth(self, depth, mask, flip, random_shape):
        result = []
        for i in range(len(depth)):
            img = dloader.load_depth(
                depth[i], mask[i], flip=flip, rand_shape=random_shape
            )
            assert not np.any(np.isnan(img))
            assert not np.any(np.isinf(img))
            # img = (img - 0.5) / (300 - 0.5)
            img = (img - 0.5) / (np.log(300) - 0.5)
            result.append(img)
        return np.array(result)

    def load(self, image_path, depth_map, mask, flip, random_shape):
        """Load input and target image."""

        image = dloader.load_color(image_path, flip=flip, rand_shape=random_shape)
        image = (image - image.min()) / (image.max() - image.min())
        depth = dloader.load_depth(depth_map, mask, flip=flip, rand_shape=random_shape)
        if depth.max() != depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())

        return image, depth
