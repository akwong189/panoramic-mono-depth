import tensorflow as tf
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
        """Number of mini-batches"""
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Images and ground truth, output should be an array of size equal to the batch_size"""
        img = None
        ground_truth = None

        return [img], [ground_truth]

    def on_epoch_end(self):
        """At end of an epoch, shuffle the dataset"""
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)