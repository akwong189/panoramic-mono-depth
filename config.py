from abc import abstractmethod
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import loader
import nyu_loader as nyl
import kitti_loader as kloader
import diode_loader as dloader
import data
import diode as diode_generator
import tensorflow_datasets as tfds

from models import efficientnet, mobilenet, optimizedmobilenet, vgg
from utils import loss_function, accuracy_function, DownSampling
from models.TCSVT import DownSampling, UpSampling, Scene_Understanding

NYU_TFDS_LOAD = False
DATASETS = ["pano", "kitti", "diode", "nyu"]


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = f"{seed}"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TrainConfig:
    def __init__(self, dataset, model, output, path, load):
        self.dataset = dataset
        self.model = model
        self.output = output
        self.path = path
        self.load = load
        self.wrap = 64
        self.shape = (256, 512 + (self.wrap * 2), 3)

    def __str__(self):
        return f"{self.dataset}, {self.model}, {self.output}, {self.path}"

    @abstractmethod
    def get_splits(self):
        raise NotImplemented("Calling from parent class and not child configuration")

    def get_model(self):
        if self.load:
            custom_func = {
                "loss_function": loss_function,
                "accuracy_function": accuracy_function,
                "DownSampling": DownSampling,
                "UpSampling": UpSampling,
                "Scene_Understanding": Scene_Understanding,
            }
            return keras.models.load_model(
                "./networks/unet-optimized-diode7.h5", custom_objects=custom_func
            )
        if self.model == "efficient":
            return efficientnet.EfficientUNet()
        elif self.model == "mobile":
            return mobilenet.MobileNet(self.shape)
        elif self.model == "opt":
            return optimizedmobilenet.OptimizedUNet(self.shape)
        elif self.model == "scene":
            return optimizedmobilenet.OptimizedUNet_Scene(self.shape)
        elif self.model == "vgg":
            return vgg.VGG(self.shape)

    @staticmethod
    def gen_config(args):
        dataset = args.dataset
        model = args.model
        out_file = args.output
        seed = args.seed
        data_path = args.path
        load = args.load

        set_seed(seed)

        if dataset == "kitti":
            return KittiConfig(dataset, model, out_file, data_path, load)
        elif dataset == "pano":
            return PanoConfig(dataset, model, out_file, data_path, load)
        elif dataset == "nyu":
            return NYUConfig(dataset, model, out_file, data_path, load)
        else:
            return DiodeConfig(dataset, model, out_file, data_path, load)


class KittiConfig(TrainConfig):
    def __init__(self, dataset, model, output, path, load):
        super(KittiConfig, self).__init__(dataset, model, output, path, load)

        self.wrap = 64
        self.shape = (256, 512 + (self.wrap * 2), 3)
        self.buffer_size = 1000

    def get_splits(self):
        train = kloader.generate_dataframe("./splits/kitti_train.csv", "train")
        test = kloader.generate_dataframe("./splits/kitti_test.csv", "val")

        train_generator = data.DataGenerator(
            train, batch_size=32, shuffle=True, datatype="kitti"
        )
        test_generator = data.DataGenerator(
            test, batch_size=32, shuffle=False, datatype="kitti"
        )

        print(len(train_generator), len(test_generator))

        images, depths = next(iter(train_generator))
        print("train", images.shape, depths.shape)

        images, depths = next(iter(test_generator))
        print("test", images.shape, depths.shape)

        return train_generator, test_generator, test_generator


class NYUConfig(TrainConfig):
    def __init__(self, dataset, model, output, path, load):
        super(NYUConfig, self).__init__(dataset, model, output, path, load)

        self.batch_size = 8
        self.val_batch_size = 8
        self.buffer_size = 16
        self.shape = (480, 640, 3)
        self.steps_per_epoch = 47584 // self.batch_size
        self.validation_steps = 654 // self.val_batch_size

    def nyu_labeled(self):
        train = nyl.generate_nyu_dataframe("/data3/awong/data/nyu2_train.csv")
        test = nyl.generate_nyu_dataframe("/data3/awong/data/nyu2_test.csv")

        train_generator = data.DataGenerator(
            train, batch_size=16, shuffle=True, is_nyu=True
        )
        test_generator = data.DataGenerator(
            test, batch_size=32, shuffle=False, is_nyu=True
        )
        print(len(train_generator), len(test_generator))

        images, depths = next(iter(train_generator))
        print(images.shape, depths.shape)

        images, depths = next(iter(test_generator))
        print(images.shape, depths.shape)

        return train_generator, test_generator, test_generator

    def nyudepth(self):
        train_df, val_df = tfds.load(
            "nyu_depth_v2",
            data_dir="/data3/awong/nyu",
            download=False,
            split=["train", "validation"],
        )

        train = train_df.map(
            data.prepare_nyu, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        validation = val_df.map(data.prepare_nyu)

        train_generator = (
            train.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        )
        train_generator = train_generator.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        val_generator = validation.batch(self.val_batch_size)

        dp = next(iter(train_generator))
        print(dp)

        return train_generator, val_generator, val_generator

    def get_splits(self):
        if NYU_TFDS_LOAD:
            return self.nyudepth()
        return self.nyu_labeled()


class DiodeConfig(TrainConfig):
    def __init__(self, dataset, model, output, path, load):
        super(DiodeConfig, self).__init__(dataset, model, output, path, load)

        self.wrap = 64
        self.shape = (256, 512 + (self.wrap * 2), 3)
        self.buffer_size = 1000

    def get_splits(self):
        train = dloader.generate_dataframe("./splits/diode_train.csv")
        test = dloader.generate_dataframe("./splits/diode_val.csv")

        train_generator = diode_generator.DataGenerator(
            train, batch_size=16, shuffle=True
        )
        test_generator = diode_generator.DataGenerator(
            test, batch_size=32, shuffle=False
        )

        print(len(train_generator), len(test_generator))

        images, depths = next(iter(train_generator))
        print("train", images.shape, depths.shape)

        images, depths = next(iter(test_generator))
        print("test", images.shape, depths.shape)

        return train_generator, test_generator, test_generator


class PanoConfig(TrainConfig):
    def __init__(self, dataset, model, output, path, load):
        super(PanoConfig, self).__init__(dataset, model, output, path, load)

        self.wrap = 64
        self.shape = (256, 512 + (self.wrap * 2), 3)
        self.buffer_size = 1000

    def get_splits(self):
        pano_path = self.path + "pano/M3D_low/"

        train = loader.generate_dataframe("./splits/M3D_v1_train.yaml", pano_path)
        test = loader.generate_dataframe("./splits/M3D_v1_test.yaml", pano_path)
        validation = loader.generate_dataframe("./splits/M3D_v1_val.yaml", pano_path)
        print(len(train["images"]), len(validation["images"]))

        train_generator = data.DataGenerator(
            train, batch_size=8, shuffle=True, wrap=self.wrap
        )
        val_generator = data.DataGenerator(
            validation, batch_size=8, shuffle=False, wrap=self.wrap
        )
        test_generator = data.DataGenerator(
            test, batch_size=16, shuffle=False, wrap=self.wrap
        )
        print(len(train_generator), len(val_generator), len(test_generator))

        images, depths = next(iter(train_generator))
        print(images.shape, depths.shape)

        return train_generator, val_generator, test_generator
