import os
import argparse

from config import TrainConfig

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import tensorflow as tf
import numpy as np
import random
import loader
import nyu_loader as nyl
import kitti_loader as kloader
import diode_loader as dloader
import data
import diode as diode_generator
from models import efficientnet, mobilenet, optimizedmobilenet, vgg
import utils
import tensorflow_datasets as tfds

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

DATASET = "diode"

if DATASET in ("pano", "kitti", "diode"):
    wrap = 64
    shape = (256, 512 + (wrap * 2), 3)
    buffer_size = 1000
elif DATASET == "nyu":
    batch_size = 8
    val_batch_size = 8
    buffer_size = 16
    shape = (480, 640, 3)
    steps_per_epoch= 47584//batch_size
    validation_steps=654//val_batch_size
    
# Set seed value
seed_value = 43
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

PATH = "/data3/awong/pano/M3D_low/"

def pano3d():
    train = loader.generate_dataframe("./splits/M3D_v1_train.yaml", PATH)
    test = loader.generate_dataframe("./splits/M3D_v1_test.yaml", PATH)
    validation = loader.generate_dataframe("./splits/M3D_v1_val.yaml", PATH)
    print(len(train["images"]), len(validation["images"]))

    train_generator = data.DataGenerator(train, batch_size=8, shuffle=True, wrap=wrap)
    val_generator = data.DataGenerator(validation, batch_size=8, shuffle=False, wrap=wrap)
    test_generator = data.DataGenerator(test, batch_size=16, shuffle=False, wrap=wrap)
    print(len(train_generator), len(val_generator), len(test_generator))
    
    images, depths = next(iter(train_generator))
    print(images.shape, depths.shape)
    
    return train_generator, val_generator, test_generator

def nyudepth():
    train_df, val_df = tfds.load('nyu_depth_v2', data_dir="/data3/awong/nyu", download=False, split=["train", "validation"])
    
    train = train_df.map(data.prepare_nyu, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validation = val_df.map(data.prepare_nyu)

    train_generator = train.cache().shuffle(buffer_size).batch(batch_size).repeat()
    train_generator = train_generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_generator = validation.batch(val_batch_size)
    
    dp = next(iter(train_generator))
    print(dp)
    
    return train_generator, val_generator, val_generator

def nyu_labeled():
    train = nyl.generate_nyu_dataframe("/data3/awong/data/nyu2_train.csv")
    test = nyl.generate_nyu_dataframe("/data3/awong/data/nyu2_test.csv")

    train_generator = data.DataGenerator(train, batch_size=16, shuffle=True, is_nyu=True)
    test_generator = data.DataGenerator(test, batch_size=32, shuffle=False, is_nyu=True)
    print(len(train_generator), len(test_generator))

    images, depths = next(iter(train_generator))
    print(images.shape, depths.shape)

    images, depths = next(iter(test_generator))
    print(images.shape, depths.shape)

    return train_generator, test_generator, test_generator


def kitti():
    train = kloader.generate_dataframe("./splits/kitti_train.csv", "train")
    test = kloader.generate_dataframe("./splits/kitti_test.csv", "val")

    train_generator = data.DataGenerator(train, batch_size=32, shuffle=True, datatype="kitti")
    test_generator = data.DataGenerator(test, batch_size=32, shuffle=False, datatype="kitti")

    print(len(train_generator), len(test_generator))
    
    images, depths = next(iter(train_generator))
    print("train", images.shape, depths.shape)

    images, depths = next(iter(test_generator))
    print("test", images.shape, depths.shape)

    return train_generator, test_generator, test_generator

def diode():
    train = dloader.generate_dataframe("./splits/diode_train.csv")
    test = dloader.generate_dataframe("./splits/diode_val.csv")

    train_generator = diode_generator.DataGenerator(train, batch_size=16, shuffle=True)
    test_generator = diode_generator.DataGenerator(test, batch_size=32, shuffle=False)

    print(len(train_generator), len(test_generator))
    
    images, depths = next(iter(train_generator))
    print("train", images.shape, depths.shape)

    images, depths = next(iter(test_generator))
    print("test", images.shape, depths.shape)

    return train_generator, test_generator, test_generator

# TEST CODE
# import glob
# d = {"images": glob.glob("/data3/awong/pano/M3D_test/image/*"), "depth": glob.glob("/data3/awong/pano/M3D_test/depth/*")}
# print(d)
# train_generator = data.DataGenerator(d, batch_size=8)
# val_generator = data.DataGenerator(d, shuffle=False)
# TEST CODE

if DATASET == "pano":
    train_generator, val_generator, test_generator = pano3d()
elif DATASET == "nyu":
    train_generator, val_generator, test_generator = nyu_labeled()
elif DATASET == "kitti":
    train_generator, val_generator, test_generator = kitti()
elif DATASET == "diode":
    train_generator, val_generator, test_generator = diode()
    
# model = efficientnet.EfficientUNet()
# model = mobilenet.MobileNet(shape)
model = optimizedmobilenet.OptimizedUNet_Scene(shape)
# model = vgg.VGG(shape)
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", mode="min", patience=10, restore_best_weights=True
)

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(utils.polynomial_decay, verbose=1),
    # early_stop,
]

model.compile(
    optimizer=utils.opt, loss=utils.loss_function #, metrics=[utils.accuracy_function]
)

history = model.fit(
    train_generator, validation_data=val_generator, epochs=60 #, callbacks=callbacks
)

model.evaluate(test_generator)
model.save("unet-optimized-diode7.h5")
"""
def check_file_exist(arg):
    if os.path.exists(f"./results/{arg}"):
        parser.error(f"The file {arg} already exists, please choose a different file name")
    elif ".h5" not in arg:
        parser.error(f"File is not an .h5 extension")
    else:
        return arg

parser = argparse.ArgumentParser(description="Monocular Panoramic NN Thesis Code")
parser.add_argument('-d', '--dataset', help="select dataset to train", default="pano", required=True, choices=["pano", "kitti", "diode", "nyu"])
parser.add_argument('-m', '--model', help="model to train dataset on", default="optimized", required=True, choices=["efficient", "mobile", "opt", "scene", "vgg"])
parser.add_argument('-o', '--output', help="output .h5 file name to results", required=True, type=check_file_exist)
parser.add_argument('-g', '--gpu', help="set gpu to train on", type=int, default=1, choices=range(0, 4))
parser.add_argument('-s', '--seed', help="set seed for training", type=int, default=43)
parser.add_argument('-p', '--path', help="path to dataset", default="/data3/awong/")
args = parser.parse_args()

if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = f"{args.gpu}"
    config = TrainConfig.gen_config(args)
    # train_generator, val_generator, test_generator = config.get_splits()
    model = config.get_model()

    model.summary()
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10, restore_best_weights=True
    )

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(utils.polynomial_decay, verbose=1),
        # early_stop,
    ]

    model.compile(
        optimizer=utils.opt, loss=utils.loss_function #, metrics=[utils.accuracy_function]
    )

    history = model.fit(
        train_generator, validation_data=val_generator, epochs=60 #, callbacks=callbacks
    )

    model.evaluate(test_generator)
    model.save("unet-optimized-diode7.h5") 
    """