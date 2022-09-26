import os
import argparse
from config import TrainConfig
import tensorflow as tf
import utils


def check_file(arg):
    if os.path.exists(f"./results/{arg}"):
        return 1
    elif ".h5" not in arg:
        return -1
    else:
        return 0


def check_file_not_exists(arg):
    ret = check_file(arg)
    if ret == 1:
        parser.error(
            f"The file {arg} already exists, please choose a different file name"
        )
    elif ret == -1:
        parser.error(f"File {arg} is not an .h5 extension")
    else:
        return arg


def check_file_exists(arg):
    ret = check_file(arg)
    if ret == 1:
        return arg
    elif ret == -1:
        parser.error(f"File {arg} is not an .h5 extension")
    else:
        parser.error(f"File {arg} not found")


parser = argparse.ArgumentParser(description="Monocular Panoramic NN Thesis Code")
parser.add_argument(
    "-d",
    "--dataset",
    help="select dataset to train",
    default="pano",
    required=True,
    choices=["pano", "kitti", "diode", "nyu"],
)
parser.add_argument(
    "-m",
    "--model",
    help="model to train dataset on",
    default="optimized",
    required=True,
    choices=["efficient", "mobile", "opt", "scene", "vgg", "shuffle"],
)
parser.add_argument(
    "-o",
    "--output",
    help="output .h5 file name to results",
    required=True,
    type=check_file_not_exists,
)
parser.add_argument(
    "-g", "--gpu", help="set gpu to train on", type=int, default=1, choices=range(0, 4)
)
parser.add_argument("-s", "--seed", help="set seed for training", type=int, default=43)
parser.add_argument("-p", "--path", help="path to dataset", default="/data3/awong/")
parser.add_argument(
    "-l",
    "--load",
    help="load .h5 file for retraining and/or metric calculations",
    type=check_file_exists,
)
parser.add_argument(
    "-e", "--epochs", type=int, default=40, help="set the number of epochs to train for"
)
parser.add_argument(
    "-lr", "--rate", help="set learning rate", default=0.005, type=float
)
parser.add_argument(
    "--cpu", help="Use CPU instead of GPU", action='store_true'
)
parser.add_argument(
    "--summary", help="Display the model summary", action='store_true'
)
args = parser.parse_args()

if __name__ == "__main__":
    if not args.cpu:
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = f"{args.gpu}"
    config = TrainConfig.gen_config(args)
    train_generator, val_generator, test_generator = config.get_splits()
    model = config.get_model()

    if args.summary:
        model.summary()
        exit(0)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10, restore_best_weights=True
    )

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(utils.polynomial_decay, verbose=1),
        # early_stop,
    ]

    model.compile(
        optimizer=utils.opt,
        loss=utils.loss_function,  # , metrics=[utils.accuracy_function]
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,  # , callbacks=callbacks
    )

    model.evaluate(test_generator)
    model.save(args.output)
