import os
import argparse
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def check_file(arg):
    if os.path.exists(arg):
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
    if args.model is None and args.load is None:
        print(f"Failed to find model to use!")
        exit(1)

    if not args.cpu:
        print(f"Setting the GPU to be used to GPU #{args.gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    from config import TrainConfig
    import tensorflow as tf
    import tensorflow_addons as tfa
    import utils
    import pickle

    config = TrainConfig.gen_config(args)
    train_generator, val_generator, test_generator = config.get_splits()
    model = config.get_model()
    # optimizer = tfa.optimizers.AdamW(1e-5, learning_rate=args.rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.rate)

    tf.debugging.enable_check_numerics()
    if args.summary:
        model.summary()
        exit(0)
    tf.keras.backend.clear_session()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=7, restore_best_weights=True
    )

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(utils.learning_decay, verbose=1),
        # early_stop,
    ]

    print(f"learning rate set to {args.rate}")
    model.compile(
        optimizer=optimizer, # tf.keras.optimizers.Adam(learning_rate=args.rate),
        # loss=utils.new_loss_function,  
        # metrics=[utils.SSIM],
        loss=utils.new_new_loss,
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks
    )

    with open(f'./{args.output[:-3]}.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.evaluate(test_generator)
    model.save(args.output)
    print(f"{args.output} is done")
