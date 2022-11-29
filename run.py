import os
import argparse
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

LOSSES = ["ssim", "l1", "berhu", "sobel", "edges", "smooth"]

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
subparser = parser.add_subparsers(dest="cmd")

train = subparser.add_parser("train", help="Train model")
optimize = subparser.add_parser("optimize", help="Optimize .h5 model")
metrics = subparser.add_parser("metrics", help="Metrics calculations for model")

train.add_argument(
    "-d",
    "--dataset",
    help="select dataset to train",
    default="pano",
    required=True,
    choices=["pano", "kitti", "diode", "nyu"],
)
train.add_argument(
    "-m",
    "--model",
    help="model to train dataset on",
    default="mobile",
    choices=["efficient", "mobile", "opt", "scene", "vgg", "shuffle", "mobilev3"],
)
train.add_argument(
    "-o",
    "--output",
    help="output .h5 file name to results",
    required=True,
    type=check_file_not_exists,
)
train.add_argument(
    "-g", "--gpu", help="set gpu to train on", type=int, default=1, choices=range(0, 4)
)
train.add_argument("-s", "--seed", help="set seed for training", type=int, default=43)
train.add_argument("-p", "--path", help="path to dataset", default="/data3/awong/")
train.add_argument(
    "-l",
    "--load",
    help="load .h5 file for retraining and/or metric calculations",
    type=check_file_exists,
)
train.add_argument(
    "-e", "--epochs", type=int, default=40, help="set the number of epochs to train for"
)
train.add_argument(
    "-lr", "--rate", help="set learning rate", default=0.005, type=float
)
train.add_argument(
    "--cpu", help="Use CPU instead of GPU", action='store_true'
)
train.add_argument(
    "--summary", help="Display the model summary", action='store_true'
)
train.add_argument(
    "--loss",
    help="set loss function to use for training",
    nargs="+",
    choices=LOSSES,
    default=["ssim", "l1", "sobel"]
)
train.add_argument(
    "--verbose",
    help="set verbosity of training",
    type=int,
    metavar="[0-2]",
    choices=[0, 1, 2],
    default=2
)

optimize.add_argument(
    "-m", "--model", help=".h5 or .onnx model", type=check_file_not_exists, required=True
)
optimize.add_argument(
    "-q", "--quantize", help="quantize model"
)
optimize.add_argument(
    "-l", "--loss",
    help="set loss function to use for training",
    nargs="+",
    choices=LOSSES,
    default=["ssim", "l1", "sobel"]
)
optimize.add_argument(
    "--cpu", help="Use CPU instead of GPU", action='store_true'
)
optimize.add_argument(
    "-g", "--gpu", help="set gpu to train on", type=int, default=1, choices=range(0, 4)
)

metrics.add_argument(
    "-m", "--model", help=".h5 or .onnx model", type=check_file_not_exists, required=True
)
metrics.add_argument(
    "--cpu", help="Use CPU instead of GPU", action='store_true'
)
metrics.add_argument(
    "-g", "--gpu", help="set gpu to train on", type=int, default=1, choices=range(0, 4)
)

args = parser.parse_args()

if __name__ == "__main__":
    if args.cmd == "train":
        if args.model is None and args.load is None:
            print(f"Failed to find model to use!")
            exit(1)

        if not args.cpu:
            print(f"Setting the GPU to be used to GPU #{args.gpu}")
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

        from config import TrainConfig
        import tensorflow as tf
        import utils
        import pickle

        loss = utils.set_loss_function(args.loss)
        config = TrainConfig.gen_config(args)
        model = config.get_model(loss)

        tf.debugging.enable_check_numerics()
        if args.summary:
            model.summary()
            exit(0)
        print(f"Using loss(es) {args.loss} for training")
        tf.keras.backend.clear_session()
        train_generator, val_generator, test_generator = config.get_splits()

        # set training information
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.rate)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=7, restore_best_weights=True
        )

        callbacks = [
            # tf.keras.callbacks.LearningRateScheduler(utils.learning_decay, verbose=1),
        ]

        print(f"learning rate set to {args.rate}")
        model.compile(
            optimizer=optimizer,
            loss=loss,
        )

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=args.verbose
        )

        with open(f'./{args.output[:-3]}.history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        model.evaluate(test_generator)
        model.save(args.output)
        print(f"{args.output} is done")

    if args.cmd == "optimize":
        if args.model is None:
            print(f"Failed to find model to use!")
            exit(1)

        if not args.cpu:
            print(f"Setting the GPU to be used to GPU #{args.gpu}")
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

        import tensorflow as tf
        import utils
        from optimize import quantize_model, optimize_model

        loss = utils.set_loss_function(args.loss)

        if args.quantize:
            path = quantize_model(args.model, loss)
        else:
            path = optimize_model(args.model, loss)
        print(f"ONNX model {path} has been created")

    if args.cmd == "metrics":
        if args.model is None:
            print(f"Failed to find model to use!")
            exit(1)

        if not args.cpu:
            print(f"Setting the GPU to be used to GPU #{args.gpu}")
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"


        from metrics import run_metrics
        run_metrics(args.model, args.cpu)
        
