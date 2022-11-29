import tensorflow as tf
import numpy as np
from tensorflow import keras
import data.pano_loader as loader
from data import data
import onnxruntime
from utils import set_loss_function

# https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    # prevent divide by 0
    gt += 1e-7
    pred += 1e-7
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(np.square(np.subtract(gt,pred)).mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def calculate_errors(model, test_gen):
    absrel = 0
    sqrel = 0
    rmse = 0
    rmse_log = 0
    a1 = 0
    a2 = 0
    a3 = 0
    
    num = len(test_gen.images)
    
    for i in range(len(test_gen)):
        img, depths = test_gen[i]
        preds = model.predict(img)

        for p, d in zip(preds, depths):
            ab, sq, rm, rm_log, _a1, _a2, _a3 = compute_errors(d, p)
            absrel += ab
            sqrel += sq
            rmse += rm
            rmse_log += rm_log
            a1 += _a1
            a2 += _a2
            a3 += _a3
        
    print("Abs Rel: %0.2f | Sq Rel: %0.2f | RMSE: %0.2f | RMSE log: %0.2f | a1: %0.2f | a2: %0.2f | a3: %0.2f" % (absrel / num, sqrel / num, rmse / num, rmse_log / num, a1 / num, a2 / num, a3 / num))
    return absrel, sqrel, rmse, rmse_log, a1, a2, a3

def calculate_errors_onnx(sess, test_gen):
    absrel = 0
    sqrel = 0
    rmse = 0
    rmse_log = 0
    a1 = 0
    a2 = 0
    a3 = 0
    
    num = len(test_gen.images)
    inputs = sess.get_inputs()[0].name
    output = sess.get_outputs()[0].name
    
    for i in range(len(test_gen)):
        img, depths = test_gen[i]
        preds = sess.run([output], {inputs: img})[0]

        for p, d in zip(preds, depths):
            ab, sq, rm, rm_log, _a1, _a2, _a3 = compute_errors(d, p)
            absrel += ab
            sqrel += sq
            rmse += rm
            rmse_log += rm_log
            a1 += _a1
            a2 += _a2
            a3 += _a3
            
        
    print("Abs Rel: %0.2f | Sq Rel: %0.2f | RMSE: %0.2f | RMSE log: %0.2f | a1: %0.2f | a2: %0.2f | a3: %0.2f" % (absrel / num, sqrel / num, rmse / num, rmse_log / num, a1 / num, a2 / num, a3 / num))
    return absrel, sqrel, rmse, rmse_log, a1, a2, a3


def pano_test_generator(pano_path="/data3/awong/pano/M3D_low/", splits="./splits/M3D_v1_test.yaml"):
    test = loader.generate_dataframe(splits, pano_path)
    test_generator = data.DataGenerator(
                test, batch_size=32, shuffle=False, wrap=64
    )
    return test_generator

def run_metrics_h5(model_path, test_generator):
    keras.backend.clear_session()
    loss = set_loss_function(["ssim"])
    custom_func = {"new_new_loss": loss, "loss_function": loss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_func)
    calculate_errors(model, test_generator)

def run_metrics_onnx(model_path, test_generator, cpu=False):
    providers = ["TensorrtExecutionProvider"]
    if cpu:
        providers = ["CPUExecutionProvider"]
    sess = onnxruntime.InferenceSession(model_path, providers=providers)
    calculate_errors_onnx(sess, test_generator)

def run_metrics(model_path, cpu=False):
    test_generator = pano_test_generator()
    if ".onnx" in model_path:
        run_metrics_onnx(model_path, test_generator, cpu)
    else:
        run_metrics_h5(model_path, test_generator)