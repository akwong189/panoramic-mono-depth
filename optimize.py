from tensorflow import keras
import tf2onnx
from models.TCSVT import DownSampling, UpSampling, Scene_Understanding
import onnxruntime
import os
from onnxruntime.quantization import CalibrationDataReader
import data.pano_loader as loader
from data import data
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

pano_path = "/data3/awong/pano/M3D_low/"
train = loader.generate_dataframe("./splits/M3D_v1_train.yaml", pano_path)
train_generator = data.DataGenerator(
            train, batch_size=32, shuffle=False, wrap=64
        )

class MobileNetDataReader(CalibrationDataReader):
    def __init__(self, model_path):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape
        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(train_generator)
        
        vals = next(self.enum_data, None)
        if vals:
            return {self.input_name: vals[0]}
        return None

    def rewind(self):
        self.enum_data = None

def optimize_model(model_name, loss_function):
    custom_func = {
        "loss_function": loss_function,
        "DownSampling": DownSampling,
        "UpSampling": UpSampling,
        "Scene_Understanding": Scene_Understanding
    }
    model = keras.models.load_model(model_name, custom_objects=custom_func)
    output_path = model_name[:-3] + ".onnx"

    model_proto, _ = tf2onnx.convert.from_keras(model, opset=13, output_path=output_path)
    return output_path


def quantize_model(model_name, loss_function):
    if ".onnx" in model_name:
        input_model_path = model_name
    else:
        input_model_path = optimize_model(model_name, loss_function)
    output_model_path = input_model_path[:-5] + "_quant.onnx"
    dr = MobileNetDataReader(input_model_path)

    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        optimize_model=False,
    )
    return output_model_path