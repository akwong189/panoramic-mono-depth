import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, QuantFormat
import loader
import data

import tensorflow as tf

pano_path = "/data3/awong/pano/M3D_low/"
train = loader.generate_dataframe("./splits/M3D_v1_train.yaml", pano_path)
train_generator = data.DataGenerator(
            train, batch_size=32, shuffle=False, wrap=64
        )

class MobileNetDataReader(CalibrationDataReader):
    def __init__(self, model_path):
        self.enum_data = None

#         # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape
        self.input_name = session.get_inputs()[0].name
#         self.datasize = len(train.images)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(train_generator)

        vals = next(self.enum_data, None)
        if vals:
            return {self.input_name: vals[0]}
        return None

    def rewind(self):
        self.enum_data = None



input_model_path = 'temp.onnx'
output_model_path = 'quant_qdq.onnx'
dr = MobileNetDataReader(input_model_path)

# print(dr.get_next())

quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        optimize_model=False,
    )
