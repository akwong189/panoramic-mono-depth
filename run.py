import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import tensorflow as tf
import numpy as np
import random
import loader
import data
from models import efficientnet, mobilenet
import utils

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

#Set seed value
seed_value = 43
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

PATH="/data3/awong/pano/M3D_low/"

train = loader.generate_dataframe("./splits/M3D_v1_train.yaml", PATH)
test = loader.generate_dataframe("./splits/M3D_v1_test.yaml", PATH)
validation = loader.generate_dataframe("./splits/M3D_v1_val.yaml", PATH)
print(len(train['images']), len(validation['images']))

train_generator = data.DataGenerator(train, batch_size=8, shuffle=True)
val_generator = data.DataGenerator(validation, batch_size=8, shuffle=False)
test_generator = data.DataGenerator(test, batch_size=16, shuffle=False)
print(len(train_generator), len(val_generator), len(test_generator))


# TEST CODE
# import glob
# d = {"images": glob.glob("/data3/awong/pano/M3D_test/image/*"), "depth": glob.glob("/data3/awong/pano/M3D_test/depth/*")}
# print(d)
# train_generator = data.DataGenerator(d, batch_size=8)
# val_generator = data.DataGenerator(d, shuffle=False)
# TEST CODE

images, depths = next(iter(train_generator))
print(images.shape, depths.shape)

model = efficientnet.EfficientUNet()
# model = mobilenet.MobileNet()
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              patience=3,
                                              restore_best_weights=True)

callbacks = [tf.keras.callbacks.LearningRateScheduler(utils.polynomial_decay, verbose=1), utils.checkpoint, early_stop]

model.compile(optimizer=utils.opt, loss=utils.loss_function, metrics=[utils.accuracy_function])
history = model.fit(train_generator, validation_data=val_generator, epochs=30,callbacks=callbacks)

model.evaluate(test_generator)
model.save('unet-efficientnet-pano2.h5')
