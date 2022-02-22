import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import loader
import data
from models import efficientnet
import utils

#Set seed value
seed_value = 43
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

#Hyper params
split = 0.8
height, width = 128, 256

PATH="/data3/awong/pano/M3D_low/"

train = loader.generate_dataframe("./splits/M3D_v1_train.yaml", PATH)
test = loader.generate_dataframe("./splits/M3D_v1_test.yaml", PATH)
validation = loader.generate_dataframe("./splits/M3D_v1_val.yaml", PATH)
print(len(train), len(validation))

train_generator = data.DataGenerator(train, batch_size=4, shuffle=True, dim=(512,256))
val_generator = data.DataGenerator(validation, batch_size=4, shuffle=False, dim=(512,256))
test_generator = data.DataGenerator(test, batch_size=16, shuffle=False, dim=(512,256))
print(len(train_generator), len(val_generator), len(test_generator))

images,depths = next(iter(val_generator))
print(images.shape, depths.shape)

model = efficientnet.EfficientUNet()
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              patience=3,
                                              restore_best_weights=True)

callbacks = [tf.keras.callbacks.LearningRateScheduler(utils.polynomial_decay, verbose=1), utils.checkpoint, early_stop]

model.compile(optimizer=utils.opt, loss=utils.loss_function, metrics=[utils.accuracy_function])
history = model.fit(train_generator, validation_data=val_generator, epochs=30,callbacks=callbacks)

model.evaluate(test_generator)
model.save('unet-efficient-pano1.h5')
