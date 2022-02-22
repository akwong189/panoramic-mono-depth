# Model
import tensorflow as tf
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, concatenate, Concatenate, Input
from tensorflow.keras import Model

def EfficientUNet():
    def upsampling(input_tensor, n_filters, concat_layer, concat=True):
        '''
        Block of Decoder
        '''
        # Bilinear 2x upsampling layer
        x = UpSampling2D(size=(2,2), interpolation='bilinear')(input_tensor)
        # concatenation with encoder block 
        if concat:
            x = concatenate([x, concat_layer])
        # decreasing the depth filters by half
        x = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same')(x)
        x = BatchNormalization()(x)
        return x

    base_model = efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(256, 512, 3))
    for layer in base_model.layers:
        layer.trainable = True

    inputs = base_model.input
    x = base_model.output

    names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation"][::-1]
    bneck = Conv2D(filters=512, kernel_size=(1,1), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(bneck)
    x = upsampling(bneck, 256, base_model.get_layer(names[0]).output)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 128, base_model.get_layer(names[1]).output)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 64, base_model.get_layer(names[2]).output)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 32, base_model.get_layer(names[3]).output)
    x = LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 16, base_model.layers[0].output)
    x = Conv2D(filters=1, activation='sigmoid', kernel_size=(3,3), padding='same')(x)

    model = Model(inputs=inputs, outputs=x)
    return model