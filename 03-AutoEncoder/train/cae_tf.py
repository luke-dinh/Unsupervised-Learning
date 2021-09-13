import keras
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, MaxPool2D
from keras.datasets import cifar100
import numpy as np

# Define model

input_image = keras.Input(shape=(32,32,3))

# Encoder
x = Conv2D(16, (3,3))(input_image)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(16, (3,3))(x)
x = LeakyReLU(alpha=0.1)(x)
encode = MaxPool2D(pool_size=(2,2))(x)

# Decoder
x = Conv2DTranspose(16, (3,3))(encode)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2DTranspose(16, (3,3))(x)
x = LeakyReLU(alpha=0.1)(x)
decode = Conv2D(1, (3,3), activation='sigmoid')(x)

cae = keras.Model(input_image, decode)
cae.compile(optimizer='adam', loss='binary_crossentropy')


(x_train, _), (x_test, _) = cifar100.load_data()

x_train = x_train.astype('float32')/255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.astype('float32')/255.0
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))