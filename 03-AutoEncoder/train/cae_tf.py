import keras
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, MaxPool2D
from keras.datasets import cifar100

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

