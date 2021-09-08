import keras
from keras.datasets import mnist
from keras.layers import Dense
import numpy as np

encoding_dim = 64

input_img = keras.Input(shape=(784,))

encoder = Dense(512, activation='relu')(input_img)
encoder = Dense(256, activation='relu')(encoder)
encoder = Dense(128, activation='relu')(encoder)
encoder = Dense(encoding_dim, activation='relu')(encoder)

decoder = Dense(128, activation='relu')(encoder)
decoder = Dense(256, activation='relu')(decoder)
decoder = Dense(512, activation='relu')(decoder)
decoder = Dense(784, activation='relu')(decoder)

ae = keras.Model(input_img, decoder)

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.astype('float32')/255.0
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

ae.compile(optimizer='adam', loss = 'binary_crossentropy')
ae.fit(x_train, x_train, epochs=50, batch_size=128, verbose=1, validation_data=(x_test, x_test), shuffle=True)
