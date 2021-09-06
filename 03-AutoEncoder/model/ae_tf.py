import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Flatten

encoding_dim = 64

input_img = keras.Input(shape=(784,))

encoder = Dense(512, activation='relu')(input_img)
encoder = Dense(256, activation='relu')(encoder)
encoder = Dense(128, activatiob='relu')(encoder)
encoder = Dense(encoding_dim, activation='relu')(encoder)

decoder = Dense(128, activation='relu')(encoder)
decoder = Dense(256, activation='relu')(decoder)
decoder = Dense(512, activation='relu')(decoder)
decoder = Dense(784, activation='relu')(decoder)

ae = keras.Model(input_img, decoder)
