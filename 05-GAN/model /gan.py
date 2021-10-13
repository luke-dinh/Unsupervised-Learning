from keras import optimizers
from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.optimizers import Adam, RMSprop

import numpy as np
import matplotlib.pyplot as plt
import random 
from tqdm import tqdm_notebook

# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')/255.0

x_test = x_test.reshape(10000, 784)
x_test = x_test.reshape(10000, 784)

# Model

z_dim = 100
loss = Adam(learning_rate=1e-3, beta_1=0.5)

# Generator
g = Sequential()
g.add(Dense(256, input_dim=z_dim, activation=LeakyReLU(alpha=0.1)))
g.add(Dense(512, activation=LeakyReLU(alpha=0.1)))
g.add(Dense(1024, activation=LeakyReLU(alpha=0.1)))

g.add(Dense(784, activation='sigmoid'))
g.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Discriminator
d = Sequential()
d.add(Dense(1024, input_dim=784, activation=LeakyReLU(alpha=0.1)))
d.add(Dense(521, activation=LeakyReLU(alpha=0.1)))
d.add(Dense(256, activation=LeakyReLU(alpha=0.1)))

d.add(Dense(1, activation='sigmoid'))
d.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')