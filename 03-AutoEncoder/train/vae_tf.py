import keras
from keras.layers import Dense
from keras.datasets import mnist
from keras import backend as K
import numpy as np

encoding_dim = 64
latent_dims = 2

inputs = keras.Input(shape=(784,))

# Encoder block
encoder = Dense(512, activation='relu')(inputs)
encoder = Dense(256, activation='relu')(encoder)
encoder = Dense(128, activation='relu')(encoder)
h = Dense(encoding_dim, activation='relu')(encoder)

z_mean = Dense(latent_dims)(h)
z_log_sigma = Dense(latent_dims)(h)

# Sampling block

def sampling(args):

    z_mean, z_log_sigma = args 

    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dims), 
                                mean=0, stddev=0.1)

    return z_mean + K.exp(z_log_sigma) * epsilon

z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

latent_inputs = keras.Input(shape=(latent_dims, ), name='z_sampling')
x = Dense(encoding_dim, activation='relu')(latent_inputs)

# Decoder block
decode = Dense(128, activation='relu')(x)
decode = Dense(256, activation='relu')(decode)
decode = Dense(512, activation='relu')(decode)
output = Dense(784, activation='sigmoid')(decode)

decoder = keras.Model(latent_inputs, output, name='decoder')

final_out = decoder(encoder(inputs)[2])

vae = keras.Model(inputs, final_out, name='vae')

# Loss
recon_loss = keras.losses.binary_crossentropy(inputs, final_out)
recon_loss *= 784

# KL Divergence
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

# Total loss
vae_loss = K.mean(recon_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train the model with MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_train), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train, batch_size=128, epochs=50, verbose=1, validation_data=(x_test, x_test))