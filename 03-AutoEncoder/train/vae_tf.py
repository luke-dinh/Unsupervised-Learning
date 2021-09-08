import keras
from keras.layers import Dense
from keras.datasets import mnist
from keras import backend as K

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