# After reading the paper, this is a simple implementation of the network architectures I think they used. (Eliot C.)
# weirdly, we can't run the 3 networks together because of memory issues, so we have to run them separately.

#%% IMPORTS
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.models import Sequential

IMG_SIZE = (256, 256, 3)
N_ATTRIBUTES = 40

#%% GENERAL NOTES
# Let Ck be a Convolution-BatchNorm-ReLU layer
# with k filters. Convolutions use kernel of size 4 × 4, with a stride of 2, and a padding of 1, so that
# each layer of the encoder divides the size of its input by 2. We use leaky-ReLUs with a slope of 0.2
# in the encoder, and simple ReLUs in the decoder.


#%% Encoder
# The encoder consists of the following 7 layers:
# C16 − C32 − C64 − C128 − C256 − C512 − C512
encoder = Sequential([
    Conv2D(16, (4, 4), strides=(2, 2), padding='same', input_shape=IMG_SIZE),
    BatchNormalization(),
    LeakyReLU(0.2),
    
    Conv2D(32, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    
    Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    
    Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    
    Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    
    Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
    
    Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    LeakyReLU(0.2),
])

print("Encoder Summary:")
encoder.summary()


#%% Decoder
# the decoder is symmetric to the encoder, but uses transposed convolutions for the up-sampling:
# C512+2n − C512+2n − C256+2n − C128+2n − C64+2n − C32+2n − C16+2n .
decoder = Sequential([
    Conv2DTranspose(512+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same', input_shape=(2, 2, 512)),
    BatchNormalization(),
    ReLU(),
    
    Conv2DTranspose(512+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    ReLU(),
    
    Conv2DTranspose(256+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    ReLU(),
    
    Conv2DTranspose(128+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    ReLU(),
    
    Conv2DTranspose(64+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    ReLU(),
    
    Conv2DTranspose(32+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    ReLU(),
    
    Conv2DTranspose(16+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNormalization(),
    ReLU(),
])


print("Decoder Summary:")
decoder.summary()


#%% Discriminator
# The discriminator is a C512 layer followed by a fully-connected neural network of two layers of size 512 and n repsectively.
discriminator = Sequential([
    Conv2D(512, (4, 4), strides=(2, 2), padding='same', input_shape=IMG_SIZE),
    BatchNormalization(),
    LeakyReLU(0.2),
    
    Flatten(),
    Dense(512),
    BatchNormalization(),
    LeakyReLU(0.2),
    
    Dense(N_ATTRIBUTES),  # n is the number of attributes
])

print("Discriminator Summary:")
discriminator.summary()
# %%
