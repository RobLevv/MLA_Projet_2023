# After reading the paper, this is a simple implementation of the network architectures I think they used. (Eliot C.)
# weirdly, we can't run the 3 networks together because of memory issues, so we have to run them separately.

#%% IMPORTS
from torch.nn import Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, LeakyReLU, Flatten, Linear

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
    Conv2d(16, (4, 4), strides=(2, 2), padding='same', input_shape=IMG_SIZE),
    BatchNorm2d(),
    LeakyReLU(0.2),
    
    Conv2d(32, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    LeakyReLU(0.2),
    
    Conv2d(64, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    LeakyReLU(0.2),
    
    Conv2d(128, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    LeakyReLU(0.2),
    
    Conv2d(256, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    LeakyReLU(0.2),
    
    Conv2d(512, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    LeakyReLU(0.2),
    
    Conv2d(512, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    LeakyReLU(0.2),
])

print("Encoder Summary:")
encoder.summary()


#%% Decoder
# the decoder is symmetric to the encoder, but uses transposed convolutions for the up-sampling:
# C512+2n − C512+2n − C256+2n − C128+2n − C64+2n − C32+2n − C16+2n .
decoder = Sequential([
    ConvTranspose2d(512+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same', input_shape=(2, 2, 512)),
    BatchNorm2d(),
    ReLU(),
    
    ConvTranspose2d(512+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    ReLU(),
    
    ConvTranspose2d(256+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    ReLU(),
    
    ConvTranspose2d(128+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    ReLU(),
    
    ConvTranspose2d(64+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    ReLU(),
    
    ConvTranspose2d(32+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    ReLU(),
    
    ConvTranspose2d(16+2*N_ATTRIBUTES, (4, 4), strides=(2, 2), padding='same'),
    BatchNorm2d(),
    ReLU(),
])


print("Decoder Summary:")
decoder.summary()


#%% Discriminator
# The discriminator is a C512 layer followed by a fully-connected neural network of two layers of size 512 and n repsectively.
discriminator = Sequential([
    Conv2d(512, (4, 4), strides=(2, 2), padding='same', input_shape=IMG_SIZE),
    BatchNorm2d(),
    LeakyReLU(0.2),
    
    Flatten(),
    Linear(512),
    BatchNorm2d(),
    LeakyReLU(0.2),
    
    Linear(N_ATTRIBUTES),  # n is the number of attributes
])

print("Discriminator Summary:")
discriminator.summary()
# %%
