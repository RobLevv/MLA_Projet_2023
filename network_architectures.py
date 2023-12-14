# After reading the paper, this is a simple implementation of the network architectures I think they used. (Eliot C.)
# weirdly, we can't run the 3 networks together because of memory issues, so we have to run them separately.

#%% IMPORTS
from torch.nn import Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, LeakyReLU, Flatten, Linear, Sigmoid, Dropout

IMG_SIZE = (3, 256, 256)
N_ATTRIBUTES = 40

#%% GENERAL NOTES
# Let Ck be a Convolution-BatchNorm-ReLU layer
# with k filters. Convolutions use kernel of size 4 × 4, with a stride of 2, and a padding of 1, so that
# each layer of the encoder divides the size of its input by 2. We use leaky-ReLUs with a slope of 0.2
# in the encoder, and simple ReLUs in the decoder.

#%% Encoder
# The encoder consists of the following 7 layers:
# C16 − C32 − C64 − C128 − C256 − C512 − C512
encoder_layers = Sequential(
    # C16: Convolutional layer with 16 filters
    Conv2d(in_channels=IMG_SIZE[0], out_channels=16, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(16),
    LeakyReLU(0.2, inplace=True),

    # C32: Convolutional layer with 32 filters
    Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(32),
    LeakyReLU(0.2, inplace=True),

    # C64: Convolutional layer with 64 filters
    Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(64),
    LeakyReLU(0.2, inplace=True),

    # C128: Convolutional layer with 128 filters
    Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(128),
    LeakyReLU(0.2, inplace=True),

    # C256: Convolutional layer with 256 filters
    Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(256),
    LeakyReLU(0.2, inplace=True),

    # C512: Convolutional layer with 512 filters
    Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(512),
    LeakyReLU(0.2, inplace=True),

    # C512: Convolutional layer with 512 filters
    Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(512),
    LeakyReLU(0.2, inplace=True)
)

if __name__ == "__main__":

    print("Encoder Summary:")
    print(encoder_layers)


#%% Decoder
# the decoder is symmetric to the encoder, but uses transposed convolutions for the up-sampling:
# C512+2n − C512+2n − C256+2n − C128+2n − C64+2n − C32+2n − C16+2n .
decoder_layers = Sequential(
    # C512+2n: Transposed convolutional layer with 512 + 2 * N_ATTRIBUTES filters
    ConvTranspose2d(in_channels=512 + N_ATTRIBUTES, out_channels=512 + N_ATTRIBUTES, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(512 + N_ATTRIBUTES),
    ReLU(inplace=True),

    # C512+2n: Transposed convolutional layer with 512 + 2 * N_ATTRIBUTES filters
    ConvTranspose2d(in_channels=512 +  N_ATTRIBUTES, out_channels=512 + N_ATTRIBUTES, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(512 + N_ATTRIBUTES),
    ReLU(inplace=True),

    # C256+2n: Transposed convolutional layer with 256 + 2 * N_ATTRIBUTES filters
    ConvTranspose2d(in_channels=512 +  N_ATTRIBUTES, out_channels=256 + N_ATTRIBUTES, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(256 + N_ATTRIBUTES),
    ReLU(inplace=True),

    # C128+2n: Transposed convolutional layer with 128 + 2 * N_ATTRIBUTES filters
    ConvTranspose2d(in_channels=256 +  N_ATTRIBUTES, out_channels=128 + N_ATTRIBUTES, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(128 + N_ATTRIBUTES),
    ReLU(inplace=True),

    # C64+2n: Transposed convolutional layer with 64 + 2 * N_ATTRIBUTES filters
    ConvTranspose2d(in_channels=128 +  N_ATTRIBUTES, out_channels=64 + N_ATTRIBUTES, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(64 + N_ATTRIBUTES),
    ReLU(inplace=True),

    # C32+2n: Transposed convolutional layer with 32 + 2 * N_ATTRIBUTES filters
    ConvTranspose2d(in_channels=64 +  N_ATTRIBUTES, out_channels=32 + N_ATTRIBUTES, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(32 + N_ATTRIBUTES),
    ReLU(inplace=True),

    # C16+2n: Transposed convolutional layer with 16 + 2 * N_ATTRIBUTES filters
    ConvTranspose2d(in_channels=32 +  N_ATTRIBUTES, out_channels=3, kernel_size=4, stride=2, padding=1),
    BatchNorm2d(3),
    ReLU(inplace=True),
    
    
)

if __name__ == "__main__":

    print("Decoder Summary:")
    print(decoder_layers)
    print( "Total number of parameters:", sum(p.numel() for p in decoder_layers.parameters() if p.requires_grad) )
    print( "Total number of trainable parameters:", sum(p.numel() for p in decoder_layers.parameters() if p.requires_grad and p.is_leaf) )
    print( "Total number of non-trainable parameters:", sum(p.numel() for p in decoder_layers.parameters() if not p.requires_grad) )


#%% Discriminator
# The discriminator is a C512 layer followed by a fully-connected neural network of two layers of size 512 and n repsectively.
discriminator_layers = Sequential(
    # C512: Convolutional layer with 512 filters
    Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding="same"),
    BatchNorm2d(512),
    LeakyReLU(0.2, inplace=True),
    
    # Dropout layer with probability 0.2
    Dropout(0.2),
    
    # Flatten the output for fully-connected layers
    Flatten(),

    # Fully-connected layer with size 512
    Linear(512*2*2, 512),
    LeakyReLU(0.2, inplace=True),
    
    # Dropout layer with probability 0.2
    Dropout(0.2),

    # Fully-connected layer with size n (assuming n is the number of attributes)
    Linear(512, N_ATTRIBUTES),
    
    # Sigmoid activation function
    Sigmoid()
)

if __name__ == "__main__":
    
    print("Discriminator Summary:")
    print(discriminator_layers)
# %%
