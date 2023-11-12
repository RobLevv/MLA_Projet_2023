# After reading the paper, this is a simple implementation of the network architectures I think they used. (Eliot C.)

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (256, 256, 3)
N_ATTRIBUTES = 40


# Encoder
encoder = models.Sequential([
    layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same', input_shape=IMG_SIZE),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    
    layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    
    layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    
    layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    
    layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    
    layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    
    layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
])

# Decoder
decoder = models.Sequential([
    layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', input_shape=(2, 2, 512)),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
])

# Discriminator
discriminator = models.Sequential([
    layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', input_shape=IMG_SIZE),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    
    layers.Flatten(),
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    
    layers.Dense(N_ATTRIBUTES),  # n is the number of attributes
])

# Printing model summaries
print("Encoder Summary:")
encoder.summary()

print("\nDecoder Summary:")
decoder.summary()

print("\nDiscriminator Summary:")
discriminator.summary()
