
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


from network_architectures import encoder, decoder, discriminator

class autoencoder(Model):
    def __init__(self, encoder, decoder,latent_dim):
        super(autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, x):
        decoded = self.decoder(x)
        return decoded
    
class discriminator(Model):
    def __init__(self, discriminator):
        super(discriminator, self).__init__()
        self.discriminator = discriminator
        
    def call(self, x):
        return self.discriminator(x)
    
class latent_discriminator(Model):
    def __init__(self, latent_discriminator):
        super(latent_discriminator, self).__init__()
        self.latent_discriminator = latent_discriminator
        
    def call(self, x):
        return self.latent_discriminator(x)
    
class patch_discriminator(Model):
    def __init__(self, patch_discriminator):
        super(patch_discriminator, self).__init__()
        self.patch_discriminator = patch_discriminator
        
    def call(self, x):
        return self.patch_discriminator(x)
    
ae = autoencoder(encoder, decoder, 512)
dis = discriminator(discriminator)
lat_dis = latent_discriminator(latent_discriminator)
ptc_dis = patch_discriminator(patch_discriminator)

ae.compile(optimizer='adam', loss='mse')
dis.compile(optimizer='adam', loss='mse')
lat_dis.compile(optimizer='adam', loss='mse')
ptc_dis.compile(optimizer='adam', loss='mse')

   
def train(n_epochs:int, n_batch:int, autoencoder:Model, discriminator:Model, latent_discriminator:Model, patch_discriminator:Model, dataset:tf.data.Dataset):
    """
    Train the autoencoder, discriminator, latent discriminator and patch discriminator.
    """
    
    