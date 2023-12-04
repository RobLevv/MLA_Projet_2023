
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from data_loader_V2 import load_dataset


from network_architectures import encoder, decoder, discriminator

class autoencoder(Model):
    def __init__(self, encoder, decoder,latent_dim):
        super(autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded
    
    def encode(self, x):
        latent = self.encoder(x)
        return latent
    
    def decode(self, latent):
        decoded = self.decoder(latent)
        return decoded
    
class discriminator(Model):
    def __init__(self, discriminator):
        super(discriminator, self).__init__()
        self.discriminator = discriminator
        
    def call(self, x):
        return self.discriminator(x)
    
##% This is the actual training loop
# All above we need to define those elsewhere
ae = autoencoder(encoder, decoder, 512)
dis = discriminator(discriminator)

ae.compile(optimizer='adam', loss='mse')
dis.compile(optimizer='adam', loss='mse')

images, attributes = load_dataset(nb_examples=202599, images_folder='data/Img', attributes_file='data/Anno/list_attr_celeba.txt', target_size=256, shuffle=True, display=True)
   
def train(n_epochs:int, n_batch:int, autoencoder:Model, discriminator:Model, latent_discriminator:Model, patch_discriminator:Model, dataset):
    """
    Train the autoencoder, discriminator, latent discriminator and patch discriminator.
    """
    for epoch in range(n_epochs):
        autoencoder.fit(dataset, epochs=1, batch_size=n_batch)
        discriminator.fit(dataset, epochs=1, batch_size=n_batch)
        latent_discriminator.fit(dataset, epochs=1, batch_size=n_batch)
        patch_discriminator.fit(dataset, epochs=1, batch_size=n_batch)
        # Evaluate the models
        autoencoder.evaluate(dataset)
        discriminator.evaluate(dataset)
        latent_discriminator.evaluate(dataset)
        patch_discriminator.evaluate(dataset)
        
train(100, 32, ae, dis, lat_dis, ptc_dis, dataset)

############################################################################################################
# This is a good architecture but not precise enough we need:
# compute our own losses
# redefine the .fit method of each model
# redefine the .evaluate method of each model

# autoencoder fit:
# encode the input to latent space
# decode the latent space to output
# return latent space and output
# compute the loss between the input and the output

# discriminator fit:
# discriminate the input
# return the prediction I
# discriminate the latent space
# return the prediction L
# compute the loss between I and y => loss_I_y
# compute the loss between L and y => loss_L_y

# we want to minimize loss_I_y
# we want to maximize loss_L_y
############################################################################################################

    