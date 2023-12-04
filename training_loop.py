import torch
import torch.nn as nn
import pandas as pd
from data_loader_V2 import load_dataset


from network_architectures import encoder_layers, decoder_layers, discriminator_layers

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        input = x.float()
        latent = self.encoder(input)
        decoded = self.decoder(latent)
        return latent, decoded
    
    def loss(self, x, x_decoded, modified_loss=False):
        if not modified_loss:
            return nn.MSELoss(x, x_decoded)
        else:
            return ...
    
class Discriminator(nn.Module):
    def __init__(self, discriminator):
        super(Discriminator, self).__init__()
        self.discriminator = discriminator
        
    def forward(self, x):
        input = x.float()
        return self.discriminator(input)
    
    def loss(self, y, y_discriminated, modified_loss=False):
        if not modified_loss:
            return nn.MSELoss(y, y_discriminated)
        else:
            return ...
    
##% This is the actual training loop
# All above we need to define those elsewhere
ae = AutoEncoder(encoder_layers, decoder_layers)
dis = Discriminator(discriminator_layers)

images, attributes = load_dataset(nb_examples=100, images_folder='data/Img_lite', attributes_file='data/Anno/list_attr_celeba.txt', target_size=256, shuffle=False, display=False)
   
def train(n_epochs:int, n_batch:int, autoencoder:AutoEncoder, discriminator:Discriminator, image_data:pd.DataFrame, attributes_data:pd.DataFrame):
    """
    Train the autoencoder, discriminator, latent discriminator and patch discriminator.
    """
    len_dataset = len(image_data)
    for epoch in range(n_epochs):
        loss_train = 0
        for index in image_data.index:
            image = image_data.loc[index]
            attributes = attributes_data.loc[index]
            
            # convert to tensor : (H, W, C) -> (C, H, W)
            image = torch.tensor(image)
            attributes = torch.tensor(attributes)
            
            print("\n\n\n", image.shape, attributes.shape, "\n\n\n")
            
            # generate output
            latent, decoded = autoencoder(image)
            pred_y = discriminator(latent)
            # compute losses
            loss_autoencoder = autoencoder.loss(image, decoded) # NOT IMPLEMENTED YET
            loss_discriminator = discriminator.loss(attributes, pred_y) # NOT IMPLEMENTED YET
            loss_adversarial = ... # NOT IMPLEMENTED YET ( loss_autoencoder - loss_discriminator )
            # optimizer zero grad
            autoencoder.optimizer.zero_grad()        
            discriminator.optimizer.zero_grad()
            # backpropagate
            loss_autoencoder.backward()
            loss_discriminator.backward()
            # optimizer step
            autoencoder.optimizer.step()
            discriminator.optimizer.step()
            # update loss
            loss_train += loss_autoencoder.item()
        loss_train /= len_dataset
        print(f'Epoch {epoch}, loss {loss_train:.2f}')
        
train(n_epochs=10, n_batch=32, autoencoder=ae, discriminator=dis, image_data=images, attributes_data=attributes)

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

    