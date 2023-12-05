import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from data_loader_V3 import ImgDataset


from network_architectures import encoder_layers, decoder_layers, discriminator_layers

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x, y):
        input_x = x.float()
        input_y = y.float()
        latent = self.encoder(input_x)
        

        # recreate y to be of shape (batch_size, N_ATTRIBUTES, 2)
        yhot = y.clone().detach().numpy()
        yhot[yhot == -1] = 0
        yhot = np.stack((yhot, 1-yhot), axis=2)
        yhot = torch.tensor(yhot, dtype=torch.float32)
        # print("yhot", yhot.shape, yhot.dtype)
        # print("y", y.shape, y.dtype)
        # print("latent", latent.shape, latent.dtype)
        input_y = input_y.unsqueeze(2).unsqueeze(3)
        print("input_y", input_y.shape, input_y.dtype)
        input_y = input_y.expand(y.shape[0], y.shape[1], 2, 2)
        print("input_y", input_y.shape, input_y.dtype)
        
        # concatenate latent and yhot
        latent_y = torch.cat((latent, input_y), dim=1)
        print("latent_y", latent_y.shape, latent_y.dtype)
        
        
        decoded = self.decoder(latent_y)
        return latent, decoded
    
    def loss(self, x, x_decoded, modified_loss=False):
        if not modified_loss:
            assert x.shape == x_decoded.shape, "x and x_decoded must have the same shape"    
            return mse_loss(x.float(), x_decoded).float()
        else:
            return ...
        
    
    
class Discriminator(nn.Module):
    def __init__(self, discriminator):
        super(Discriminator, self).__init__()
        self.discriminator = discriminator
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x):
        input = x.float()
        # print("input", input.shape, input.dtype)
        return self.discriminator(input)
    
    def loss(self, y, y_discriminated, modified_loss=False):
        if not modified_loss:
            assert y.shape == y_discriminated.shape, "y and y_discriminated must have the same shape"
            return mse_loss(y.float(), y_discriminated).float()
        else:
            return ...
    
##% This is the actual training loop
# All above we need to define those elsewhere
ae = AutoEncoder(encoder_layers, decoder_layers)
dis = Discriminator(discriminator_layers)

dataset = ImgDataset(attributes_csv_file='data/Anno/list_attr_celeba_lite.txt', img_root_dir='data/Img_lite')
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
   
def train(n_epochs:int, n_batch:int, autoencoder:AutoEncoder, discriminator:Discriminator, dataset:ImgDataset):
    """
    Train the autoencoder, discriminator, latent discriminator and patch discriminator.
    """
    for epoch in range(n_epochs):
        loss_train = 0
        for batch_idx, batch in enumerate(data_loader):
            images, attributes = batch['image'], batch['attributes']
                        
            # print("\n\n\n", images.shape, attributes.shape)
            # print(images.dtype, attributes.dtype, "\n\n\n")
            
             
            # generate output
            latent, decoded = autoencoder(images, attributes)
            pred_y = discriminator(latent)
            # compute losses
            loss_autoencoder = autoencoder.loss(images, decoded) # NOT IMPLEMENTED YET
            loss_discriminator = discriminator.loss(attributes, pred_y) # NOT IMPLEMENTED YET
            loss_adversarial = ... # NOT IMPLEMENTED YET ( loss_autoencoder - loss_discriminator )
            # print("loss_autoencoder", loss_autoencoder.shape, loss_autoencoder.dtype, loss_autoencoder)
            # print("loss_discriminator", loss_discriminator.shape, loss_discriminator.dtype, loss_discriminator)
            
            # optimizer step
            nn.utils.clip_grad_norm(AutoEncoder.parameters(), 0.5)
            autoencoder.optimizer.zero_grad() 
            loss_autoencoder.backward(retain_graph=True)
            autoencoder.optimizer.step()     
            discriminator.optimizer.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            discriminator.optimizer.step()

            # update loss
            loss_train += loss_autoencoder.item()
        loss_train /= len(data_loader)
        print(f'Epoch {epoch}, loss {loss_train:.2f}')
        
train(n_epochs=10, n_batch=10, autoencoder=ae, discriminator=dis, dataset=dataset)

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

    