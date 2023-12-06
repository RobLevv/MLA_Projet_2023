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
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        
    def forward(self, x, y):
        input_x = x.float()
        input_y = y.float()
        latent = self.encoder(input_x)
        

        # recreate y to be of shape (batch_size, N_ATTRIBUTES, 2)
        # yhot = y.clone().detach().numpy()
        # yhot[yhot == -1] = 0
        # yhot = np.stack((yhot, 1-yhot), axis = 2)
        # yhot = torch.tensor(yhot, dtype = torch.float32)
        # print("yhot", yhot.shape, yhot.dtype)
        # print("y", y.shape, y.dtype)
        # print("latent", latent.shape, latent.dtype)
        input_y = input_y.unsqueeze(2).unsqueeze(3)
        # print("input_y", input_y.shape, input_y.dtype)
        input_y = input_y.expand(y.shape[0], y.shape[1], 2, 2)
        # print("input_y", input_y.shape, input_y.dtype)
        
        # concatenate latent and yhot
        latent_y = torch.cat((latent, input_y), dim = 1)
        # print("latent_y", latent_y.shape, latent_y.dtype)
        
        
        decoded = self.decoder(latent_y)
        return latent, decoded
    
    def loss(self, x, x_decoded, modified_loss = False):
        if not modified_loss:
            assert x.shape == x_decoded.shape, "x and x_decoded must have the same shape"    
            return mse_loss(x.float(), x_decoded).float()
        else:
            return torch.mean((x_decoded - x.float())**2)
    
    def autoencodeur_step(self, loss, n_epochs, dataloader, autoencoder):
        for batch_idx, batch in enumerate(data_loader):
            images, attributes = batch['image'], batch['attributes']
                        
            # print("\n\n\n", images.shape, attributes.shape)
            # print(images.dtype, attributes.dtype, "\n\n\n")
            
             
            # generate output
            latent, decoded = autoencoder(images, attributes)
            
            # compute losses
            loss_autoencoder = autoencoder.loss(images, decoded) # NOT IMPLEMENTED YET
            autoencoder.optimizer.zero_grad() 
            loss_autoencoder.backward()
            autoencoder.optimizer.step()


        
    
    
class Discriminator(nn.Module):
    def __init__(self, discriminator):
        super(Discriminator, self).__init__()
        self.discriminator = discriminator
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        
    def forward(self, x):
        input = x.float()
        # print("input", input.shape, input.dtype)
        return self.discriminator(input)
    
    def loss(self, y, y_discriminated, modified_loss = False):
        if not modified_loss:
            assert y.shape == y_discriminated.shape, "y and y_discriminated must have the same shape"
            return mse_loss(y.float(), y_discriminated).float()
        else:
            return -torch.mean(torch.log(y_discriminated))
    
##% This is the actual training loop
# All above we need to define those elsewhere

   
def train(n_epochs:int, device, autoencoder:AutoEncoder, discriminator:Discriminator, dataset:ImgDataset):
    """
    Train the autoencoder, discriminator, latent discriminator and patch discriminator.
    """
    for epoch in range(n_epochs):
        loss_train = 0
        for batch_idx, batch in enumerate(data_loader):
            images, attributes = batch['image'], batch['attributes']
            autoencoder.to(device)
            images, attributes = images.to(device), attributes.to(device)
                        
            # print("\n\n\n", images.shape, attributes.shape)
            # print(images.dtype, attributes.dtype, "\n\n\n")
            
             
            # generate output
            latent, decoded = autoencoder(images, attributes)
            
            # compute losses
            loss_autoencoder = autoencoder.loss(images, decoded) # NOT IMPLEMENTED YET
            autoencoder.optimizer.zero_grad() 
            loss_autoencoder.backward()
            autoencoder.optimizer.step()   
            
            # pred_y = discriminator(latent)            
            # loss_discriminator = discriminator.loss(attributes, pred_y) # NOT IMPLEMENTED YET
            # discriminator.optimizer.zero_grad()
            # loss_discriminator.backward()
            # discriminator.optimizer.step()
            loss_adversarial = ... # NOT IMPLEMENTED YET ( loss_autoencoder - loss_discriminator )
            # print("loss_autoencoder", loss_autoencoder.shape, loss_autoencoder.dtype, loss_autoencoder)
            # print("loss_discriminator", loss_discriminator.shape, loss_discriminator.dtype, loss_discriminator)

            if batch_idx%1000 == 0:
                print("Computed {}/{} images ({}%)\r".format(batch_idx*10, 202599, batch_idx*10/202599))
                print(f'Epoch {epoch}, loss {loss_train/(batch_idx+1):.2f}')
                images_cpu, decoded_cpu = images.cpu(), decoded.cpu()
                fig, ax = plt.subplots(2, 10, figsize = (20, 4))
                for i in range(10):
                    ax[0, i].imshow(images_cpu[i].permute(1, 2, 0))
                    ax[0, i].axis('off')
                    ax[1, i].imshow(decoded_cpu[i].permute(1, 2, 0).detach().numpy())
                    ax[1, i].axis('off')
                plt.tight_layout()
                plt.savefig("result_batch{}.png".format(batch_idx))

            # update loss
            loss_train += loss_autoencoder.item()
        loss_train /= len(data_loader)
        print(f'Epoch {epoch}, loss {loss_train:.2f}')
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = AutoEncoder(encoder_layers, decoder_layers)
    dis = Discriminator(discriminator_layers)
    
    dataset = ImgDataset(attributes_csv_file = 'data/Anno/list_attr_celeba.txt', img_root_dir = 'data/Img')
    data_loader = DataLoader(dataset, batch_size = 32, shuffle = True)
    train(n_epochs = 10, device = device, autoencoder = ae, discriminator = dis, dataset = dataset)
    batch = data_loader.__iter__().__next__()
    images, attributes = batch['image'], batch['attributes']
    images, attributes = images.to(device), attributes.to(device)
    latent, decoded = ae(images, attributes)
    images, decoded = images.cpu(), decoded.cpu()
    
    
    fig, ax = plt.subplots(2, 10, figsize = (20, 4))
    for i in range(10):
        ax[0, i].imshow(images[i].permute(1, 2, 0))
        ax[0, i].axis('off')
        ax[1, i].imshow(decoded[i].permute(1, 2, 0).detach().numpy())
        ax[1, i].axis('off')
    plt.tight_layout()
    plt.savefig("result_epoch{}.png".format(10))
    plt.show()

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
# compute the loss between I and y = > loss_I_y
# compute the loss between L and y = > loss_L_y

# we want to minimize loss_I_y
# we want to maximize loss_L_y
############################################################################################################

    