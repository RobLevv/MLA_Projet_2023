import torch

from data_loader_V3 import ImgDataset
from AutoEncoder import AutoEncoder
from Discriminator import Discriminator
from objectives import adversarial_objective, discriminator_objective

import numpy as np

   
def train_loop(
    n_epochs:int, 
    device:torch.device,
    autoencoder:AutoEncoder, 
    discriminator:Discriminator, 
    data_loader:torch.utils.data.DataLoader,
    display:bool = False,
    display_ultra_detailed:bool = False
    ) -> None:
    """
    Train loop for the autoencoder and the discriminator
    """
    
    for epoch in range(n_epochs):
        
        # initialize epoch loss
        epoch_loss = 0.
        
        for batch_nb, batch in enumerate(data_loader):
            
            # get the images and the attributes from the batch
            images, attributes = batch['image'], batch['attributes']
            
            # send model, images and attributes to the device ( GPU if available )
            autoencoder.to(device)
            images, attributes = images.to(device), attributes.to(device)
            
            if display_ultra_detailed:
                print("images", images.shape, images.dtype)
                print("attributes", attributes.shape, attributes.dtype)           
             
            # Generate the latent space and the decoded images (outputs from the autoencoder)
            latent, decoded = autoencoder(images, attributes)
            
            # Generate the prediction of the discriminator
            y_pred = discriminator(latent)
            
            # Update the Encoder and Decoder weights
            loss_autoencoder = adversarial_objective(images, decoded, attributes, y_pred, lamb=0.9)
            autoencoder.optimizer.zero_grad() 
            loss_autoencoder.backward()
            autoencoder.optimizer.step()

            # Detach the gradient computation from autoencoder to avoid backpropagating through it
            latent.detach_()
            
            # Generate the prediction of the discriminator
            y_pred = discriminator(latent)
            
            # Update the Discriminator weights          
            loss_discriminator = discriminator_objective(attributes, y_pred)
            discriminator.optimizer.zero_grad()
            loss_discriminator.backward()
            discriminator.optimizer.step()
            
            if display_ultra_detailed:
                # print the losses
                print("  epoch : ", epoch, 
                    "  batch_index : ", batch_nb, 
                    "  loss_autoencoder : ", round(loss_autoencoder.item(), 2), 
                    "  loss_discriminator : ", round(loss_discriminator.item(), 4))

                # print the number of attributes predicted correctly by the discriminator
                pred_attributes = torch.where(y_pred > 0, torch.ones_like(y_pred), -torch.ones_like(y_pred))
                print("  nb of attributes predicted correctly : ", torch.sum(pred_attributes == attributes).item(), "over", pred_attributes.shape[0]*pred_attributes.shape[1])

            if (batch_nb%1000 == 0) and display: # print every 1000 mini-batches, TODO implement a logger
                print("Computed {}/{} images ({}%)\r".format(batch_nb*10, 202599, batch_nb*10/202599))
                print(f'Epoch {epoch}, loss {epoch_loss/(batch_nb+1):.2f}')
                images_cpu, decoded_cpu = images.cpu(), decoded.cpu()
                fig, ax = plt.subplots(2, 10, figsize = (20, 4))
                for i in range(10):
                    ax[0, i].imshow(images_cpu[i].permute(1, 2, 0))
                    ax[0, i].axis('off')
                    ax[1, i].imshow(np.clip(decoded_cpu[i].permute(1, 2, 0).detach().numpy(), a_min=0, a_max=1))
                    ax[1, i].axis('off')
                plt.tight_layout()
                plt.savefig("result_batch{}.png".format(batch_nb))

            # update epoch loss with the loss of the batch
            epoch_loss += loss_autoencoder.item()
        
        epoch_loss /= len(data_loader)
        if display:
            print(f'Epoch {epoch}, loss {epoch_loss:.2f}')
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # initialize the gpu if available as material acceleration
    GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize the models
    ae = AutoEncoder()
    dis = Discriminator()
    
    # initialize the dataset and the data loader
    dataset = ImgDataset(attributes_csv_file = 'data/Anno/list_attr_celeba_lite.txt', img_root_dir = 'data/Img_lite')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)
    
    # train the models
    train_loop(
        n_epochs = 10, 
        device = GPU, 
        autoencoder = ae, 
        discriminator = dis, 
        data_loader = data_loader,
        display = False,
        display_ultra_detailed = True
        )
    
    
    # At the end of the training, we want to plot the images and the decoded images to compare
    end_training_visualisation = False
    if end_training_visualisation:
        # Get the first batch of the data loader
        batch = data_loader.__iter__().__next__()
        # Get the images and the attributes from the batch
        images, attributes = batch['image'], batch['attributes']
        # Send images and attributes to the GPU, model already on GPU after training
        images, attributes = images.to(GPU), attributes.to(GPU)
        # Generate the latent space and the decoded images (outputs from the autoencoder)
        latent, decoded = ae(images, attributes)
        # Send images and decoded  back to the CPU
        images, decoded = images.cpu(), decoded.cpu()
        
        
        # plot the images and the decoded images to compare
        fig, ax = plt.subplots(2, 10, figsize = (20, 4))
        for i in range(10):
            ax[0, i].imshow(images[i].permute(1, 2, 0))
            ax[0, i].axis('off')
            ax[1, i].imshow(decoded[i].permute(1, 2, 0).detach().numpy())
            ax[1, i].axis('off')
        plt.tight_layout()
        plt.show()