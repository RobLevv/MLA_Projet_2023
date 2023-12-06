import torch

from data_loader_V3 import ImgDataset
from AutoEncoder import AutoEncoder
from Discriminator import Discriminator

   
def train(n_epochs:int, device, autoencoder:AutoEncoder, discriminator:Discriminator, data_loader:torch.utils.data.DataLoader) -> None:
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
            
            # Autoencoder backward pass
            loss_autoencoder = autoencoder.loss(images, decoded)
            autoencoder.optimizer.zero_grad() 
            loss_autoencoder.backward()
            autoencoder.optimizer.step()   

            # Detach the gradient computation from autoencoder to avoid backpropagating through it
            latent.detach_()

            # Discriminator backward pass
            pred_y = discriminator(latent)            
            loss_discriminator = discriminator.loss(attributes, pred_y)
            discriminator.optimizer.zero_grad()
            loss_discriminator.backward()
            discriminator.optimizer.step()

            
            loss_adversarial = ... # NOT IMPLEMENTED YET ( loss_autoencoder - loss_discriminator )
            
            """PRINT THE LOSSES"""
            # print("  epoch", epoch, 
            #       "  batch_idx", batch_idx, 
            #       "  loss_autoencoder", round(loss_autoencoder.item(), 2), 
            #       "  loss_discriminator", round(loss_discriminator.item(), 4))

            """PRINT THE NUMBER OF ATTRIBUTES PREDICTED CORRECTLY BY THE DISCRIMINATOR"""
            # pred_attributes = torch.where(pred_y > 0, torch.ones_like(pred_y), -torch.ones_like(pred_y))
            # print("nb of attributes predicted correctly", torch.sum(pred_attributes == attributes).item(), "over", pred_attributes.shape[0]*pred_attributes.shape[1])

            if batch_idx%1000 == 0: # print every 1000 mini-batches, TODO implement a logger
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
    ae = AutoEncoder()
    dis = Discriminator()
        
    dataset = ImgDataset(attributes_csv_file = 'data/Anno/list_attr_celeba.txt', img_root_dir = 'data/Img')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)
    train(n_epochs = 10, device = device, autoencoder = ae, discriminator = dis, data_loader = data_loader)
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

    