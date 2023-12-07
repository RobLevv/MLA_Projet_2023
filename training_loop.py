import torch

from data_loader_V3 import ImgDataset
from AutoEncoder import AutoEncoder
from Discriminator import Discriminator

   
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
            
            # send the images and the attributes to the device
            autoencoder.to(device)
            images, attributes = images.to(device), attributes.to(device)
            
            if display_ultra_detailed:
                print("images", images.shape, images.dtype)
                print("attributes", attributes.shape, attributes.dtype)           
             
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
            
            if display_ultra_detailed:
                # print the losses
                print("  epoch", epoch, 
                    "  batch_idx", batch_nb, 
                    "  loss_autoencoder", round(loss_autoencoder.item(), 2), 
                    "  loss_discriminator", round(loss_discriminator.item(), 4))

                # print the number of attributes predicted correctly by the discriminator
                pred_attributes = torch.where(pred_y > 0, torch.ones_like(pred_y), -torch.ones_like(pred_y))
                print("  nb of attributes predicted correctly", torch.sum(pred_attributes == attributes).item(), "over", pred_attributes.shape[0]*pred_attributes.shape[1])

            if (batch_nb%1000 == 0) and display: # print every 1000 mini-batches, TODO implement a logger
                print("Computed {}/{} images ({}%)\r".format(batch_nb*10, 202599, batch_nb*10/202599))
                print(f'Epoch {epoch}, loss {epoch_loss/(batch_nb+1):.2f}')
                images_cpu, decoded_cpu = images.cpu(), decoded.cpu()
                fig, ax = plt.subplots(2, 10, figsize = (20, 4))
                for i in range(10):
                    ax[0, i].imshow(images_cpu[i].permute(1, 2, 0))
                    ax[0, i].axis('off')
                    ax[1, i].imshow(decoded_cpu[i].permute(1, 2, 0).detach().numpy())
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
    
    # initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize the models
    ae = AutoEncoder()
    dis = Discriminator()
    
    # initialize the dataset and the data loader
    dataset = ImgDataset(attributes_csv_file = 'data/Anno/list_attr_celeba.txt', img_root_dir = 'data/Img')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)
    
    # train the models
    train_loop(
        n_epochs = 10, 
        device = device, 
        autoencoder = ae, 
        discriminator = dis, 
        data_loader = data_loader,
        display = True,
        display_ultra_detailed = False
        )
    
    # ??? TODO: comment this part
    batch = data_loader.__iter__().__next__()
    images, attributes = batch['image'], batch['attributes']
    images, attributes = images.to(device), attributes.to(device)
    latent, decoded = ae(images, attributes)
    images, decoded = images.cpu(), decoded.cpu()
    
    
    # plot the images and the decoded images to compare
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

    