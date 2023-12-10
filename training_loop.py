import torch

from data_loader_V3 import ImgDataset
from AutoEncoder import AutoEncoder
from Discriminator import Discriminator
from objectives import adversarial_objective, discriminator_objective

import matplotlib.pyplot as plt
import time

   
def train_loop(
    n_epochs:int, 
    device:torch.device,
    autoencoder:AutoEncoder, 
    discriminator:Discriminator, 
    data_loader:torch.utils.data.DataLoader,
    display:bool = False,
    display_ultra_detailed:bool = False,
    logger:bool = False
    ) -> None:
    """
    Train loop for the autoencoder and the discriminator
    """

    # Create the log file and write the statistics of the training
    if logger:
        t0 = time.time()
        with open("Logs/log.txt", "w") as f:
            f.write("Training statistics : \n")
            f.write(" Number of Epochs : " + str(n_epochs) + "\n")
            f.write(" Size of training dataset : " + str(len(data_loader)) + "\n")
            f.write(" Batch size : " + str(data_loader.batch_size) + "\n")
            f.write(" Number of images in the training dataset : " + str(len(data_loader)*data_loader.batch_size) + "\n")
            f.write("#"*50 + "\n")
            f.write("Start training : "+ str(time.time()) + "\n")
            f.write("#"*50 + "\n")
            
    # loop over the epochs
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
                print("  nb of attributes predicted correctly : ", torch.sum(pred_attributes == attributes).item(), " / ", pred_attributes.shape[0]*pred_attributes.shape[1])

            # Update the log file
            if logger:
                # open the log file in append mode
                with open("Logs/log.txt", "a") as f:
                    # write the losses
                    f.write("  epoch : " + str(epoch) + " / " + str(n_epochs) +
                        "  batch_index : " + str(batch_nb) + " / " + str(len(data_loader)) + "\n" +
                        "  loss_autoencoder : " + str(round(loss_autoencoder.item(), 2)) + 
                        "  loss_discriminator : " + str(round(loss_discriminator.item(), 4)) + "\n")

                    # write the number of attributes predicted correctly by the discriminator
                    pred_attributes = torch.where(y_pred > 0, torch.ones_like(y_pred), -torch.ones_like(y_pred))
                    f.write("  nb of attributes predicted correctly : " + str(torch.sum(pred_attributes == attributes).item()) + " / " + str(pred_attributes.shape[0]*pred_attributes.shape[1]) + "\n")
            
            # update epoch loss with the loss of the batch
            epoch_loss += loss_autoencoder.item()
            
        # Plot and save the images and the decoded images to compare
        if logger:
            # Get the first batch of the data loader
            batch = data_loader.__iter__().__next__()
            # Get the images and the attributes from the batch
            images, attributes = batch['image'], batch['attributes']
            # Send images and attributes to the GPU, model already on GPU after training
            images, attributes = images.to(device), attributes.to(device)
            # Generate the latent space and the decoded images (outputs from the autoencoder)
            latent, decoded = autoencoder(images, attributes)
            # Send images and decoded back to the CPU
            images, decoded = images.cpu(), decoded.cpu()
            # plot the images and the decoded images to compare
            fig, ax = plt.subplots(2, len(batch), figsize = (20, 4))
            for i in range(len(batch)):
                ax[0, i].imshow(images[i].permute(1, 2, 0))
                ax[0, i].axis('off')
                ax[1, i].imshow(decoded[i].permute(1, 2, 0).detach().numpy().clip(0, 1))
                ax[1, i].axis('off')
            plt.tight_layout()
            plt.savefig("Logs/epoch_" + str(epoch) + ".png")
            plt.close()
        
        epoch_loss /= len(data_loader)
        if display:
            print(f'Epoch {epoch}, loss {epoch_loss:.2f}')
    if logger:
        # write the end of the training in the log file
        with open("Logs/log.txt", "a") as f:
            f.write("#"*50 + "\n")
            f.write("End training : "+ str(time.time()) + "\n")
            f.write("#"*50 + "\n")
            f.write("The training took : " + str((time.time() - t0)//3600) + " hours, " + str((time.time() - t0)%3600//60) + " minutes, " + str((time.time() - t0)%60) + " seconds\n")
            

if __name__ == "__main__":
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
        display_ultra_detailed = False,
        logger = True
        )
    
    # plot the losses
    with open("Logs/log.txt", "r") as f:
        lines = f.readlines()
        ae_losses = []
        dis_losses = []
        for line in lines:
            if "loss_autoencoder" in line:
                ae_losses.append(float(line.split()[2]))
            if "loss_discriminator" in line:
                dis_losses.append(float(line.split()[5]))
    plt.figure(figsize = (20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(ae_losses, label = "autoencoder loss")
    plt.xlabel("batch index")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    plt.plot(dis_losses, label = "discriminator loss")
    plt.xlabel("batch index")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("Logs/losses.png")
    