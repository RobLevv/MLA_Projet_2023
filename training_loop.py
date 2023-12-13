import torch

from data_loader_V3 import ImgDataset
from utils import train_validation_test_split
from AutoEncoder import AutoEncoder
from Discriminator import Discriminator
from objectives import adversarial_objective, discriminator_objective, reconstruction_objective

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm


def train_loop(
    n_epochs:int, 
    device:torch.device,
    autoencoder:AutoEncoder, 
    discriminator:Discriminator, 
    data_loader:torch.utils.data.DataLoader,
    log_directory:str = "Logs",
    plot_images:bool = True
    ) -> None:
    """
    Train loop for the autoencoder and the discriminator
    """

    start_time = time.time()
    
    writer = SummaryWriter(log_directory, filename_suffix="_log")
    
    writer.add_text("Description", "Number of epochs : " + str(n_epochs) + "\n" +
                    "Size of training dataset : " + str(len(data_loader)) + "\n" +
                    "Batch size : " + str(data_loader.batch_size) + "\n" +
                    "Number of images in the training dataset : " + str(len(data_loader)*data_loader.batch_size) + "\n" +
                    "#"*50 + "\n" +
                    "Start training : "+ str(start_time) + "\n" +
                    "#"*50 + "\n")
    
    # loop over the epochs
    for epoch in range(n_epochs):
        
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
        
        # initialize epoch loss
        epoch_loss = 0.
        
        current_time = time.time() - start_time
        
        print("Epoch : " + str(epoch) + " / " + str(n_epochs) + ", {}H{}M{}S".format(current_time//3600, current_time%3600//60, current_time%60))
        
        for batch_nb, batch in enumerate(data_loader):
            
            pbar.update(1)
            
            # get the images and the attributes from the batch
            images, attributes = batch['image'], batch['attributes']
            
            # send model, images and attributes to the device ( GPU if available )
            images, attributes = images.to(device), attributes.to(device)    
             
            # Generate the latent space and the decoded images (outputs from the autoencoder)
            latent, decoded = autoencoder(images, attributes)
            
            # Generate the prediction of the discriminator
            y_pred = discriminator(latent)
            
            # Update the Encoder and Decoder weights
            loss_autoencoder = adversarial_objective(images, decoded, attributes, y_pred, lambda_ae=0.9) # TODO: change lambda
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

            # update epoch loss with the loss of the batch
            epoch_loss += loss_autoencoder.item()
            
            # write the losses in the log file
            global_batch_nb = epoch*len(data_loader) + batch_nb
            writer.add_scalar("Reconstruction_objective", reconstruction_objective(images, decoded).item(), global_batch_nb)
            writer.add_scalar("Adversarial_objective", loss_autoencoder.item(), global_batch_nb)
            writer.add_scalar("Discriminator_objective", loss_discriminator.item(), global_batch_nb)
            
        # Plot and save the images and the decoded images to compare
        if plot_images:
            # Get the first batch of the data loader
            batch = data_loader.__iter__().__next__()
            len_batch = len(batch['image'])
            # Get the images and the attributes from the batch
            images, attributes = batch['image'], batch['attributes']
            # Send images and attributes to the GPU, model already on GPU after training
            images, attributes = images.to(device), attributes.to(device)
            # Generate the latent space and the decoded images (outputs from the autoencoder)
            latent, decoded = autoencoder(images, attributes)
            # Send images and decoded back to the CPU
            images, decoded = images.cpu(), decoded.cpu()           
            # plot the images and the decoded images to compare
            fig, ax = plt.subplots(2, len_batch, figsize = (20, 4))
            for i,image in enumerate(images):
              image = image.detach().numpy()
              image = image/np.max(image)
              ax[0,i].imshow(np.transpose(image, (1,2,0)))
              ax[0,i].axis('off')
              ax[0,i].set_title('Original')
            for i,image in enumerate(decoded):
              image = image.detach().numpy()
              image = image/np.max(image)
              ax[1,i].imshow(np.transpose(image, (1,2,0)))
              ax[1,i].axis('off')
              ax[1,i].set_title('Decoded')
            plt.tight_layout()
            plt.savefig("Logs/epoch_" + str(epoch) + ".png")
            plt.close()
        
        epoch_loss /= len(data_loader)
    
    current_time = time.time() - start_time
    
    # write the end of the training in the log file
    writer.add_text("Description", "End training : "+ str(time.time()) + "\n" +
                    "#"*50 + "\n" +
                    "The training took : " + str(current_time//3600) + " hours, " + str(current_time%3600//60) + " minutes, " + str(current_time%60) + " seconds\n")
    writer.close()
    

if __name__ == "__main__":
    # initialize the gpu if available as material acceleration
    GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize the models
    ae = AutoEncoder()
    dis = Discriminator()
    
    ae.to(GPU)
    dis.to(GPU)

    # load model
    # ae.load_state_dict(torch.load("Models/autoencoder_e1.pt", map_location = torch.device('cpu')))
    
    # initialize the dataset and the data loader
    dataset = ImgDataset(attributes_csv_file = 'data/Anno/list_attr_celeba.txt', img_root_dir = 'data/Img')
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)

    train_set, validation_set, test_set = train_validation_test_split(dataset, train_split = 0.1, test_split = 0.9, val_split = 0., shuffle = True)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size = 50, shuffle = True)
        
    # train the models
    train_loop(
        n_epochs = 5, 
        device = GPU, 
        autoencoder = ae, 
        discriminator = dis, 
        data_loader = train_data_loader,
        log_directory = "Logs"
        )

    # save the model
    torch.save(ae.state_dict(), "Models/autoencoder.pt")
    # torch.save(dis.state_dict(), "Models/discriminator.pt")
    
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
