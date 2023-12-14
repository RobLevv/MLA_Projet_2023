import torch

from ImgDataset import get_celeba_dataset
from utils import train_validation_test_split, save_plot_images_comparision, save_plot_losses
from AutoEncoder import AutoEncoder
from Discriminator import Discriminator
from Logger import Logger
from objectives import adversarial_objective, discriminator_objective, reconstruction_objective

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
    attributes_columns:list,
    log_directory:str = "Logs",
    plot_images:bool = True,
    ) -> str:
    """
    Train loop for the autoencoder and the discriminator
    return the name of the log directory
    """

    start_time = time.time()
    
    dir_name = log_directory + "/start_" + time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime(start_time)) + "_logs"
        
    writer = Logger(log_dir=dir_name, separator="\n")
    
    writer.add("Description", "Number of epochs : " + str(n_epochs) + "\n" +
                    "Size of training dataset : " + str(len(data_loader)) + "\n" +
                    "Batch size : " + str(data_loader.batch_size) + "\n" +
                    "Number of images in the training dataset : " + str(len(data_loader)*data_loader.batch_size) + "\n" +
                    "#"*50 + "\n" +
                    "Start training : "+ str(start_time) + "\n" +
                    "#"*50 + "\n")
    
    writer.add("Description", "Autoencoder : \n" + str(autoencoder) + "\n")
    writer.add("Description", "Discriminator : \n" + str(discriminator) + "\n")
    
    # loop over the epochs
    for epoch in range(n_epochs):
        
        # initialize epoch loss
        epoch_loss = 0.
        
        current_time = time.time() - start_time
        
        print("\nEpoch : " + str(epoch + 1) + "/" + str(n_epochs) + " time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
        
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), ncols=100)
        
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
            writer.add("Reconstruction_objective", reconstruction_objective(images, decoded).item())
            writer.add("Adversarial_objective", loss_autoencoder.item())
            writer.add("Discriminator_objective", loss_discriminator.item())
            
            
        # Plot and save the images and the decoded images to compare
        if plot_images:
            
            save_plot_images_comparision(
                images = images, 
                decoded = decoded, 
                attributes = attributes, 
                attributes_columns = attributes_columns,
                file_name = dir_name + "/plots/epoch_" + str(epoch) + ".png",
                nb_images = 10
                )
        
        epoch_loss /= len(data_loader)
        pbar.close()
    
    current_time = time.time() - start_time
    
    # write the end of the training in the log file
    writer.add("Description", "End training : "+ str(time.time()) + "\n" +
                    "#"*50 + "\n" +
                    "The training took : " + str(current_time//3600) + " hours, " + str(current_time%3600//60) + " minutes, " + str(round(current_time%60)) + " seconds\n")
    
    return dir_name
    

if __name__ == "__main__":
    # initialize the gpu if available as material acceleration
    GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {GPU}")
    
    # initialize the models
    ae = AutoEncoder()
    dis = Discriminator()
    
    ae.to(GPU)
    dis.to(GPU)
    
    # load model
    # ae.load_state_dict(torch.load("Models/autoencoder_allnight.pt"))
    # dis.load_state_dict(torch.load("Models/discriminator_allnight.pt"))
    
    # initialize the dataset and the data loader
    dataset = get_celeba_dataset()
    
    train_set, validation_set, test_set = train_validation_test_split(dataset, train_split = 0.002, test_split = 0.998, val_split = 0., shuffle = True)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size = 15, shuffle = True)
    
    # train the models
    log_dir_name = train_loop(
        n_epochs = 5, 
        device = GPU, 
        autoencoder = ae, 
        discriminator = dis, 
        data_loader = train_data_loader,
        log_directory = "Logs",
        attributes_columns=dataset.attributes_df.columns[1:]
        )

    # save the model
    torch.save(ae.state_dict(), log_dir_name + "/autoencoder.pt")
    torch.save(dis.state_dict(), log_dir_name + "/discriminator.pt")
    
    # plot the losses
    save_plot_losses(
        list_files=[
            log_dir_name + "/Reconstruction_objective.txt",
            log_dir_name + "/Adversarial_objective.txt",
            log_dir_name + "/Discriminator_objective.txt"
            ],
        file_name=log_dir_name + "/plots/losses.png",
        xlabel="batch",
        ylabel="loss",
        title="Losses for " + log_dir_name.split("/")[-1],
    )
