import time
import torch
import tqdm
from src.AutoEncoder import AutoEncoder
from src.Discriminator import Discriminator
from src.Logger import Logger
from src.objectives import adversarial_objective, discriminator_objective, reconstruction_objective
from src.utils.plots import save_plot_images_comparision


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
    
    lambda_dis = 0
    
    # loop over the epochs
    for epoch in range(n_epochs):
        
        # initialize epoch loss
        epoch_loss = 0.
        
        current_time = time.time() - start_time
        
        print("\nEpoch : " + str(epoch + 1) + "/" + str(n_epochs) + " time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), ncols=100)
        
        for batch_nb, batch in enumerate(data_loader):
            
            pbar.update(1)
            
            # get the images and the attributes from the batch
            images, attributes = batch['image'], batch['attributes']
            
            # send model, images and attributes to the device ( GPU if available )
            images, attributes = images.to(device), attributes.to(device)    
            
            # normalize the images between 0 and 1 (instead of 0 and 255) to avoid overflow in the loss function
            images = images.float() / 255.
            
            # Generate the latent space and the decoded images (outputs from the autoencoder)
            latent, decoded = autoencoder(images, attributes)
            
            # Generate the prediction of the discriminator
            y_pred = discriminator(latent)
            
            # Update the Encoder and Decoder weights
            lambda_dis += 0.0001/500000
            loss_autoencoder = adversarial_objective(images, decoded, attributes, y_pred, lambda_dis) # TODO: change lambda
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