import matplotlib.pyplot as plt
import numpy as np
import torch


def save_plot_images_comparision(
    images:torch.tensor,
    decoded:torch.tensor,
    attributes:torch.tensor,
    attributes_columns:list,
    file_name:str,
    nb_images:int=10
    ) -> None:
    """
    Plot the images and the decoded images to compare.
    """
    
    if images.shape[0] < nb_images:
        nb_images = images.shape[0]
    
    fig, ax = plt.subplots(3, nb_images, figsize = (20, 6))
    
    if nb_images == 1:
        fig, ax = plt.subplots(3, 2, figsize = (20, 6))
    
    for i in range(nb_images):
        image = images[i].detach().cpu().numpy()
        decoded_image = decoded[i].detach().cpu().numpy() / np.max(decoded[i].detach().cpu().numpy())
        attributes_image = attributes[i].detach().cpu().numpy()
        
        # first raw: original image
        ax[0,i].imshow(np.transpose(image, (1,2,0)))
        ax[0,i].axis('off')
        ax[0,i].set_title('Original')

        # second raw: attributes
        text = ""
        for j, attr in enumerate(attributes_image):
            if attr == 1:
                text += attributes_columns[j] + "\n"  
        ax[1,i].text(0.5, 0.5, text[:-1], horizontalalignment='center', verticalalignment='center', fontsize=8)
        ax[1,i].axis('off')
        
        # third raw: decoded image
        ax[2,i].imshow(np.transpose(decoded_image, (1,2,0)))
        ax[2,i].axis('off')
        ax[2,i].set_title('Decoded')
        
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def save_plot_losses(
    list_files:list,
    file_name:str,
    title:str="Training losses",
    xlabel:str="Batch",
    ylabel:str="Losses",
    sep:str='\n',
    ) -> None:
    """
    Plot the losses.
    """
    # first read the files to get the losses
    for i, file in enumerate(list_files):
        with open(file, 'r') as f:
            if i == 0:
                losses = np.array(f.read().split(sep)[:-1], dtype=float)
            else:
                losses = np.vstack((losses, np.array(f.read().split(sep)[:-1], dtype=float)))
    
    # then plot the losses
    plt.figure(figsize=(20, 10))
    for i, loss in enumerate(losses):
        plt.plot(loss, label=list_files[i].split('/')[-1].split('.')[0])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(file_name)
    plt.close()