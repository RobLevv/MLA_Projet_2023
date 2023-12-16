"""
This script is used to make inferences on images from datasets, then plot the results and save the plot.
It is designed to be run from the root directory of the project.
Everything coulb be changed in this file, but it is not recommended.
"""

if __name__ == '__main__':
    
    # %% IMPORTS
    
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    from src.AutoEncoder import AutoEncoder
    from src.Discriminator import Discriminator
    from src.ImgDataset import get_celeba_dataset, ImgDataset
    from src.inference import inference
    from torch.utils.data import DataLoader
    
    # %% DATA LOADING
    if False:
        dataset = get_celeba_dataset()
        print("Let's use the inference function to make inference on a random image from the CelebA dataset") 
    else:        
        dataset = ImgDataset(attributes_csv_file="data/Anno/list_attr_etu.txt",
                             img_root_dir="data/Img_etu",
                             transform=None)
        
        print("Let's use the inference function to make inference on a random image from the ETU dataset")
        
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # %% GET N RANDOM IMAGES
    
    N = 5
    if N > len(dataset):
        N = len(dataset)
    images = []
    attributes = []
    for i, sample in enumerate(data_loader):
        image, attribute = sample['image'], sample['attributes']
        if i >= N:
            break
        images.append(image)
        attributes.append(attribute)
        
    
    # %% LOAD MODELS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    directory = "Logs/start_2023_12_15_11-17-28_logs/"
    
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(directory + "autoencoder.pt", map_location=device))
    
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(directory + "discriminator.pt", map_location=device))
    
    # %% ATTRIBUTE TO CHANGE
    val_min = -5
    val_max = 5
    nb_steps = 10
    
    # change the attribute num n_attr
    n_attr = 21
    attr_name = dataset.attributes_df.columns[n_attr+1]
    
    # %% INFERENCE AND PLOT
    
    fig, ax = plt.subplots(N, nb_steps + 1, figsize=(1.7*(nb_steps), 4*N))    
    
    for i in range(N):
        
        scaled_image = images[i].clone()
        
        ax[i, 0].imshow(scaled_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
        ax[i, 0].axis('off')
        ax[i, 0].set_title("Original image")
        
        for j in range(nb_steps):
            # change the attributes of the image step by step going from 0 to 1 or from 1 to 0 depending on the value of the attribute
            new_attributes = attributes[i].clone()
            new_attributes[0, n_attr] = val_min + j * (val_max - val_min) / nb_steps
            
            # make inference
            decoded, y_pred = inference(
                autoencoder=autoencoder,
                discriminator=discriminator,
                scaled_image=scaled_image,
                attributes=new_attributes,
                device = device
                )
            
            decoded = decoded.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            
            ax[i, j+1].imshow(decoded / np.max(decoded))
            ax[i, j+1].axis('off')
            ax[i, j+1].set_title(str(round(new_attributes[0, n_attr].item(), 2)), fontsize=7)
    
    fig.suptitle("Attribute {} : {} from {} to {}".format(n_attr, attr_name, val_min, val_max), fontsize=16)
    
    print("End of the inference")
    
    plt.savefig(directory+"attribute_{}.png".format(attr_name), dpi=300, bbox_inches='tight')
    plt.show()