import torch
from AutoEncoder import AutoEncoder
from Discriminator import Discriminator

def inference(
    autoencoder:AutoEncoder,
    discriminator:Discriminator,
    scaled_image:torch.tensor,
    attributes:torch.tensor,
    device:torch.device
    ) -> torch.tensor:
    """
    Inference function for the autoencoder and the discriminator
    """
    # send model, images and attributes to the device ( GPU if available )
    autoencoder.to(device)
    discriminator.to(device)
    scaled_image, attributes = scaled_image.to(device), attributes.to(device)    

    # Generate the latent space and the decoded images (outputs from the autoencoder)
    latent, decoded = autoencoder(scaled_image, attributes)

    # Generate the prediction of the discriminator
    y_pred = discriminator(latent)

    return decoded, y_pred

decoded, y_pred = inference(
    autoencoder=AutoEncoder(),
    discriminator=Discriminator(),
    scaled_image=torch.rand((1, 3, 256, 256)),
    attributes=torch.rand((1, 40)),
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
assert decoded.shape == (1, 3, 256, 256), "The inference function does not work properly. Shape issue for decoded"
assert y_pred.shape == (1, 40), "The inference function does not work properly. Shape issue for y_pred."

if __name__ == '__main__':
    print("Every assertion test passed")
    
    print("Let's use the inference function to make inference on a random image from the CelebA dataset")
    
    from ImgDataset import get_celeba_dataset, ImgDataset
    from utils import transform_img_for_celeba
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    
    # %% DATA LOADING
    if False:
        dataset = get_celeba_dataset()    
    else:        
        dataset = ImgDataset(attributes_csv_file="Data/Anno/list_attr_etu.txt",
                             images_dir="Data/Img_etu/",
                             transform=transform_img_for_celeba)
        
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # %% GET N RANDOM IMAGES
    
    N = 5
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
    
    directory = "Models/"
    # directory = "Logs/start_2023_12_15_11-17-28_logs/"
    
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(directory + "autoencoder_s.pt", map_location=device))
    
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(directory + "discriminator_s.pt", map_location=device))
    
    # %% 
    val_min = -5
    val_max = 5
    nb_steps = 10
    
    # change the attribute num n_attr
    n_attr = 21
    attr_name = dataset.attributes_df.columns[n_attr+1]
    
    # %% INFERENCE
    
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
            ax[i, j+1].set_title("{} : {}".format(attr_name, round(new_attributes[0, n_attr].item(), 2)), fontsize=7)
    
    fig.suptitle("Attribute {} : {} from 0 to 1 on {} images".format(n_attr, attr_name, N), fontsize=16)
    plt.show()
    
    print("End of the inference")
    
    plt.savefig(directory+"attribute_{}.png".format(n_attr), dpi=300, bbox_inches='tight')
    
        
        
       
    