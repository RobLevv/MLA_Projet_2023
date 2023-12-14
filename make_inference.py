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
    
    from ImgDataset import get_celeba_dataset
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    
    # %% DATA LOADING
    dataset = get_celeba_dataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # %% LOAD MODELS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load("Models/autoencoder_2023_12_14_11-46-38.pt", map_location=device))
    
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load("Models/discriminator_2023_12_14_11-46-38.pt", map_location=device))
    
    # %% GET AN IMAGE AND ITS ATTRIBUTES
    for batch in data_loader:
        scaled_image, attributes = batch['image'], batch['attributes']
        break
    
    # %% N ATTRIBUTES TO BE CHANGED (between 0 and 1)
    N = 5
    step = 0.2
    nb_steps = int(1//step) + 1
    assert N <= 40, "N must be smaller than 40"
    random_index = torch.randint(0, 40, (N,))
    
    new_attributes = attributes.clone()
    
    changed_attributes_names = dataset.attributes_df.columns[1:][random_index]
    
    # %% INFERENCE
    
    fig, ax = plt.subplots(N, nb_steps + 1, figsize=(1.7*(nb_steps), 4*N))    
    
    for i in range(N):
        
        ax[i, 0].imshow(scaled_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
        ax[i, 0].axis('off')
        ax[i, 0].set_title("Original image")
        
        for j in range(nb_steps):
            # change the attributes of the image step by step going from 0 to 1 or from 1 to 0 depending on the value of the attribute
            if attributes[0, random_index[i]] == 0:
                new_attributes[0, random_index[i]] = step * (j + 1)
            else:
                new_attributes[0, random_index[i]] = 1 - step * (j + 1)
            
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
            ax[i, j+1].set_title("{} : {}".format(changed_attributes_names[i], round(new_attributes[0, random_index[i]].item(), 2)))
    
    fig.suptitle("Attributes changed step by step", fontsize=16)
    plt.show()
    
    print("End of the inference")
        
        
       
    