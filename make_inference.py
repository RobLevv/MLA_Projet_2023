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
    
    # %% DATA LOADING
    dataset = get_celeba_dataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # %% LOAD MODELS
    
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load("Models/autoencoder.pt"))
    
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load("Models/discriminator.pt"))
    
    # %% INFERENCE
    for batch in data_loader:
        scaled_image, attributes = batch['image'], batch['attributes']
        break
        
    # same as attributes but one attribute is changed randomly
    new_attributes = attributes.clone()
    random_index = torch.randint(0, len(new_attributes[0]), (1,)).item()
    new_attributes[0][random_index] = 1 - new_attributes[0][random_index]
    
    
    decoded, y_pred = inference(
        autoencoder=autoencoder,
        discriminator=discriminator,
        scaled_image=scaled_image,
        attributes=attributes,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # %% PLOT
    # let's plot the original image, the decoded image and the attributes
    fig, ax = plt.subplots(1, 3, figsize = (20, 6))
    ax[0].imshow(scaled_image[0].permute(1, 2, 0))
    ax[0].axis('off')
    ax[0].set_title("Original image", fontsize=8)
    
    ax[1].imshow(decoded[0].detach().permute(1, 2, 0))
    ax[1].axis('off')
    ax[1].set_title("Decoded image", fontsize=8)
    
    text = ""
    for i, attr in enumerate(attributes[0]):
        if attr == 1:
            text += dataset.attributes_df.columns[i+1] + "\n"
    ax[2].text(0.5, 0.5, text[:-1], horizontalalignment='center', verticalalignment='center', fontsize=8)
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    