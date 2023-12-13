import numpy as np
import torch

from AutoEncoder import AutoEncoder
from Discriminator import Discriminator
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import crop, resize


# %% IMAGE PROCESSING FUNCTIONS

def transform_img_for_celeba(image:torch.tensor, target_size:int=256) -> torch.tensor:
    """
    Crop the image to make it square and resize it to (target_size, target_size).
    Also normalize the images between 0 and 1 (instead of 0 and 255) to avoid overflow in the loss function.
    CAREFUL: this function is specific to the CelebA dataset.
    """
    IMG_SIZE = (178, 218, 3)
    # Crop the image to make it square
    image = crop(image, 40, 0, IMG_SIZE[0], IMG_SIZE[0])
    image = resize(image, (target_size, target_size), antialias=True)
    # normalize the images between 0 and 1 (instead of 0 and 255) to avoid overflow in the loss function:
    image = image.float() / 255.    # normalize the input
    return image


assert transform_img_for_celeba(torch.rand((3, 218, 178)), target_size=121).shape == (3, 121, 121), "The transform_img_for_celeba function does not work properly. Shape issue."
assert max(transform_img_for_celeba(torch.rand((3, 50, 12)) * 255, target_size=121).flatten()) <= 1, "The transform_img_for_celeba function does not work properly. Normalization issue."
assert min(transform_img_for_celeba(torch.rand((3, 50, 12)) * 255, target_size=121).flatten()) >= 0, "The transform_img_for_celeba function does not work properly. Normalization issue."


def transform_sample_for_celeba(sample:dict, target_size:int=256) -> dict:
    """
    Apply the transform_img_for_celeba function to a sample.
    """
    image = transform_img_for_celeba(sample['image'], target_size=target_size)
    return {'image': image, 'attributes': sample['attributes'], 'image_name': sample['image_name']}

assert transform_sample_for_celeba({'image': torch.rand((3, 218, 178)), 'attributes': torch.rand(40), 'image_name': 'test'}).keys() == {'image', 'attributes', 'image_name'}, "The transform_sample_for_celeba function does not work properly. Keys issue."


# %% INFERENCE FUNCTIONS

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

# %% TRAIN VALIDATION TEST SPLIT

def train_validation_test_split(
        dataset:Dataset,
        train_split:float=0.80349,
        test_split:float=0.0985296,
        val_split:float=0.0980607,
        shuffle:bool=False,
        display:bool=False
        ) -> tuple([Dataset,Dataset,Dataset]):
    """
    Split a dataset into a train, validation and test dataset.
    """
    # first assert that the splits ratios are correct
    assert int(100*(train_split + test_split + val_split)) == 100, "train_split + test_split + val_split must be equals to 1, not {}".format(train_split + test_split + val_split)

    # build the indices
    idx = list(range(len(dataset)))

    # compute the number of examples in each dataset
    nb_val = int(np.round(len(dataset)*val_split))
    nb_test = int(np.round(len(dataset)*test_split))
    nb_train = len(dataset) - (nb_val + nb_test)
    
    if display:
        print("nb_train:", nb_train)
        print("nb_val  :", nb_val)
        print("nb_test :", nb_test)

    if shuffle:
        np.random.shuffle(idx)
        
    train_dataset = Subset(dataset, idx[:nb_train])
    val_dataset= Subset(dataset, idx[nb_train:nb_train + nb_val])
    test_dataset= Subset(dataset, idx[nb_train + nb_val:])
    
    return train_dataset,val_dataset,test_dataset




# def main3():
#     """Test the train_validation_test_split function."""
#     dataset = ImgDataset(attributes_csv_file='data/Anno/list_attr_celeba.txt', img_root_dir='data/Img')

#     train_dataset, val_dataset, test_dataset = train_validation_test_split(dataset, shuffle=True, display=True)
    
#     # let's explore the datasets
#     print("TRAIN DATASET")
#     print("  len:", len(train_dataset))
#     print("  sample 0 attributes:", train_dataset[0]['attributes'])
#     print('  image name:', train_dataset[0]['image_name'])
    
#     print("VAL DATASET")
#     print("  len:", len(val_dataset))
#     print("  sample 0 attributes:", val_dataset[0]['attributes'])
#     print('  image name:', val_dataset[0]['image_name'])
    
#     print("TEST DATASET")
#     print("  len:", len(test_dataset))
#     print("  sample 0 attributes:", test_dataset[0]['attributes'])
#     print('  image name:', test_dataset[0]['image_name'])
    
#     plt.figure(figsize=(5, 5))
#     plt.pie(
#         [len(train_dataset), len(val_dataset), len(test_dataset)],
#         labels=["Train", "Validation", "Test"], 
#         autopct='%1.1f%%',
#         startangle=90,
#         textprops={'fontsize': 14}
#     )
#     plt.title("Partition Distribution")
#     plt.show()