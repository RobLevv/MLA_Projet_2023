import torch
from torchvision.transforms.functional import crop, resize


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
    # since this function is called when building the dataset, we don't need to normalize the images
    # when read_image is called, the images are tensor with values between 0 and 255 even if normalized between 0 and 1 previously
    return image.float()/255.0

# test the function
assert transform_img_for_celeba(torch.rand((3, 218, 178)), target_size=121).shape == (3, 121, 121), "The transform_img_for_celeba function does not work properly. Shape issue."


def transform_sample_for_celeba(sample:dict, target_size:int=256) -> dict:
    """
    Apply the transform_img_for_celeba function to a sample.
    """
    image = transform_img_for_celeba(sample['image'], target_size=target_size)
    return {'image': image, 'attributes': sample['attributes'], 'image_name': sample['image_name']}

# test the function
assert transform_sample_for_celeba({'image': torch.rand((3, 218, 178)), 'attributes': torch.rand(40), 'image_name': 'test'}).keys() == {'image', 'attributes', 'image_name'}, "The transform_sample_for_celeba function does not work properly. Keys issue."


def build_Img_processed_folder(
    origin_folder:str='data/Img_lite',
    target_folder:str='data/Img_processed_lite'
    ):
    """
    Takes Everything in data/Img and apply transform_img_for_celeba to it.
    """
    # imports
    import os
    from torchvision.io import read_image
    from torchvision.utils import save_image
    import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    
    # create the folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # files if .jpg or .png
    files = [file for file in os.listdir(origin_folder) if file.endswith('.jpg') or file.endswith('.png')]
    # sort the files
    files.sort()
    
    # apply the transformation to all the files
    for file in tqdm.tqdm(files, desc="Transforming images", unit="image"):
        img = read_image(origin_folder + '/' + file)
        img = transform_img_for_celeba(img)
        save_image(img, target_folder + '/' + file, normalize=True)
        


if __name__ == "__main__" :
    
    print("Using build_Img_processed_folder to build the processed version of a target folder")
    
    # build_Img_processed_folder(origin_folder='data/Img_lite',target_folder='data/Img_processed_lite')
    build_Img_processed_folder(origin_folder='C:/Users/Robin/Documents/GitHub/data_celebA/data/img_align_celeba',target_folder='data/Img_processed')
    
    print("Done")