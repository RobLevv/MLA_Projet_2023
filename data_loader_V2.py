import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchvision
import os


def crop_for_celeba(img) -> tf.Tensor:
    """
    Crop the image to make it square.
    CAREFUL: this function is specific to the CelebA dataset.
    """
    IMG_SIZE = (178, 218, 3)
    # Crop the image to make it square
    img = torchvision.transforms.functional.crop(img, 40, 0, 178, 178)
    return img


def load_images(
    image_paths:list[str], 
    target_size:int=256, 
    display:bool=True,
    celeba:bool=True
    ) -> np.ndarray:
    """
    Load images from a list of paths.
    """
    # Load PNG images
    if display:
        print('Loading {} images from {}'.format(len(image_paths), '/'.join(image_paths[0].split('/')[:-1])))
    images = []
    for path in image_paths:
        img = torchvision.io.read_image(path)
        img = torch.tensor(img)
        if celeba:
            img = crop_for_celeba(img)
        img = torchvision.transforms.functional.resize(img, (target_size, target_size))
        images.append(img)

    images = np.array(images, dtype=np.uint8)

    return images


def load_attributes(attributes_file:str) -> pd.DataFrame:
    """
    Load attributes from the attributes file.
    """
    # Load attributes
    attributes = pd.read_csv(attributes_file, sep=" ", header=1)
    attributes.columns = ['image_id'] + list(attributes.columns[1:]) # rename the first column to 'index'
    attributes = attributes.replace(to_replace=-1, value=0) # replace -1 by 0
    return attributes


def load_dataset(
    nb_examples:int=1000, 
    images_folder:str='data/Img_lite', 
    attributes_file:str='data/list_attr_celeba.txt', 
    target_size:int=256,
    shuffle:bool=True,
    display:bool=False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load an image dataset with corresponding attributes.
    
    RETURNS:
        - images_df: a DataFrame containing the images
        - attributes_df: a DataFrame containing the attributes
    each row of the DataFrames corresponds to an image and its index is the image_id (the name of the image file)
    """
    
    # first load all the attributes
    attributes_df = load_attributes(attributes_file)
    
    N_MAX = len(attributes_df) # total number of images in the dataset
    
    assert nb_examples <= N_MAX, f'nb_examples should be less than {N_MAX}'
    
    
    # first generate an array of images id to load, randomly if shuffle is True, sequentially otherwise
    if shuffle:
        ids = np.random.choice(N_MAX, size=nb_examples, replace=False)
    else:
        ids = np.arange(nb_examples)
    
    # then build the list of image paths
    image_paths = []
    for id in ids:
        image_paths.append(images_folder + '/' + attributes_df.iloc[id]['image_id'])
        
    # load the images
    images = load_images(image_paths, target_size=target_size, display=display)
    
    images_ids = attributes_df['image_id']
    
    # building images_df
    if display:
        print("images_ids[ids]", images_ids[ids])
        print("images_ids[ids].shape", images_ids[ids].shape)
        print("images.shape", images.shape)
        print("type(images_ids[ids])", type(images_ids[ids]))
        print("type(images)", type(images))

    images_df = pd.DataFrame({"images":images.tolist()}, index=images_ids[ids])
    
    # select the attributes corresponding to the images
    attributes_df = attributes_df.iloc[ids]
    attributes_df.drop(columns=['image_id'], inplace=True)
    attributes_df.index = images_ids[ids]
    
    
    return images_df, attributes_df





if __name__ == "__main__":
    
    # We will now implement a solution to load N images and their corresponding attributes.
    
    images_df, attributes_df = load_dataset(
        nb_examples=80, 
        images_folder='data/Img', 
        attributes_file='data/Anno/list_attr_celeba.txt', 
        shuffle=True, # MUST BE FALSE WHEN USING A BIGGER ATTRIBUTES FILE THAN IMAGES FOLDER
        display=True
        )
    
    # Let's check the shape of the images and attributes
    print("images_df.shape", images_df.shape)
    print("attributes_df.shape", attributes_df.shape)
    
    # Let's display some images
    
    import matplotlib.pyplot as plt
    
    # plot 5 images
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    for i in range(5):
        axes[0][i].imshow(images_df["images"].iloc[i])
        axes[0][i].set_title(images_df.index[i])
        axes[0][i].axis('off')
        
        # display the attributes (only the ones that are True)
        attributes_txt = ''
        for attribute in attributes_df.columns:
            if attributes_df.iloc[i][attribute] == 1:
                attributes_txt += attribute + '\n'

        axes[1][i].text(0.5, 0.5, attributes_txt, horizontalalignment='center', verticalalignment='center')
        axes[1][i].axis('off')      
        
    plt.show()