import numpy as np
import os
import pandas as pd
import torch


from torchvision.io import read_image
from utils import transform_sample_for_celeba


def get_celeba_dataset():
    return ImgDataset(attributes_csv_file='data/Anno/list_attr_celeba.txt', img_root_dir='data/Img', transform=transform_sample_for_celeba)

def get_celeba_dataset_lite():
    return ImgDataset(attributes_csv_file='data/Anno/list_attr_celeba_lite.txt', img_root_dir='data/Img_lite', transform=transform_sample_for_celeba)

class ImgDataset(torch.utils.data.Dataset):
    """General Image dataset."""

    def __init__(self, 
                 attributes_csv_file:str, 
                 img_root_dir:str, 
                 transform):
        """
        Arguments:
            csv_file (string): Path to the csv file with attributes.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attributes_df = pd.read_csv(attributes_csv_file, sep=" ", header=1)
        self.root_dir = img_root_dir
        self.transform = transform
        
        assert len(self.attributes_df) == len(os.listdir(self.root_dir)), "The number of images and the number of attributes do not match."

    def __len__(self):
        return len(self.attributes_df)

    def __getitem__(self, idx):
        """
        Returns a sample of the dataset. {'image': image, 'attributes': attributes, 'image_name': image_name}
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.attributes_df.iloc[idx, 0])
        image = read_image(img_name)
        attributes = self.attributes_df.iloc[idx, 1:]
        attributes = np.array([attributes], dtype=float).reshape(-1)
        # replace -1 by 0
        attributes[attributes == -1] = 0
        
        sample = {'image': image, 'attributes': attributes, 'image_name': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    """Test the ImgDataset class."""
    
    print("DATASET TEST\n")
    
    dataset = get_celeba_dataset()

    print(len(dataset))
    
    sample0 = dataset[0]
    print("dataset:", dataset)
    im0, attr0 = sample0['image'], sample0['attributes']
    print("Image 0 shape:", im0.shape)
    print("Attributes 0 shape:", attr0.shape)
    
    N = 8
    
    fig, ax = plt.subplots(2, N, figsize=(1.7*N, 4))
    
    for idx in range(N):
        image, attributes = dataset[idx]['image'], dataset[idx]['attributes']
        ax[0, idx].imshow(image.permute(1, 2, 0))
        ax[0, idx].axis('off')
        ax[0, idx].set_title("Image {}".format(dataset.attributes_df.iloc[idx, 0]), fontsize=8)
        
        text = ""
        for i, attr in enumerate(attributes):
            if attr == 1:
                text += dataset.attributes_df.columns[i+1] + '\n'
                
        ax[1, idx].text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=8)
        ax[1, idx].axis('off')
    
    
    plt.tight_layout()
    plt.show()
    
