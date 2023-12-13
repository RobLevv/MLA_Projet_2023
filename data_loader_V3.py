import os
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
from torchvision.transforms.functional import crop, resize


def transform_img_for_celeba(sample:dict, target_size:int=256) -> dict:
    """
    Crop the image to make it square and resize it to (target_size, target_size).
    CAREFUL: this function is specific to the CelebA dataset.
    """
    img = sample['image']
    IMG_SIZE = (178, 218, 3)
    # Crop the image to make it square
    img = crop(img, 40, 0, IMG_SIZE[0], IMG_SIZE[0])
    img = resize(img, (target_size, target_size), antialias=True)
    return {'image': img, 'attributes': sample['attributes'], 'image_name': sample['image_name']}


class ImgDataset(Dataset):
    """General Image dataset."""

    def __init__(self, attributes_csv_file, img_root_dir, transform=transform_img_for_celeba):
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
        # normalize the images between 0 and 1 (instead of 0 and 255) to avoid overflow in the loss function:
        image = image.float() / 255.
        attributes = self.attributes_df.iloc[idx, 1:]
        attributes = np.array([attributes], dtype=float).reshape(-1)
        # replace -1 by 0
        attributes[attributes == -1] = 0
        
        sample = {'image': image, 'attributes': attributes, 'image_name': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample



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




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def main1():
        """Test the ImgDataset class."""
        
        print("DATASET TEST\n")
        
        dataset = ImgDataset(attributes_csv_file='data/Anno/list_attr_celeba.txt', img_root_dir='data/Img')

        print(len(dataset))
        
        sample0 = dataset[0]
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
    
    
    def main2():
        """Test the ImgDataset class with a dataloader."""
        # OR with a dataloader
        print("\nWith a dataloader:\n")
        dataset = ImgDataset(attributes_csv_file='data/Anno/list_attr_celeba.txt', img_root_dir='data/Img')
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        for i, sample in enumerate(dataloader):
            images, attributes = sample['image'], sample['attributes']
            print("Batch {}:".format(i))
            print("Images shape:", images.shape)
            print("Attributes shape:", attributes.shape)
            break
            
    
    def main3():
        """Test the train_validation_test_split function."""
        dataset = ImgDataset(attributes_csv_file='data/Anno/list_attr_celeba.txt', img_root_dir='data/Img')
    
        train_dataset, val_dataset, test_dataset = train_validation_test_split(dataset, shuffle=True, display=True)
        
        # let's explore the datasets
        print("TRAIN DATASET")
        print("  len:", len(train_dataset))
        print("  sample 0 attributes:", train_dataset[0]['attributes'])
        print('  image name:', train_dataset[0]['image_name'])
        
        print("VAL DATASET")
        print("  len:", len(val_dataset))
        print("  sample 0 attributes:", val_dataset[0]['attributes'])
        print('  image name:', val_dataset[0]['image_name'])
        
        print("TEST DATASET")
        print("  len:", len(test_dataset))
        print("  sample 0 attributes:", test_dataset[0]['attributes'])
        print('  image name:', test_dataset[0]['image_name'])
        
        plt.figure(figsize=(5, 5))
        plt.pie(
            [len(train_dataset), len(val_dataset), len(test_dataset)],
            labels=["Train", "Validation", "Test"], 
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 14}
        )
        plt.title("Partition Distribution")
        plt.show()
    
    
    ## MAIN ##
    
    main3()
