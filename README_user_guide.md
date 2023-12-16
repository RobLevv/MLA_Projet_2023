# User Guide
## Introduction
This is a user guide for our FaderNetwork project.
It will guide you through the process of training and using the model.

## Setup

### Environment Libraries

The following libraries are required to run the code:

- Python 3
- PyTorch 2
- Tqdm
- Numpy
- Matplotlib
- Pandas

### Dataset

To train or infer the model, you will need to have a folder containing the dataset images and a csv file containing the attributes of the images.

#### Images

The images must be 256x256 pixels and must be in the RGB format. If you have images in another dimension, you can use the `build_Img_processed_folder` function in [image_processing.py](src/utils/image_processing.py) to resize them by executing the script.

```python
build_Img_processed_folder(origin_folder='path/to/origin/folder', destination_folder='path/to/destination/folder')
```

#### Attributes

The csv file must have the following format:

| description | | | | |
|-------------|-----|-----|-----|-----|
|            | `attribute_1` | `attribute_2` | ... | `attribute_n` |
|------------|-------------|-------------|-----|-------------|
| image_name | value_1 | value_2 | ... | value_n |

The first row may contain a description of the attributes, but it is not required.

The first column must contain the name of the image file ending with the extension.

The other columns must contain the values of the attributes for each image.


## Training

If you want to train the model, you can use the `train` script [train.py](train.py). You will probably need to modify some lines of the script to fit your needs.
```python
# initialize the dataset and the data loader (use get_celeba_dataset_lite() for a smaller dataset)
    dataset = get_celeba_dataset()
```
The `get_celeba_dataset` function will return a dataset containing all the images and attributes in the [data/Img_processed](data/Img_processed) folder and [data/Anno/list_attr_celeba.txt](data/Anno/list_attr_celeba.txt) file. If you want to use an other dataset, you can use the `ImgDataset` class in [ImgDataset.py](src/ImgDataset.py).

```python
from src.ImgDataset import ImgDataset
dataset = ImgDataset(attributes_csv_file='path/to/attributes/file', 
                     img_root_dir='path/to/images/folder', 
                     transform=f_img)
```

Then specify the split of the dataset for training, validation and test, the batch size and the number of epochs.


Everything about the training is stored in the [Logs](Logs) folder in a folder named with the current date and time. You can find the model weights, the objective evolutions and the images generated at the end of each epoch.

## Inference

If you want to use the model to generate images, you can use the `make_inference` script [make_inference.py](make_inference.py). You will probably need to modify some lines of the script to fit your needs.

Like training, you will need to specify the dataset but also the model weights to use. To do so, you can indicate the path to the Log folder concerned by the inference.

```python
directory = "Logs/start_2023_12_16_13-56-39_logs/" # Path to the Log folder
```

Then you can specify the attributes you want to use to generate the images. You can also specify the number of images to generate and the number of images per row in the generated image.

## Conclusion

We hope that this guide will help you to use our model. If you have any questions, feel free to contact us.