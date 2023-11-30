# In this script, we define the data loader for the dataset

#%% IMPORTS
import os
import numpy as np
import tensorflow as tf
import cv2

#%% GENERAL NOTES
# The data loader should take as input:
# 1. The path to the CelebA dataset
# 2. The path to the attribute file
# 3. The batch size
# 4. The image size
# 5. The number of attributes
# The data loader should return a tuple of two elements:
# 1. A batch of images of shape (batch_size, 256, 256, 3)
# 2. A batch of attribute vectors of shape (batch_size, 40)

#%% IMAGE OBJECT
class Image:
    def __init__(self, name:str, size:tuple, image:np.ndarray, attributes:np.ndarray):
        # Name of the file (e.g. 000001.jpg)
        self.name = name
        # Size of the image (e.g. (218, 178, 3)=>input, (256, 256, 3)=>output)
        self.size = size
        # Image data stored as a numpy array of shape (size[0], size[1], size[2])
        self.image = image
        # Attributes stored as a numpy array of shape (40,)
        self.attributes = attributes
        
    def resize(self, size:tuple) -> None:
        # Crop the image to a square of size (size, size) by removing the top
        cropped_image = self.image[-178:, :, :]
        # Resize the image to the desired size
        resized_image = cv2.resize(np.array(cropped_image), size, interpolation=cv2.INTER_AREA)
        # Update the size of the image
        self.size = resized_image.shape
        # Update the image data
        self.image = tf.convert_to_tensor(resized_image)
        

#%% GET DATA FROM DIRECTORY AND FILE
def get_images_from_directory(path_to_directory:str) -> np.ndarray:
    # Get all the images from the directory
    image_paths = os.listdir(path_to_directory)
    # Create an empty array to store the images
    images = np.zeros(len(image_paths), dtype=object)
    # Iterate over the images
    for i, image_path in enumerate(image_paths):
        # Get the name of the image
        image_name = image_path[-11:-4]
        # Get the size of the image
        image_size = tf.image.decode_jpeg(tf.io.read_file(os.path.join(path_to_directory, image_path))).shape
        # Get the image data
        image_data = tf.image.decode_jpeg(tf.io.read_file(os.path.join(path_to_directory, image_path)))
        # Get the attributes of the image
        image_attributes = np.zeros(40)
        # Create an image object
        image = Image(image_name, image_size, image_data, image_attributes)
        # Add the image to the array
        images[i] = image
    # Return the array of images
    return images

def get_attributes_from_file(path_to_file:str, images:np.ndarray) -> np.ndarray:
    # Get the attributes from the file
    attributes = np.genfromtxt(path_to_file, skip_header=2, usecols=range(1, 41), dtype=int)
    for i, image in enumerate(images):
        value = np.zeros((40, 2))
        for j, attribute in enumerate(attributes[i]):
            if attribute == 1:
                value[j] = np.array([1, 0])
            else:
                value[j] = np.array([0, 1])
        # Get the attributes of the image
        image.attributes = value
    # Return the array of images
    return images

#%% SPLIT DATA INTO TRAINING, VALIDATION AND EVALUATION SETS
def split_data(images:np.ndarray, training_percentage:float, validation_percentage:float, evaluation_percentage:float) -> tuple:
    # Get the number of images
    n_images = len(images)
    # Get the number of images for each set
    train_index = int(n_images * training_percentage)
    valid_index = int(n_images * validation_percentage + train_index)
    eval_index = int(n_images * evaluation_percentage + valid_index)
    # Shuffle the images
    np.random.shuffle(images)
    # Get the training images
    training_images = images[:train_index]
    # Get the validation images
    validation_images = images[train_index:valid_index]
    # Get the evaluation images
    evaluation_images = images[valid_index:eval_index]
    # Return the sets of images 
    return training_images, validation_images, evaluation_images

#%% TESTS
if __name__ == "__main__":
    # test the attribute loader
    images = np.array([Image("000001.jpg", (218, 178, 3), tf.image.decode_jpeg(tf.io.read_file(os.path.join("data/Img/000001.jpg"))), np.zeros(40))])
    # images = get_attributes_from_file("data/Anno/list_attr_celeba.txt", images)
    # print(images[0].attributes)
    # print(type(images[0].attributes))
    
    # print(type(images[0].attributes[0]))
    
    import matplotlib.pyplot as plt
    
    temp = images[0].image
    # plt.imshow(images[0].image)
    # plt.show()
    images[0].resize((256, 256))
    # plt.imshow(images[0].image)
    # plt.show()
    
    plt.subplot(1, 2, 1)
    plt.imshow(temp)
    plt.subplot(1, 2, 2)
    plt.imshow(images[0].image)
    plt.show()
    print(type(images[0].image),type(temp))

#%% OLD CODE
# #!/usr/bin/env python
# import os
# import matplotlib.image as mpimg
# import cv2
# import numpy as np
# import torch


# N_IMAGES = 202599
# IMG_SIZE = 256
# IMG_PATH = 'images_%i_%i.pth' % (IMG_SIZE, IMG_SIZE)
# ATTR_PATH = 'attributes.pth'


# def preprocess_images():

#     if os.path.isfile(IMG_PATH):
#         print("%s exists, nothing to do." % IMG_PATH)
#         return

#     print("Reading images from img_align_celeba/ ...")
#     raw_images = []
#     for i in range(1, N_IMAGES + 1):
#         if i % 10000 == 0:
#             print(i)
#         raw_images.append(mpimg.imread('img_align_celeba/%06i.jpg' % i)[20:-20])

#     if len(raw_images) != N_IMAGES:
#         raise Exception("Found %i images. Expected %i" % (len(raw_images), N_IMAGES))

#     print("Resizing images ...")
#     all_images = []
#     for i, image in enumerate(raw_images):
#         if i % 10000 == 0:
#             print(i)
#         assert image.shape == (178, 178, 3)
#         if IMG_SIZE < 178:
#             image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
#         elif IMG_SIZE > 178:
#             image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
#         assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
#         all_images.append(image)

#     data = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
#     data = torch.from_numpy(data)
#     assert data.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

#     print("Saving images to %s ..." % IMG_PATH)
#     torch.save(data[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
#     torch.save(data, IMG_PATH)


# def preprocess_attributes():

#     if os.path.isfile(ATTR_PATH):
#         print("%s exists, nothing to do." % ATTR_PATH)
#         return

#     attr_lines = [line.rstrip() for line in open('list_attr_celeba.txt', 'r')]
#     assert len(attr_lines) == N_IMAGES + 2

#     attr_keys = attr_lines[1].split()
#     attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in attr_keys}

#     for i, line in enumerate(attr_lines[2:]):
#         image_id = i + 1
#         split = line.split()
#         assert len(split) == 41
#         assert split[0] == ('%06i.jpg' % image_id)
#         assert all(x in ['-1', '1'] for x in split[1:])
#         for j, value in enumerate(split[1:]):
#             attributes[attr_keys[j]][i] = value == '1'

#     print("Saving attributes to %s ..." % ATTR_PATH)
#     torch.save(attributes, ATTR_PATH)


# preprocess_images()
# preprocess_attributes()
