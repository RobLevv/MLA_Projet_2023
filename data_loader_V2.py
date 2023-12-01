import numpy as np
import tensorflow as tf
import os

def load_images_from_directory(target_size:int, directory_path:str):
    """
    Load celebA dataset with PNG images.
    """
    # Load PNG images
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.jpg')]
    print('Loading {} images from {}'.format(len(image_paths), directory_path))
    images = []
    for path in image_paths:
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [target_size, target_size])
        images.append(img)

    images = np.array(images, dtype=np.uint8)

    return images

if name == "main":
    imgs_dataset = load_images_from_directory(256, 'data/Img_lite')
    print("Shape of the dataset: ", imgs_dataset.shape) # (202599, 256, 256, 3)
    print("Shape of the first image: ", imgs_dataset[0].shape) # (256, 256, 3)

    # Plot the first image
    import matplotlib.pyplot as plt
    plt.imshow(imgs_dataset[0])
    plt.show()