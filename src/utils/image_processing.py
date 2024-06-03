"""This module contains functions to process images."""

import os
import torch
from torchvision.transforms.functional import crop, resize
from torchvision.io import read_image
from torchvision.utils import save_image
import tqdm

IMG_SIZE = (178, 218, 3)

def transform_img_for_celeba(
    image: torch.tensor, target_size: int = 256
) -> torch.tensor:
    """
    Crop the image to make it square and resize it to (target_size, target_size).
    Also normalize the images from 0-255 to 0-1 to avoid overflow in the loss function.
    CAREFUL: this function is specific to the CelebA dataset.

    Args:
        image (torch.tensor): The input image tensor.
        target_size (int, optional): The target size for the transformed image. Defaults to 256.

    Returns:
        torch.tensor: The transformed image tensor.
    """
    # Crop the image to make it square
    image = crop(image, 40, 0, IMG_SIZE[0], IMG_SIZE[0])
    image = resize(image, (target_size, target_size), antialias=True)
    image = image / 255.0
    # since this function is called when building the dataset, we don't need to normalize the images
    # when read_image is called, the images are tensor with values between 0 and 255
    # even if normalized between 0 and 1 previously
    return image.float()


# test the function
assert transform_img_for_celeba(torch.rand((3, 218, 178)), target_size=121).shape == (
    3,
    121,
    121,
), "The transform_img_for_celeba function does not work properly. Shape issue."


def transform_sample_for_celeba(sample: dict, target_size: int = 256) -> dict:
    """
    Apply the transform_img_for_celeba function to a sample.

    Args:
        sample (dict): A dictionary containing the sample data.
        target_size (int, optional): The desired size of the transformed image. Defaults to 256.

    Returns:
        dict: A dictionary containing the transformed sample data.
    """
    image = transform_img_for_celeba(sample["image"], target_size=target_size)
    return {
        "image": image,
        "attributes": sample["attributes"],
        "image_name": sample["image_name"],
    }


# test the function
assert transform_sample_for_celeba(
    {
        "image": torch.rand((3, 218, 178)),
        "attributes": torch.rand(40),
        "image_name": "test",
    }
).keys() == {
    "image",
    "attributes",
    "image_name",
}, "The transform_sample_for_celeba function does not work properly. Keys issue."


def build_img_processed_folder(
    origin_folder: str = "data/Img_lite", target_folder: str = "data/Img_processed_lite"
):
    """
    Takes everything in the specified origin_folder and applies the transform_img_for_celeba
    function to each image.

    Args:
        origin_folder (str): The path to the folder containing the original images.
        Defaults to "data/Img_lite".
        target_folder (str): The path to the folder where the processed images will be saved.
        Defaults to "data/Img_processed_lite".
    """

    # create the folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # files if .jpg or .png
    files = [
        file
        for file in os.listdir(origin_folder)
        if file.endswith(".jpg") or file.endswith(".png")
    ]
    # sort the files
    files.sort()

    # apply the transformation to all the files
    for file in tqdm.tqdm(files, desc="Transforming images", unit="image"):
        img = read_image(origin_folder + "/" + file)
        img = transform_img_for_celeba(img)
        save_image(img, target_folder + "/" + file)


if __name__ == "__main__":
    print(
        "Using build_Img_processed_folder to build the processed version of a target folder"
    )

    build_img_processed_folder(
        origin_folder="data/Img_lite", target_folder="data/Img_processed_lite"
    )

    print("Done")
