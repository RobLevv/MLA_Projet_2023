import time
import torch
import tqdm
from src.AutoEncoder import AutoEncoder
from src.Discriminator import Discriminator
from src.Logger import Logger
from src.objectives import (
    adversarial_objective,
    discriminator_objective,
    reconstruction_objective,
)
from src.utils.plots import save_plot_images_comparision


def load_data(batch: dict, device: torch.device):
    """
    Load the images and the attributes from the batch and send them to the device
    """
    # get the images and the attributes from the batch
    images, attributes = batch["image"], batch["attributes"]

    # send model, images and attributes to the device ( GPU if available )
    images, attributes = images.to(device), attributes.to(device)

    # normalize the images between from 0-255 to 0-1 to avoid overflow in the loss function
    images = images.float() / 255.0

    return images, attributes


def autoencoder_step(
    images: torch.Tensor,
    attributes: torch.Tensor,
    autoencoder: AutoEncoder,
    discriminator: Discriminator,
    lambda_ae: float = 0.0,
):
    """
    Perform a step of the autoencoder.

    Args:
        images (torch.Tensor): The input images.
        attributes (torch.Tensor): The attributes associated with the images.
        autoencoder (AutoEncoder): The autoencoder model.
        discriminator (Discriminator): The discriminator model.
        lambda_ae (float, optional): The weight for the autoencoder loss. Defaults to 0.0.

    Returns:
        tuple: A tuple containing the loss of the autoencoder, the latent space representation,
        and the decoded images.
    """
    # Generate the latent space and the decoded images (outputs from the autoencoder)
    latent, decoded = autoencoder(images, attributes)

    # Generate the prediction of the discriminator
    y_pred = discriminator(latent)

    # Update the Encoder and Decoder weights
    loss_autoencoder = adversarial_objective(
        images, decoded, attributes, y_pred, lambda_ae=0.0
    )  # TODO: change lambda
    autoencoder.optimizer.zero_grad()
    loss_autoencoder.backward()
    autoencoder.optimizer.step()

    return loss_autoencoder, latent, decoded


def discriminator_step(
    latent: torch.Tensor,
    attributes: torch.Tensor,
    discriminator: Discriminator,
):
    """Perform a single step of the discriminator training.

    Args:
        latent (torch.Tensor): The latent space representation of the input.
        attributes (torch.Tensor): The target attributes for the input.
        discriminator (Discriminator): The discriminator model.

    Returns:
        torch.Tensor: The loss of the discriminator.
    """

    # Generate the prediction of the discriminator
    y_pred = discriminator(latent)

    # Update the Discriminator weights
    loss_discriminator = discriminator_objective(attributes, y_pred)
    discriminator.optimizer.zero_grad()
    loss_discriminator.backward()
    discriminator.optimizer.step()

    return loss_discriminator


def train_loop(
    n_epochs: int,
    device: torch.device,
    autoencoder: AutoEncoder,
    discriminator: Discriminator,
    data_loader: torch.utils.data.DataLoader,
    attributes_columns: list,
    log_directory: str = "Logs",
    plot_images: bool = True,
    verbose: bool = False,
) -> str:
    """
    Train loop for the autoencoder and the discriminator
    return the name of the log directory
    """

    start_time = time.time()

    dir_name = f"{log_directory}/start_{time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime(start_time))}_logs"

    writer = Logger(log_dir=dir_name, separator="\n")

    writer.add(
        "Description",
        f"Number of epochs : {n_epochs}\n"
        f"Size of training dataset : {len(data_loader)}\n"
        f"Batch size : {data_loader.batch_size}\n"
        f"Number of images in the training dataset : {len(data_loader) * data_loader.batch_size}\n"
        f"{'#' * 50}\n"
        f"Start training : {start_time}\n"
        f"{'#' * 50}\n",
    )

    writer.add("Description", "Autoencoder : \n" + str(autoencoder) + "\n")
    writer.add("Description", "Discriminator : \n" + str(discriminator) + "\n")

    # loop over the epochs
    for epoch in range(n_epochs):
        # initialize epoch loss
        epoch_loss = 0.0

        current_time = time.time() - start_time

        print(
            "\nEpoch : "
            + str(epoch + 1)
            + "/"
            + str(n_epochs)
            + " time: "
            + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        )

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), ncols=100)

        for _, batch in enumerate(data_loader):
            pbar.update(1)

            ## Load the data
            # get the images and the attributes from the batch
            images, attributes = batch["image"], batch["attributes"]

            # send model, images and attributes to the device ( GPU if available )
            images, attributes = images.to(device), attributes.to(device)

            # normalize the images between from 0-255 to 0-1 to avoid overflow in the loss function
            images = images.float() / 255.0

            ## Autoencoder step
            loss_autoencoder, latent, decoded = autoencoder_step(
                images, attributes, autoencoder, discriminator
            )

            # Detach the gradient computation from autoencoder to avoid backpropagating through it
            latent.detach_()

            ## Discriminator step
            loss_discriminator = discriminator_step(latent, attributes, discriminator)

            if verbose:
                # update epoch loss with the loss of the batch
                epoch_loss += loss_autoencoder.item()

                # write the losses in the log fileepoch_loss
                writer.add(
                    "Reconstruction_objective",
                    reconstruction_objective(images, decoded).item(),
                )
                writer.add("Adversarial_objective", loss_autoencoder.item())
                writer.add("Discriminator_objective", loss_discriminator.item())

        # Plot and save the images and the decoded images to compare
        if plot_images and (epoch % 5 == 0):
            save_plot_images_comparision(
                images=images,
                decoded=decoded,
                attributes=attributes,
                attributes_columns=attributes_columns,
                file_name=dir_name + "/plots/epoch_" + str(epoch) + ".png",
                nb_images=10,
            )

        epoch_loss /= len(data_loader)
        pbar.close()

    current_time = time.time() - start_time

    if verbose:
        # write the end of the training in the log file
        writer.add(
            "Description",
            f"End training : {time.time()}\n"
            f"{'#' * 50}\n"
            f"The training took : {current_time // 3600} hours, "
            f"{current_time % 3600 // 60} minutes, {round(current_time % 60)} seconds\n",
        )

    return dir_name
