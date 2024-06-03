import torch
from src.AutoEncoder import AutoEncoder
from src.Discriminator import Discriminator


def inference(
    autoencoder: AutoEncoder,
    discriminator: Discriminator,
    scaled_image: torch.tensor,
    attributes: torch.tensor,
    device: torch.device,
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
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
assert decoded.shape == (
    1,
    3,
    256,
    256,
), "The inference function does not work properly. Shape issue for decoded"
assert y_pred.shape == (
    1,
    40,
), "The inference function does not work properly. Shape issue for y_pred."
