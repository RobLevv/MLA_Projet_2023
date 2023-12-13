import torch

def reconstruction_objective(x, x_recon):
    """
    x: image
    x_recon: reconstructed image
    """
    return torch.nn.functional.mse_loss(x.float(), x_recon).float()


def discriminator_objective(y, y_pred):
    """
    y: attributes
    y_discriminated: predicted attributes
    """
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y).float()


def adversarial_objective(x, x_recon, y, y_discriminated, lambda_ae=0.9):
    """
    x: image
    x_recon: reconstructed image
    y: attributes
    y_discriminated: predicted attributes
    lambda_ae: lambda auto-encoder parameter
    """
    return reconstruction_objective(x, x_recon) - lambda_ae * discriminator_objective(1 - y, y_discriminated)