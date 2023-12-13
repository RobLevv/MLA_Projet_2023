import torch
from network_architectures import discriminator_layers

class Discriminator(torch.nn.Module):
    """
    Discriminator network
    The discriminator network aims at predicting the attributes from the latent space.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = discriminator_layers
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        
    def forward(self, x, display:bool = False) -> torch.Tensor:
        """
        x: latent space
        """
        # first convert x to float
        input = x.float()
        
        if display:
            print("input", input.shape, input.dtype)
            
        return self.discriminator(input)

y_pred = Discriminator()(torch.rand((1, 512, 2, 2)))

assert y_pred.shape == (1, 40), "The inference function does not work properly. Shape issue for y_pred."
assert max(y_pred.flatten()) <= 1 and min(y_pred.flatten()) >= 0, "The inference function does not work properly. Normalization issue. for y_pred (min = {}, max = {})".format(min(y_pred.flatten()), max(y_pred.flatten()))
