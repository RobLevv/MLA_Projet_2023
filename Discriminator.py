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
        
    