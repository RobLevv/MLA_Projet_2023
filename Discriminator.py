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
    
    def loss(self, y, y_discriminated, version:int = 0):
        """
        y: attributes
        y_discriminated: predicted attributes
        version: 0 or 1
        """
        if version == 0: # MSE built-in loss
            assert y.shape == y_discriminated.shape, "y and y_discriminated must have the same shape"
            return torch.nn.functional.mse_loss(y.float(), y_discriminated).float()
        else:
            # Use torch.bernoulli to get binary values directly
            m = torch.nn.LogSoftmax(dim=1)
            criterion = torch.nn.NLLLoss()
            loss_discriminator = criterion(m(y_discriminated), y)
        return loss_discriminator
    