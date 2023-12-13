import torch
from network_architectures import encoder_layers, decoder_layers

class AutoEncoder(torch.nn.Module):
    """
    Autoencoder network
    The autoencoder network aims at encoding the image to a latent space and decoding the latent space to an image.
    The latent space should be invariant to the attributes.
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder_layers
        self.decoder = decoder_layers
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        
    def forward(self, x, y, display:bool = False) -> (torch.Tensor, torch.Tensor):
        """
        x: image
        y: attributes
        """
        # first convert x and y to float
        input_x = x.float()
        input_y = y.float()
        
        # encode x to latent space
        latent = self.encoder(input_x)
        
        # expand y to match latent space dimensions (2, 2)
        input_y = input_y.unsqueeze(2).unsqueeze(3)
        input_y = input_y.expand(y.shape[0], y.shape[1], 2, 2)       
         
        # concatenate latent and input_y along the channel dimension for the decoder to be able to process the y (attributes)
        latent_y = torch.cat((latent, input_y), dim = 1)
        
        # decode latent_y to output (image)
        decoded = self.decoder(latent_y)
        decoded = decoded/max(decoded.flatten())    #normalize the output
        
        if display: 
            print("input_x", input_x.shape, input_x.dtype)
            print("input_y", input_y.shape, input_y.dtype)
            print("latent", latent.shape, latent.dtype)
            print("latent_y", latent_y.shape, latent_y.dtype)
            print("decoded", decoded.shape, decoded.dtype)
            
        return latent, decoded

# %% TESTS

latent, decoded = AutoEncoder()(torch.rand((1, 3, 256, 256)), torch.rand((1, 40)))

assert latent.shape == (1, 512, 2, 2), "The inference function does not work properly. Shape issue for latent"
assert decoded.shape == (1, 3, 256, 256), "The inference function does not work properly. Shape issue for decoded"
assert max(decoded.flatten()) <= 1 and min(decoded.flatten()) >= 0, "The inference function does not work properly. Normalization issue. for decoded (min = {}, max = {})".format(min(decoded.flatten()), max(decoded.flatten()))
