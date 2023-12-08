import torch
import numpy as np



def adverserial_loss(x,x_decoded,y_predict,lamb):

        # first check that x and x_decoded have the same shape
        assert x.shape == x_decoded.shape, "x and x_decoded must have the same shape"

        return torch.mean((x_decoded - x.float())**2-lamb*torch.mean(torch.log(1-y_predict)))

