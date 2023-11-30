import numpy as np
import pytorch

# This first code is an introduction with the parameters without linked to the other code
 


 def eval_reconstruction_loss(data, param, auto_encoder): # data: Our data set   #param : Wich include the batch_size,
        """
        Compute the loss 
        """
        auto_encoder.eval()  # set the auto_encoder to evaluation mode.
        batch_size= param.batch_size  

        eval_reconstruction_loss = []     
        for i in range(0, len(data), batch_size): # for each iteration, we processes a new batch of data.
            batch_x, batch_y = data.evaluate(i, i + batch_size) # i = the begining of the batch i+batch_size= end of the batch
            outputs = auto_encoder(batch_x, batch_y) # We save for every iteration the outputs of our encoder
            loss.append((outputs[-1] - batch_x) ** 2)  #MLE loss

        return np.mean(loss)  # fuction output the mean of the loss
    
    