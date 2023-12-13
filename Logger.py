import os

class Logger:
    """
    Class to log the training of the autoencoder and the discriminator.
    """
    def __init__(self, log_dir:str='Logs', separator:str="\n") -> None:
        self.content = dict()
        self.log_dir = log_dir
        self.separator = separator
    
    def add(self, key:str, value:any) -> None:
        """
        Add a key and a value to the content dictionary.
        """
        if key in self.content.keys():
            self.content[key] += self.separator + str(value)
        else:
            self.content[key] = str(value)
    
    def write(self) -> None:
        """
        Write the content of the dictionary in the log directory.
        """
        for key, value in self.content.items():
            
            if value == "":
                continue
            
            name_file = self.log_dir + "/{}.txt".format(key)
            
            # if the file already exists, we append the new values
            if os.path.isfile(name_file):
                with open(name_file, "a") as f:
                    f.write(str(value))
            
            # otherwise we create the directory and the file
            else:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(name_file, "w") as f:
                    f.write(str(value))
            
            # reset the content
            self.content[key] = ""