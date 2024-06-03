import os


class Logger:
    """
    Class to log the training of the autoencoder and the discriminator.
    """

    def __init__(self, log_dir: str = "Logs", separator: str = "\n") -> None:
        self.log_dir = log_dir
        self.separator = separator
        # create the plots directory in the log directory
        os.makedirs("{}/plots".format(log_dir), exist_ok=True)

    def add(self, file: str, message: any) -> None:
        """
        Write the content of the dictionary in the log directory.
        """

        name_file = self.log_dir + "/{}.txt".format(file)

        # if the file already exists, we append the new values
        if os.path.isfile(name_file):
            with open(name_file, "a") as f:
                f.write(str(message) + self.separator)

        # otherwise we create the directory and the file
        else:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(name_file, "w") as f:
                f.write(str(message) + self.separator)
