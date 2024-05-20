"""
This file is used to train the models.
It is designed to be run from the root directory of the project.
Everything coulb be changed in this file, but it is not recommended.
"""

if __name__ == "__main__":
    
    # %% IMPORTS
    
    import torch
    from src.AutoEncoder import AutoEncoder
    from src.Discriminator import Discriminator
    from src.ImgDataset import get_celeba_dataset, get_celeba_dataset_lite
    from src.training_loop import train_loop
    from src.utils.plots import save_plot_losses
    from src.utils.train_validation_test_split import train_validation_test_split
    
    # %% INITIALIZATION
    
    # initialize the gpu if available as material acceleration
    GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {GPU}")
    
    # initialize the models
    ae = AutoEncoder()
    dis = Discriminator()
    
    ae.to(GPU)
    dis.to(GPU)
    
    # load model (UNCOMMENT TO LOAD A MODEL)
    # It's recommended to add manually a note saying which model is loaded in Logs/start_date_and_hour_log/Description.txt
    # ae.load_state_dict(torch.load("Logs/start_2023_12_16_13-56-39_logs/autoencoder.pt"))
    # dis.load_state_dict(torch.load("Logs/start_2023_12_16_13-56-39_logs/discriminator.pt"))
    
    # initialize the dataset and the data loader (use get_celeba_dataset_lite() for a smaller dataset)
    dataset = get_celeba_dataset_lite()
    
    train_set, validation_set, test_set = train_validation_test_split(dataset, train_split = 0.5, test_split = 0.5, val_split = 0., shuffle = False)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size = 15, shuffle = True)
    
    # %% TRAINING
    
    log_dir_name = train_loop(
        n_epochs = 25, 
        device = GPU, 
        autoencoder = ae, 
        discriminator = dis, 
        data_loader = train_data_loader,
        log_directory = "Logs",
        attributes_columns=dataset.attributes_df.columns[1:]
        )

    # %% SAVE MODEL AND PLOTS
    
    # save the model
    torch.save(ae.state_dict(), log_dir_name + "/autoencoder.pt")
    torch.save(dis.state_dict(), log_dir_name + "/discriminator.pt")
    
    # plot the losses
    save_plot_losses(
        list_files=[
            log_dir_name + "/Reconstruction_objective.txt",
            log_dir_name + "/Adversarial_objective.txt",
            log_dir_name + "/Discriminator_objective.txt"
            ],
        file_name=log_dir_name + "/plots/losses.png",
        xlabel="batch",
        ylabel="loss",
        title="Losses for " + log_dir_name.split("/")[-1],
    )
