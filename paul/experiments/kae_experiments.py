import os
import datetime
import traceback
from autoenkoop import *
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import pydmd

SEED = 1337

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_linear_weights(model, lift_loader, loader):#, t_steps, latent_dim):
    Z = []#np.zeros((t_steps, latent_dim))
    latent_dim = model.latent_dimension
    with torch.no_grad():
        for x_t in lift_loader:
            x_t = x_t.to(DEVICE)
            #encoded current state
            z_t = model.encoder(x_t)
            
            Z.append(z_t.cpu().numpy())

    #num_train_examples = len(loader.dataset)
    #linearity_loss_epoch = 0.0
    #with torch.no_grad():
    #    for x_t, x_tp1 in loader:
    #        x_t = x_t.to(DEVICE)
    #        x_tp1 = x_tp1.to(DEVICE)
    #        batch_size = x_t.size(0)
    #        #encoded current state
    #        z_t = model.encoder(x_t)
    #        z_tp1 = model.encoder(x_tp1)
    #        pz_tp1 = model.linear_dynamics(z_t)
    #        
    #        linearity_loss_epoch += mse(z_tp1, pz_tp1).item() * batch_size
    #
    #linearity_loss_mean = linearity_loss_epoch / num_train_examples

    Z_all = np.vstack(Z).T

    Z0_all = Z_all[:,:-1]
    Z1_all = Z_all[:,1:]

    dmd = pydmd.DMD(-1, forward_backward=False)
    dmd.fit(Z_all)

    #Z0_all = np.vstack(Z0).T
    #Z1_all = np.vstack(Z1).T
    #K_star = Z1_all @ np.linalg.pinv(Z0_all)
    K_star = (dmd.modes @ np.diag(dmd.eigs) @ np.linalg.pinv(dmd.modes)).real
    with torch.no_grad():
        K = model.linear_dynamics.weight.detach().cpu().numpy()
    
    optimal_residual = Z1_all - K_star @ Z0_all
    model_residual = Z1_all - K @ Z0_all

    num_data = Z0_all.shape[1] * latent_dim
    

    true_latent_linearity_error = np.linalg.norm(optimal_residual)**2 / num_data
    total_linearity_error = np.linalg.norm(model_residual)**2 / num_data
    #print()
    #print(model_residual.shape)
    #print(num_data)
    #print(total_linearity_error)
    #print(linearity_loss_mean)
    #print(linearity_loss_mean/total_linearity_error)
    return K_star, Z_all, total_linearity_error, true_latent_linearity_error

def train_model(model, data_loader, lifting_data_loader, num_epochs):
    model.train()
    num_train_examples = len(data_loader.dataset)
    mse = nn.MSELoss()

    train_history = {}
    train_history["loss_reconstruction"] = np.zeros((num_epochs))
    train_history["loss_linearity"] = np.zeros((num_epochs))
    train_history["lr"] = np.zeros((num_epochs))
    train_history["lifted_states"] = np.zeros((num_epochs, len(lifting_data_loader.dataset), model.latent_dimension))
    train_history["K"] = np.zeros((num_epochs, model.latent_dimension, model.latent_dimension))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10, cooldown=5)
    
    epoch_progress = tqdm(range(num_epochs))

    for epoch in epoch_progress:
        reconstruction_loss_epoch = 0.0
        linearity_loss_epoch = 0.0

        lin_weights, Z, total_linearity_error, true_latent_linearity_error = calculate_linear_weights(model, lifting_data_loader, data_loader)
        lin_weights = torch.tensor(lin_weights, device=DEVICE, requires_grad=True)
        train_history["lifted_states"][epoch] = Z.T

        for x_t, x_t_plus_1 in data_loader:
            x_t, x_t_plus_1 = x_t.to(DEVICE), x_t_plus_1.to(DEVICE)
            batch_size = x_t.size(0)

            optimizer.zero_grad()

            #encoded current state
            z_t = model.encoder(x_t)
            #encoded next state
            z_t_plus_1 = model.encoder(x_t_plus_1)
            #decoded current state
            reconstructed_x_t = model.decoder(z_t)
            reconstructed_x_t_plus_1 = model.decoder(z_t_plus_1)
            #linear projected encoded next state
            if model_type == "dmd":
                pz_t_plus_1 = torch.nn.functional.linear(z_t, lin_weights)
            else:
                pz_t_plus_1 = model.linear_dynamics(z_t)
            #projected decoded next state

            ### RECONSTRUCTION LOSS ###
            loss_reconstruction = 0.5*(mse(x_t, reconstructed_x_t) + mse(x_t_plus_1, reconstructed_x_t_plus_1))

            ### LINEARITY LOSS ###
            loss_linearity = mse(pz_t_plus_1, z_t_plus_1)

            ### TOTAL LOSS ###
            if epoch >= RECON_EPOCH_THRESHOLD:
                #with torch.no_grad():
                #    reconstruction_weight = 0.9*loss_linearity.item() / loss_reconstruction.item()
                loss_total = loss_reconstruction + LAMBDA_LIN*loss_linearity
            else:
                loss_total = loss_reconstruction
            
            loss_total.backward()
            optimizer.step()

            reconstruction_loss_epoch += loss_reconstruction.item() * batch_size
            #prediction_loss_epoch += loss_prediction.item() * batch_size
            linearity_loss_epoch += loss_linearity.item() * batch_size

        reconstruction_loss_mean = reconstruction_loss_epoch / num_train_examples
        #prediction_loss_mean = prediction_loss_epoch / num_train_examples
        linearity_loss_mean = linearity_loss_epoch / num_train_examples

        total_loss_mean = reconstruction_loss_mean + LAMBDA_LIN*linearity_loss_mean# + prediction_loss_mean

        if epoch >= RECON_EPOCH_THRESHOLD:
            scheduler.step(total_loss_mean)

        train_history["loss_reconstruction"][epoch] = reconstruction_loss_epoch / num_train_examples
        #train_history["loss_prediction"][epoch] = prediction_loss_epoch / num_train_examples
        train_history["loss_linearity"][epoch] = linearity_loss_epoch / num_train_examples
        train_history["lr"][epoch] = scheduler.get_last_lr()[0]

        # eigenvalues
        with torch.no_grad():
            K = model.linear_dynamics.weight.detach().cpu().numpy()
            train_history["K"][epoch] = K


        linearity_estimation_error = total_linearity_error - true_latent_linearity_error
        estimation_error_portion = linearity_estimation_error/total_linearity_error


        epoch_progress.set_description("L(total): "+"{:.1e}".format(total_loss_mean)+", L(recon): "+"{:.1e}".format(reconstruction_loss_mean)+", L(lin): "+"{:.1e}".format(linearity_loss_mean)+", true_lin: "+"{:.1e}".format(true_latent_linearity_error)+", lin_est: "+"{:.1e}".format(linearity_estimation_error)+f"({estimation_error_portion*100:.0f}%), lr: " + "{:.1e}".format(scheduler.get_last_lr()[0]))
    return model, train_history
    
LATENT_DIMENSION = 128
EPOCHS = 300
RECON_EPOCH_THRESHOLD = 0
LR = 2e-4
LAMBDA_LIN = 1

model_type = "dmd"#["train", "dmd", "lrgroup"]

RUN_NAME = f"_L{LATENT_DIMENSION}_e{EPOCHS}_lr2e-4_lamlin{LAMBDA_LIN}_{model_type}"

if __name__ == '__main__':
    # making a new folder to save the script and the results 
    script_name = os.path.basename(__file__)[:-3] # remove the ".py"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.join(script_dir, "results", f"{script_name}", f"run_{timestamp}"+RUN_NAME)
    os.makedirs(folder_name, exist_ok=True)


    # storing the script
    with open(__file__, 'r') as file:
        script_content = file.read()
    with open(os.path.join(folder_name, "script_"+os.path.basename(__file__)), 'w') as file:
        file.write("\n"+"#"*100+"\n#\tTHIS SCRIPT HAS BEEN EXECUTED ALREADY\n#\tTHIS IS A COPY OF THE ORIGINAL SCRIPT\n#\tTHIS SCRIPT IS NOT MEANT TO BE EXECUTED AGAIN\n#\tIT EXISTS ONLY FOR THE PURPOSE OF GIVING CONTEXT TO THE DATA IN THIS FOLDER\n"+"#"*100+"\n\n"+script_content)


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

    with open(os.path.join(folder_name, "traceback.txt"), "w+") as tb_file:
        try:
            data_path = os.path.join(script_dir, "data", "T1492_x1151_y1_z127_c2.npz")
            center_dataset = True
            train_test_split = 0.7
            train_data, _, _ = load_and_prepare_data(data_path, center_dataset, train_test_split)

            dataset = TOffsetDataset(train_data, t_offset=1)
            loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

            lifting_dataset = SimpleDataset(train_data)
            lifting_loader = DataLoader(lifting_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

            model = ConvAutoencoderZeroDecoder(latent_dim=LATENT_DIMENSION).to(DEVICE)

            if model_type == "lrgroup":
                coder_params = [p for name, p in model.named_parameters() if 'coder' in name]
                koopman_params = [p for name, p in model.named_parameters() if 'linear' in name]
            
                optimizer = torch.optim.Adam([
                    {'params': koopman_params, 'lr': LR*2},
                    {'params': coder_params}
                ], lr=LR)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            trained_model, train_history = train_model(model, loader, lifting_loader, EPOCHS)

            torch.save(trained_model.state_dict(), os.path.join(folder_name, "model.pt"))
            
            history_keys = ["loss_reconstruction", "loss_linearity", "lr"]
            losses_history = np.stack([train_history[key] for key in history_keys], axis=1)
            pd.DataFrame(losses_history, columns=history_keys).to_csv(os.path.join(folder_name,"losses_history.csv"), index = False)

            np.save(os.path.join(folder_name,"lifted_state_history"), train_history["lifted_states"])
            np.save(os.path.join(folder_name,"K"), train_history["K"])
            #pd.DataFrame(train_history["lifted_states"].squeeze(-1)).to_csv(os.path.join(folder_name,"lifted_state_history.csv"), index = False)
            #pd.DataFrame(train_history["K"].squeeze(-1).squeeze(-1)).to_csv(os.path.join(folder_name,"K_history.csv"), index = False)

        except:
            traceback.print_exc(file=tb_file)
