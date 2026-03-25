import torch
import os
import datetime
import traceback
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import pydmd

from autoenkoop import *

SEED = 1337

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_linear_weights(model, lift_loader, loader):#, t_steps, latent_dim):
    Z = []#np.zeros((t_steps, latent_dim))
    #mse = nn.MSELoss()
    latent_dim = model.latent_dimension
    model.eval()
    with torch.no_grad():
        for x_t in lift_loader:
            x_t = x_t.to(DEVICE)
            #encoded current state
            z_t = model.encoder(x_t)
            
            Z.append(z_t.cpu().numpy())

    #num_train_examples = len(loader.dataset)
    #print("num train examples in data_loader", num_train_examples)
    #linearity_loss_sum = 0.0
    #with torch.no_grad():
    #    for x_t, x_tp1 in loader:
    #        x_t, x_tp1 = x_t.to(DEVICE, non_blocking=True), x_tp1.to(DEVICE, non_blocking=True)
    #        torch.cuda.synchronize()
    #
    #        batch_size = x_t.size(0)
    #        #encoded current state
    #        z_t = model.encoder(x_t)
    #        z_tp1 = model.encoder(x_tp1)
    #        pz_tp1 = model.linear_dynamics(z_t)
    ##        
    #        linearity_loss_sum += mse(z_tp1, pz_tp1).item() * batch_size
    
    #linearity_loss_mean = linearity_loss_sum / num_train_examples

    Z_all = np.vstack(Z).T

    Z0_all = Z_all[:,:-1]
    Z1_all = Z_all[:,1:]

    dmd = pydmd.DMD(-1, forward_backward=False)
    dmd.fit(Z_all)

    K_star = (dmd.modes @ np.diag(dmd.eigs) @ np.linalg.pinv(dmd.modes)).real
    with torch.no_grad():
        K = model.linear_dynamics.weight.cpu().detach().numpy()

    optimal_residual = Z1_all - K_star @ Z0_all
    model_residual = Z1_all - K @ Z0_all

    num_data = Z0_all.shape[1] * latent_dim
    
    irreducible_linearity_error = np.linalg.norm(optimal_residual)**2 / num_data
    total_linearity_error = np.linalg.norm(model_residual)**2 / num_data

    model.train()
    return K_star, total_linearity_error, irreducible_linearity_error

def train_model(model, data_loader, lifting_data_loader, num_epochs):
    model.train()
    num_train_examples = len(data_loader.dataset)
    
    mse = nn.MSELoss()

    train_history = {}
    train_history["loss_reconstruction"] = np.zeros((num_epochs))
    train_history["loss_linearity"] = np.zeros((num_epochs))
    train_history["irreducible_lin_error"] = np.zeros((num_epochs))
    train_history["estimation_lin_error"] = np.zeros((num_epochs))
    train_history["lr"] = np.zeros((num_epochs))
    train_history["K"] = np.zeros((num_epochs, model.latent_dimension, model.latent_dimension))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, cooldown=5)
    
    epoch_progress = tqdm(range(num_epochs), dynamic_ncols=True)

    for epoch in epoch_progress:
        reconstruction_loss_epoch = 0.0
        linearity_loss_epoch = 0.0

        lin_weights, total_linearity_error, irreducible_linearity_error = calculate_linear_weights(model, lifting_data_loader, data_loader)
        lin_weights = torch.tensor(lin_weights, device=DEVICE, requires_grad=True)

        for x_t, x_t_plus_1 in data_loader:
            x_t, x_t_plus_1 = x_t.to(DEVICE, non_blocking=True), x_t_plus_1.to(DEVICE, non_blocking=True)
            torch.cuda.synchronize()
            batch_size = x_t.size(0)

            optimizer.zero_grad(set_to_none=True)

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
            linearity_loss_epoch += loss_linearity.item() * batch_size

        reconstruction_loss_mean = reconstruction_loss_epoch / num_train_examples
        linearity_loss_mean = linearity_loss_epoch / num_train_examples

        total_loss_mean = reconstruction_loss_mean + LAMBDA_LIN*linearity_loss_mean# + prediction_loss_mean

        if epoch >= RECON_EPOCH_THRESHOLD:
            scheduler.step(total_loss_mean)

        linearity_estimation_error = total_linearity_error - irreducible_linearity_error
        estimation_error_portion = linearity_estimation_error/total_linearity_error

        train_history["loss_reconstruction"][epoch] = reconstruction_loss_mean
        train_history["loss_linearity"][epoch] = linearity_loss_mean
        train_history["irreducible_lin_error"][epoch] = irreducible_linearity_error
        train_history["estimation_lin_error"][epoch] = linearity_estimation_error
        train_history["lr"][epoch] = scheduler.get_last_lr()[0]

        # eigenvalues
        with torch.no_grad():
            if model_type == "dmd":
                K = lin_weights
            else:
                K = model.linear_dynamics.weight.cpu().detach().numpy()
            train_history["K"][epoch] = K

        epoch_progress.set_description("L(total): "+"{:.1e}".format(total_loss_mean)+", L(recon): "+"{:.1e}".format(reconstruction_loss_mean)+", L(lin): "+"{:.1e}".format(linearity_loss_mean)+", true_lin: "+"{:.1e}".format(true_latent_linearity_error)+", lin_est: "+"{:.1e}".format(linearity_estimation_error)+f"({estimation_error_portion*100:.0f}%), lr: " + "{:.1e}".format(scheduler.get_last_lr()[0]))
    return model, train_history
    
LATENT_DIMENSION = 32 #[512, 128, 32]
EPOCHS = 300
#RECON_EPOCH_THRESHOLD = 5
LR = 2e-4
LAMBDA_LIN = 1 # [1, 10]
USEBIAS = True
NORMTYPE = "batch" #["batch", "layer"]
model_type = "train" #["train", "dmd", "lrgroup"]

use_bias_label = "bias" if USEBIAS else "nobias"
RUN_NAME = f"_L{LATENT_DIMENSION}_{use_bias_label}_norm{NORMTYPE}_lamlin{LAMBDA_LIN}_{model_type}"

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
            loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

            lifting_dataset = SimpleDataset(train_data)
            lifting_loader = DataLoader(lifting_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

            model = ConvAutoencoder(latent_dim=LATENT_DIMENSION, bias_terms=USEBIAS, norm_type=NORMTYPE).to(DEVICE)

            if model_type == "lrgroup":
                coder_params = [p for name, p in model.named_parameters() if 'coder' in name]
                koopman_params = [p for name, p in model.named_parameters() if 'linear' in name]
            
                optimizer = torch.optim.Adam([
                    {'params': coder_params, 'lr': LR},
                    {'params': koopman_params, 'lr': LR*2}
                ])
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            trained_model, train_history = train_model(model, loader, lifting_loader, EPOCHS)

            torch.save(trained_model.state_dict(), os.path.join(folder_name, "model.pt"))
            
            history_keys = ["loss_reconstruction", "loss_linearity", "irreducible_lin_error", "estimation_lin_error", "lr"]
            losses_history = np.stack([train_history[key] for key in history_keys], axis=1)
            pd.DataFrame(losses_history, columns=history_keys).to_csv(os.path.join(folder_name,"losses_history.csv"), index = False)

            np.save(os.path.join(folder_name,"K"), train_history["K"])

        except:
            traceback.print_exc(file=tb_file)
