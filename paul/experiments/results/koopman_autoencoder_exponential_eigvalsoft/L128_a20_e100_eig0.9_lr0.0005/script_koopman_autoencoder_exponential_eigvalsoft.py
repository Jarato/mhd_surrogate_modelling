
####################################################################################################
#	THIS SCRIPT HAS BEEN EXECUTED ALREADY
#	THIS IS A COPY OF THE ORIGINAL SCRIPT
#	THIS SCRIPT IS NOT MEANT TO BE EXECUTED AGAIN
#	IT EXISTS ONLY FOR THE PURPOSE OF GIVING CONTEXT TO THE DATA IN THIS FOLDER
####################################################################################################

import os
import datetime
import traceback
from autoenkoop import *
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_dynamic_weight_double(x_t, x_t_reconstructed, x_t_plus_1, x_t_plus_1_reconstructed, alpha = 10):
    with torch.no_grad():
        reconSq_diff_x_t = (x_t.cpu() - x_t_reconstructed.cpu())**2
        reconSq_diff_x_t_plus_1 = (x_t_plus_1.cpu() - x_t_plus_1_reconstructed.cpu())**2
        recon_mse_example_x_t = torch.mean(reconSq_diff_x_t,dim=[1,2,3])
        recon_mse_example_x_t_plus_1 = torch.mean(reconSq_diff_x_t_plus_1,dim=[1,2,3])
        recon_mse_example = recon_mse_example_x_t + recon_mse_example_x_t_plus_1
        dyn_weight_example = np.exp(-alpha*recon_mse_example.cpu().numpy())
    return dyn_weight_example

def calculate_dynamic_weight_single(x_t, x_t_reconstructed, alpha = 10):
    with torch.no_grad():
        reconSq_diff_x_t = (x_t - x_t_reconstructed)**2
        recon_mse_example_x_t = torch.mean(reconSq_diff_x_t,dim=[1,2,3])
        dyn_weight_example = torch.exp(-alpha*recon_mse_example_x_t)
    return dyn_weight_example

def train_model(kae_model, data_loader, num_epochs, alpha, eig_threshold):
    kae_model.train()
    num_train_examples = len(data_loader.dataset)

    train_history = {}
    train_history["loss_reconstruction"] = np.zeros((num_epochs))
    train_history["loss_prediction"] = np.zeros((num_epochs))
    train_history["loss_linearity"] = np.zeros((num_epochs))
    train_history["loss_eigvalue"] = np.zeros((num_epochs))
    train_history["lin_dyn_weight"] = np.zeros((num_epochs))
    train_history["eigenvalues"] = np.zeros((num_epochs, kae_model.latent_dimension), dtype=np.complex64)

    steady_tolerance = 0.1
    epoch_progress = tqdm(range(num_epochs))

    for epoch in epoch_progress:
        reconstruction_loss_epoch = 0.0
        prediction_loss_epoch = 0.0
        linearity_loss_epoch = 0.0
        eigvalue_loss_epoch = 0.0
        lin_dyn_weight_epoch = 0.0

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
            #linear projected encoded next state
            pz_t_plus_1 = model.linear_dynamics(z_t)
            #projected decoded next state
            reconstructed_px_t_plus_1 = model.decoder(pz_t_plus_1)

            ### RECONSTRUCTION LOSS ###
            loss_reconstruction = mse(x_t, reconstructed_x_t)# + mse(x_t_plus_1, reconstructed_x_t_plus_1)
            lin_dyn_weight = calculate_dynamic_weight_single(x_t, reconstructed_x_t, alpha=alpha)
            lin_dyn_weight_epoch += sum(lin_dyn_weight.cpu().numpy())

            ### PREDICTION LOSS ###
            loss_prediction = mse(reconstructed_px_t_plus_1, x_t_plus_1)
            
            ### LINEARITY LOSS ###
            error_linearity = (pz_t_plus_1 - z_t_plus_1)**2
            loss_linearity = torch.mean(torch.mean(error_linearity,dim=[1]) * lin_dyn_weight)

            K = model.linear_dynamics.weight
            abs_eigenvalues = torch.abs(torch.linalg.eigvals(K))
            loss_eig = torch.mean(torch.abs(abs_eigenvalues[abs_eigenvalues > eig_threshold] - 1.0))

            ### TOTAL LOSS ###
            loss_total = loss_reconstruction + loss_prediction + loss_linearity + loss_eig

            loss_total.backward()
            optimizer.step()

            reconstruction_loss_epoch += loss_reconstruction.item() * batch_size
            prediction_loss_epoch += loss_prediction.item() * batch_size
            linearity_loss_epoch += loss_linearity.item() * batch_size
            eigvalue_loss_epoch += loss_eig.item() * batch_size

        reconstruction_loss_mean = reconstruction_loss_epoch / num_train_examples
        prediction_loss_mean = prediction_loss_epoch / num_train_examples
        linearity_loss_mean = linearity_loss_epoch / num_train_examples
        eigvalue_loss_mean = eigvalue_loss_epoch / num_train_examples

        total_loss_mean = reconstruction_loss_mean+prediction_loss_mean+linearity_loss_mean+eigvalue_loss_mean

        train_history["loss_reconstruction"][epoch] = reconstruction_loss_mean
        train_history["loss_prediction"][epoch] = prediction_loss_mean
        train_history["loss_linearity"][epoch] = linearity_loss_mean
        train_history["loss_eigvalue"][epoch] = eigvalue_loss_mean
        train_history["lin_dyn_weight"][epoch] = lin_dyn_weight_epoch / num_train_examples

        # eigenvalues
        with torch.no_grad():
            K = model.linear_dynamics.weight.detach().cpu().numpy()
        eigvals, eigvecs = np.linalg.eig(K)

        train_history["eigenvalues"][epoch] = eigvals

        max_abs_eigenvalue = np.max(np.abs(eigvals))
        mask = np.abs((np.abs(eigvals) - 1.0)) < steady_tolerance
        epoch_progress.set_description(f"Loss(recon): {reconstruction_loss_mean:.4f}, Loss(pred): {prediction_loss_mean:.4f}, Loss(lin): {linearity_loss_mean:.5f}, Loss(eigV): {eigvalue_loss_mean:.4f}, MaxAbsEigenV: {max_abs_eigenvalue:.4f}, SteadyModes: {sum(mask)}")
        #print(f"Epoch {epoch+1}/{EPOCHS}\tLoss(total): {total_loss_mean:.4f}\tLoss(recon): {reconstruction_loss_mean:.4f}\tLoss(pred): {prediction_loss_mean:.4f}\tLoss(lin): {linearity_loss_mean:.6f}\tMaxAbsEigenV: {max_abs_eigenvalue:.4f}\tSteadyModes: {sum(mask)}")
    
    return model, train_history
    
LATENT_DIMENSION = 128
ALPHA = 20
EPOCHS = 100
LR = 5e-4
EIG_THRESHOLD = 0.9
RUN_NAME = f"_L{LATENT_DIMENSION}_a{ALPHA}_e{EPOCHS}_eig{EIG_THRESHOLD}_lr{LR}"

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
            datafile = np.load(os.path.join(script_dir, "data", "T1492_x1151_y1_z127_c2.npz"))
            data = datafile['timeseries']

            #mean_velocity = np.mean(data, axis=(0,1,2))
            #data_centered = data - mean_velocity[None,None,None,:]

            #print(mean_velocity)

            #mean_removed_data = data - np.mean(data, axis=0)
            dataset = TOffsetDataset(data, t_offset=1)

            #dataset = TOffsetDataset(data, t_offset=1)
            loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2, pin_memory=True, pin_memory_device=DEVICE)

            model = ConvAutoencoder(latent_dim=LATENT_DIMENSION).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            mse = nn.MSELoss()

            trained_model, train_history = train_model(model, loader, EPOCHS, ALPHA, EIG_THRESHOLD)

            torch.save(trained_model.state_dict(), os.path.join(folder_name, "model.pt"))
            
            losses_history = np.stack([history[key] for key in train_history], axis=1)
            pd.DataFrame(losses_history, columns=[key for key in train_history]).to_csv(os.path.join(folder_name,"losses_history.csv"), index = False)

            pd.DataFrame(train_history["eigenvalues"]).to_csv(os.path.join(folder_name,"eigenvalues_history.csv"), index = False)

        except:
            traceback.print_exc(file=tb_file)
