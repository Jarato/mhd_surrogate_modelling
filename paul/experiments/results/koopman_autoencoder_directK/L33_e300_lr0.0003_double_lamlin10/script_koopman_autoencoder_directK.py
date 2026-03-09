
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

def calculate_linear_weights(model, loader):#, t_steps, latent_dim):
    Z0 = []#np.zeros((t_steps, latent_dim))
    Z1 = []#np.zeros((t_steps, latent_dim))
    latent_dim = model.latent_dimension
    with torch.no_grad():
        for i, (x_t, x_t_plus_1) in enumerate(loader):
            x_t, x_t_plus_1 = x_t.to(DEVICE), x_t_plus_1.to(DEVICE)
            #encoded current state
            z_t = model.encoder(x_t)
            #encoded next state
            z_t_plus_1 = model.encoder(x_t_plus_1)
            
            Z0.append(z_t.cpu().numpy())
            Z1.append(z_t_plus_1.cpu().numpy())
    Z0_all = np.vstack(Z0).T
    Z1_all = np.vstack(Z1).T
    K = Z1_all @ np.linalg.pinv(Z0_all)
    return K

def train_model(kae_model, data_loader, num_epochs):
    kae_model.train()
    num_train_examples = len(data_loader.dataset)

    train_history = {}
    train_history["loss_reconstruction"] = np.zeros((num_epochs))
    #train_history["loss_prediction"] = np.zeros((num_epochs))
    train_history["loss_linearity"] = np.zeros((num_epochs))
    #train_history["eigenvalues"] = np.zeros((num_epochs, kae_model.latent_dimension), dtype=np.complex64)

    #steady_tolerance = 0.1
    epoch_progress = tqdm(range(num_epochs))

    for epoch in epoch_progress:
        reconstruction_loss_epoch = 0.0
        #prediction_loss_epoch = 0.0
        linearity_loss_epoch = 0.0

        lin_weights = torch.tensor(calculate_linear_weights(kae_model, data_loader), device=DEVICE)
        #print()
        #print(lin_weights.shape)
        #print()

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
            pz_t_plus_1 = torch.nn.functional.linear(z_t, lin_weights)
            #projected decoded next state
            #reconstructed_px_t_plus_1 = model.decoder(pz_t_plus_1)

            ### RECONSTRUCTION LOSS ###
            loss_reconstruction = 0.5*(mse(x_t, reconstructed_x_t) + mse(x_t_plus_1, reconstructed_x_t_plus_1))

            ### PREDICTION LOSS ###
            #loss_prediction = mse(reconstructed_px_t_plus_1, x_t_plus_1)

            ### LINEARITY LOSS ###
            loss_linearity = mse(pz_t_plus_1, z_t_plus_1)

            ### TOTAL LOSS ###
            loss_total = loss_reconstruction + 10*loss_linearity# + loss_prediction 

            loss_total.backward()
            optimizer.step()

            reconstruction_loss_epoch += loss_reconstruction.item() * batch_size
            #prediction_loss_epoch += loss_prediction.item() * batch_size
            linearity_loss_epoch += loss_linearity.item() * batch_size

        reconstruction_loss_mean = reconstruction_loss_epoch / num_train_examples
        #prediction_loss_mean = prediction_loss_epoch / num_train_examples
        linearity_loss_mean = linearity_loss_epoch / num_train_examples

        total_loss_mean = reconstruction_loss_mean + linearity_loss_mean# + prediction_loss_mean

        train_history["loss_reconstruction"][epoch] = reconstruction_loss_epoch / num_train_examples
        #train_history["loss_prediction"][epoch] = prediction_loss_epoch / num_train_examples
        train_history["loss_linearity"][epoch] = linearity_loss_epoch / num_train_examples

        # eigenvalues
        #with torch.no_grad():
        #    K = model.linear_dynamics.weight.detach().cpu().numpy()
        #eigvals, eigvecs = np.linalg.eig(K)

        #train_history["eigenvalues"][epoch] = eigvals

        #max_abs_eigenvalue = np.max(np.abs(eigvals))
        #mask = np.abs((np.abs(eigvals) - 1.0)) < steady_tolerance
        epoch_progress.set_description(f"Loss(total): {total_loss_mean:.4f}, Loss(recon): {reconstruction_loss_mean:.4f}, Loss(lin): {linearity_loss_mean:.6f}")
        #print(f"Epoch {epoch+1}/{EPOCHS}\tLoss(total): {total_loss_mean:.4f}\tLoss(recon): {reconstruction_loss_mean:.4f}\tLoss(pred): {prediction_loss_mean:.4f}\tLoss(lin): {linearity_loss_mean:.6f}\tMaxAbsEigenV: {max_abs_eigenvalue:.4f}\tSteadyModes: {sum(mask)}")
    
    return model, train_history
    
LATENT_DIMENSION = 33
EPOCHS = 300
LR = 3e-4

RUN_NAME = f"_L{LATENT_DIMENSION}_e{EPOCHS}_lr{LR}_double_lamlin10"

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

            dataset = TOffsetDataset(data, t_offset=1)
            loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

            model = ConvAutoencoderKoop(latent_dim=LATENT_DIMENSION).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            mse = nn.MSELoss()

            trained_model, train_history = train_model(model, loader, EPOCHS)

            torch.save(trained_model.state_dict(), os.path.join(folder_name, "model.pt"))
            
            losses_history = np.stack([train_history["loss_reconstruction"], train_history["loss_linearity"]], axis=1)
            pd.DataFrame(losses_history, columns=["loss_reconstruction", "loss_linearity"]).to_csv(os.path.join(folder_name,"losses_history.csv"), index = False)

            #pd.DataFrame(train_history["eigenvalues"]).to_csv(os.path.join(folder_name,"eigenvalues_history.csv"), index = False)

        except:
            traceback.print_exc(file=tb_file)
