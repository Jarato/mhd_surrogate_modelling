
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

def train_model(kae_model, data_loader, num_epochs, n_steps, alpha):
    kae_model.train()
    num_train_examples = len(data_loader.dataset)

    train_history = {}
    train_history["loss_reconstruction"] = np.zeros((num_epochs))
    train_history["loss_list_prediction"] = np.zeros((num_epochs, n_steps-1))
    train_history["loss_list_linearity"] = np.zeros((num_epochs, n_steps-1))
    train_history["lin_dyn_weight"] = np.zeros((num_epochs))
    train_history["eigenvalues"] = np.zeros((num_epochs, kae_model.latent_dimension), dtype=np.complex64)

    steady_tolerance = 0.1
    epoch_progress = tqdm(range(num_epochs))

    for epoch in epoch_progress:
        reconstruction_loss_epoch = 0.0
        prediction_loss_epoch = np.zeros((n_steps-1))
        linearity_loss_epoch = np.zeros((n_steps-1))
        lin_dyn_weight_epoch = 0.0

        for X_N in data_loader:
            X_N = [x_t.to(DEVICE) for x_t in X_N]
            #x_t, x_t_plus_1 = x_t.to(DEVICE), x_t_plus_1.to(DEVICE)
            batch_size = X_N[0].size(0)

            optimizer.zero_grad()

            # encode states
            Z_N = [model.encoder(x_t) for x_t in X_N]

            # project states
            projected_Z_N = [Z_N[0]]
            for _ in range(len(Z_N)-1):
                projected_Z_N.append(model.linear_dynamics(projected_Z_N[-1]))

            # decode states
            decoded_X_N = [model.decoder(z_t) for z_t in projected_Z_N]
           
            ### RECONSTRUCTION LOSS ###
            lin_dyn_weight = calculate_dynamic_weight_single(X_N[0], decoded_X_N[0], alpha=alpha)
            lin_dyn_weight_epoch += sum(lin_dyn_weight.cpu().numpy())

            ### PREDICTION LOSS ###
            list_loss_prediction = [mse(X_N[i], decoded_X_N[i]) for i in range(len(X_N))]
            
            ### LINEARITY LOSS ###
            list_error_linearity = [(projected_Z_N[i] - Z_N[i])**2 for i in range(1, len(Z_N))]
            list_loss_linearity = [torch.mean(torch.mean(error_linearity,dim=[1]) * lin_dyn_weight) for error_linearity in list_error_linearity]

            loss_reconstruction = list_loss_prediction[0]
            loss_prediction = sum(list_loss_prediction[1:]) / (n_steps-1)
            loss_linearity = sum(list_loss_linearity) / (n_steps-1)

            ### TOTAL LOSS ###
            loss_total = loss_reconstruction + loss_prediction + loss_linearity

            loss_total.backward()
            optimizer.step()

            reconstruction_loss_epoch += loss_reconstruction.item() * batch_size
            prediction_loss_epoch += np.array([list_loss_prediction[i].detach().cpu().numpy() * batch_size for i in range(1, len(list_loss_prediction))])
            linearity_loss_epoch += np.array([loss_linearity.detach().cpu().numpy() * batch_size for loss_linearity in list_loss_linearity])

        reconstruction_loss_mean = reconstruction_loss_epoch / num_train_examples
        prediction_loss_mean = np.mean(prediction_loss_epoch) / num_train_examples
        linearity_loss_mean = np.mean(linearity_loss_epoch) / num_train_examples

        total_loss_mean = reconstruction_loss_mean+prediction_loss_mean+linearity_loss_mean

        train_history["loss_reconstruction"][epoch] = reconstruction_loss_epoch / num_train_examples
        train_history["loss_list_prediction"][epoch] = prediction_loss_epoch / num_train_examples
        train_history["loss_list_linearity"][epoch] = linearity_loss_epoch / num_train_examples
        train_history["lin_dyn_weight"][epoch] = lin_dyn_weight_epoch / num_train_examples

        # eigenvalues
        with torch.no_grad():
            K = model.linear_dynamics.weight.detach().cpu().numpy()
        eigvals, eigvecs = np.linalg.eig(K)

        train_history["eigenvalues"][epoch] = eigvals

        max_abs_eigenvalue = np.max(np.abs(eigvals))
        mask = np.abs((np.abs(eigvals) - 1.0)) < steady_tolerance
        epoch_progress.set_description(f"L(total): {total_loss_mean:.4f}, L(recon): {reconstruction_loss_mean:.4f}, L(pred): {prediction_loss_mean:.4f}, L(lin): {linearity_loss_mean:.6f}, MaxAbsEigV: {max_abs_eigenvalue:.4f}, StableModes: {sum(mask)}")
        #print(f"Epoch {epoch+1}/{EPOCHS}\tLoss(total): {total_loss_mean:.4f}\tLoss(recon): {reconstruction_loss_mean:.4f}\tLoss(pred): {prediction_loss_mean:.4f}\tLoss(lin): {linearity_loss_mean:.6f}\tMaxAbsEigenV: {max_abs_eigenvalue:.4f}\tSteadyModes: {sum(mask)}")
    
    return model, train_history
    
LATENT_DIMENSION = 128
ALPHA = 20
EPOCHS = 300
LR = 1e-4
NUM_STEPS = 8
RUN_NAME = f"_L{LATENT_DIMENSION}_a{ALPHA}_e{EPOCHS}_lr{LR}_st{NUM_STEPS}"

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

            dataset = MultiStepDataset(data, num_steps=NUM_STEPS)

            loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
            
            model = ConvAutoencoder(latent_dim=LATENT_DIMENSION).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            mse = nn.MSELoss()

            trained_model, train_history = train_model(model, loader, EPOCHS, NUM_STEPS, ALPHA)

            torch.save(trained_model.state_dict(), os.path.join(folder_name, "model.pt"))
            
            losses_history = np.column_stack((train_history["loss_reconstruction"], train_history["loss_list_prediction"], train_history["loss_list_linearity"], train_history["lin_dyn_weight"]))
            pd.DataFrame(losses_history, columns=["loss_reconstruction"]+[f"loss_prediction_{i}" for i in range(1, NUM_STEPS)]+[f"loss_linearity_{i}" for i in range(1, NUM_STEPS)]+["lin_dyn_weight"]).to_csv(os.path.join(folder_name,"losses_history.csv"), index = False)

            pd.DataFrame(train_history["eigenvalues"]).to_csv(os.path.join(folder_name,"eigenvalues_history.csv"), index = False)

        except:
            traceback.print_exc(file=tb_file)
