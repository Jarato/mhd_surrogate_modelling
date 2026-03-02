
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

class KoopmanDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size=20):
        # data: (Total_T, 2, W, H)
        self.data = torch.FloatTensor(data).permute(0,3,1,2)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Returns a sequence of length 'window_size'
        return self.data[idx : idx + self.window_size]


def train_model(kae_model, data_loader, num_epochs):
    kae_model.train()
    num_train_examples = len(data_loader.dataset)

    train_history = {}
    train_history["loss_reconstruction"] = np.zeros((num_epochs))
    train_history["loss_prediction"] = np.zeros((num_epochs))
    #train_history["eigenvalues"] = np.zeros((num_epochs, kae_model.latent_dimension), dtype=np.complex64)

    steady_tolerance = 0.1
    epoch_progress = tqdm(range(num_epochs))

    for epoch in epoch_progress:
        reconstruction_loss_epoch = 0.0
        prediction_loss_epoch = 0.0
        linearity_loss_epoch = 0.0

        for X_T in data_loader:
            X_T = X_T.to(DEVICE)
    
            optimizer.zero_grad()
                    
            B, T, C, H, W = X_T.shape
            X = X_T.view(B * T, C, H, W)
    
            Z, X_recon = model(X)
    
            Z_T = Z.view(B, T, -1)
            Z_T_null = Z_T[:, :-1, :]
            Z_T_shift = Z_T[:, 1:, :]
            Z_null = Z_T_null.reshape(B*(T-1), -1).cpu()
            Z_shift = Z_T_shift.reshape(B*(T-1), -1).cpu()
            K = torch.linalg.lstsq(Z_null, Z_shift, driver='gelsd').solution.to(DEVICE)
    
            Z_0 = Z_T[:, 0, :]
            predictions = [Z_0]
            for m in range(1, T):
                # z_m = z_0 @ (K^m)
                Z_m = torch.matmul(predictions[-1], K)
                predictions.append(Z_m)
            Z_pred = torch.stack(predictions, dim=1).view(B*T, -1)
            
            X_pred = model.decoder(Z_pred)

            loss_recon = mse(X_recon, X)
            loss_pred =  mse(X_pred, X)

            ### TOTAL LOSS ###
            loss_total = loss_recon + loss_pred

            loss_total.backward()
            optimizer.step()

            reconstruction_loss_epoch += loss_recon.item() * B
            prediction_loss_epoch += loss_pred.item() * B
        
        reconstruction_loss_mean = reconstruction_loss_epoch / num_train_examples
        prediction_loss_mean = prediction_loss_epoch / num_train_examples

        total_loss_mean = reconstruction_loss_mean + prediction_loss_mean

        train_history["loss_reconstruction"][epoch] = reconstruction_loss_mean
        train_history["loss_prediction"][epoch] = prediction_loss_mean

        epoch_progress.set_description(f"L(total): {total_loss_mean:.6f}, L(recon): {reconstruction_loss_mean:.6f}, L(pred): {prediction_loss_mean:.6f}")
            
    return model, train_history
    
LATENT_DIMENSION = 128
EPOCHS = 100
LR = 5e-4
WINDOW = 8
RUN_NAME = f"_L{LATENT_DIMENSION}_e{EPOCHS}_lr{LR}_win{WINDOW}"

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

            dataset = KoopmanDataset(data, window_size=WINDOW)

            loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
            
            model = ConvAutoencoderKoop(latent_dim=LATENT_DIMENSION).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            mse = nn.MSELoss()

            trained_model, train_history = train_model(model, loader, EPOCHS)

            torch.save(trained_model.state_dict(), os.path.join(folder_name, "model.pt"))
            
            train_history_keys = ["loss_reconstruction", "loss_prediction"]
            losses_history = np.stack([train_history[key] for key in train_history_keys], axis=1)
            pd.DataFrame(losses_history, columns=train_history_keys).to_csv(os.path.join(folder_name,"losses_history.csv"), index = False)

            #pd.DataFrame(train_history["eigenvalues"]).to_csv(os.path.join(folder_name,"eigenvalues_history.csv"), index = False)

        except:
            traceback.print_exc(file=tb_file)
