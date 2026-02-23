import torch
import numpy as np

class TOffsetDataset(torch.utils.data.Dataset):
    def __init__(self, data, t_offset = 1):
        # data: numpy array of shape (t, x, z, v)
        self.X_0 = torch.tensor(data[:-t_offset], dtype=torch.float32).permute(0,3,1,2)
        self.X_t = torch.tensor(data[t_offset:], dtype=torch.float32).permute(0,3,1,2)
        # Now shape: (t, v=2, x=1151, z=127)

    def __len__(self):
        return self.X_0.shape[0]

    def __getitem__(self, idx):
        return self.X_0[idx], self.X_t[idx]


class MultiStepDataset(torch.utils.data.Dataset):
    def __init__(self, data, num_steps=2):
        # data: numpy array of shape (t, x, z, v)
        #self.X_0 = torch.tensor(data[:-t_offset], dtype=torch.float32).permute(0,3,1,2)
        self.X_n = [torch.tensor(data[t:t-num_steps], dtype=torch.float32).permute(0,3,1,2) for t in range(num_steps)]

        #self.X_t = torch.tensor(data[t_offset:], dtype=torch.float32).permute(0,3,1,2)
        # Now shape: (t, v=2, x=1151, z=127)

    def __len__(self):
        return self.X_n[0].shape[0]

    def __getitem__(self, idx):      
        return [Xt[idx] for Xt in self.X_n]

def get_min_max(data):
    vx_all = data[..., 0]
    vz_all = data[..., 1]
    vx_min, vx_max = np.min(vx_all), np.max(vx_all)
    vz_min, vz_max = np.min(vz_all), np.max(vz_all)
    return (vx_min, vx_max), (vz_min, vz_max)