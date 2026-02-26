import numpy as np
import torch


def get_sorted_eigen(K_matrix):
    eigvals, eigvecs = np.linalg.eig(K_matrix)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def eigenvalue_energies(model, eigenvectors, device):
    with torch.no_grad():
        return np.array([torch.norm(model.decoder(torch.FloatTensor(v.real).unsqueeze(0).to(device))).item() for v in eigenvectors.T])

def lift_data(model, dataset, device):
    return None