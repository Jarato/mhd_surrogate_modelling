import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import Normalize


def get_sorted_eigen(K_matrix):
    eigvals, eigvecs = np.linalg.eig(K_matrix)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def eigenvalue_energies(model, eigenvectors, device):
    with torch.no_grad():
        return np.array([torch.norm(model.decoder(torch.FloatTensor(v.real).unsqueeze(0).to(device))).item() for v in eigenvectors.T])

def plot_eigenvalue_spectrum(eigenvalues, eigenvectors, energies):
    plt.figure(figsize=(11,9))
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    sc = plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), c=energies, cmap='plasma', s=50)
    plt.colorbar(sc, label="Mode Energy")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Koopman Eigenvalue Spectrum")
    circle = plt.Circle((0,0), 1.0, color='k', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.show()

def plot_vx_vz(vx, vz, norm_x, norm_z):

    plt.figure(figsize=(10,8))

    # --- vx ---
    plt.subplot(2,1,1)
    plt.imshow(vx.T, origin='lower', aspect='auto', cmap='viridis', norm=norm_x)
    plt.colorbar(label='vx')
    plt.title(f"vx")
    plt.xlabel("x-coordinate")
    plt.ylabel("z-coordinate")

    # --- vz ---
    plt.subplot(2,1,2)
    plt.imshow(vz.T, origin='lower', aspect='auto', cmap='plasma', norm=norm_z)
    plt.colorbar(label='vz')
    plt.title(f"vz")
    plt.xlabel("x-coordinate")
    plt.ylabel("z-coordinate")

    plt.tight_layout()
    plt.show()