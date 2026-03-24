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


def get_stable_modes(eigenvalues, eigenvectors, eigenenergies, steady_tolerance, normalize = False):
    mask = np.abs((np.abs(eigenvalues) - 1.0)) < steady_tolerance
    eigvals_nondecay = eigenvalues[mask]
    modes_nondecay = eigenvectors[:, mask]
    energy_nondecay = eigenenergies[mask]

    explained_energy_ratio = np.sum(energy_nondecay) / np.sum(eigenenergies)
    print(f"{sum(mask)} stable mode explain: {100*explained_energy_ratio:.2f}% of the energy")

    E_nondecay = torch.tensor(eigvals_nondecay, dtype=torch.complex64)
    if normalize:
        E_nondecay = E_nondecay / torch.abs(E_nondecay)
    V_nondecay = torch.tensor(modes_nondecay, dtype=torch.complex64)
    return E_nondecay, V_nondecay, energy_nondecay


def koopman_evolve(c0, t, eigenvalues, eigenvectors):
    Lambda_t_diag = torch.diag(torch.exp(torch.log(eigenvalues) * t))
    z_t_complex = eigenvectors @ Lambda_t_diag @ c0
    return z_t_complex.real.unsqueeze(0)


def predict_trajectory_reduced(x0, model, reduced_eigvalues, reduced_eigvectors, prediction_steps, device = "cpu"):
    z0 = model.encoder(x0).squeeze(0).to(torch.complex64).to(device)
    # Compute coefficients in modal basis
    c0 = torch.linalg.pinv(reduced_eigvectors) @ z0

    steady_state_prediction = []
    with torch.no_grad():
        for t in range(prediction_steps):
            z_t_predicted = koopman_evolve(c0, t, reduced_eigvalues, reduced_eigvectors)
            steady_state_prediction.append(model.decoder(z_t_predicted).squeeze(0).cpu().numpy())
    return steady_state_prediction

def filter_pos_imag(values, vectors, energies):
    mode_mask = values.imag >= 0
    mode_vectors = vectors[:, mode_mask]
    mode_values = values[mode_mask]
    mode_energies = energies[mode_mask]
    return mode_values, mode_vectors, mode_energies

def calculate_eigenmode_structures(eigenvectors, model, device = "cpu"):
    modes = []
    with torch.no_grad():
        for vec in eigenvectors.T:
            modes.append(model.decoder(torch.FloatTensor(vec.real).unsqueeze(0).to(device)).squeeze(0).cpu().numpy())
    return modes