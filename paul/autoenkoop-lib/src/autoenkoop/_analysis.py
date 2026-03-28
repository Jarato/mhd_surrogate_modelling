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

def get_multi_step_predictor(K, threshold=0.01):
    """
    Returns a function that predicts all steps from 1 to m simultaneously.
    """
    # 1. Eigendecomposition
    L, V = torch.linalg.eig(K)
    
    # 2. Filter non-decaying modes
    mask = torch.abs(L) > (1.0 - threshold)
    L_r = L[mask]    # Shape: (r,)
    V_r = V[:, mask] # Shape: (N, r)
    
    # 3. Left eigenvectors for projection
    V_inv_r = torch.linalg.pinv(V_r) # Shape: (r, N)

    def predict_m_steps(z_t, m):
        """
        z_t: initial lifted state (N,)
        m: number of future steps to predict
        Returns: tensor of shape (m, N) containing all predictions
        """
        # Step A: Initial projection to get mode weights (b_0)
        # Shape: (r,)
        b_0 = torch.mv(V_inv_r, z_t.to(V_inv_r.dtype))
        
        # Step B: Create a range of time steps [1, 2, ..., m]
        steps = torch.arange(0, m + 1, device=K.device).reshape(-1, 1) # Shape: (m, 1)
        
        # Step C: Compute (L_r)^steps using broadcasting
        # L_r is (r,), steps is (m, 1) -> result is (m, r)
        # This gives us the weights for every mode at every time step
        L_pow = torch.pow(L_r, steps) 
        b_future = L_pow * b_0 # Element-wise broadcasting: (m, r)
        
        # Step D: Map all weights back to the lifted state space
        # (m, r) @ (r, N) -> (m, N)
        z_future = torch.matmul(b_future, V_r.T)
        
        return z_future.real
        
    return predict_m_steps, L_r