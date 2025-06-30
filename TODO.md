# Project TODO List
This document tracks the development tasks for the MHD Surrogate Modeling project.

### Stage 1: Baseline Implementation and HPC Pipeline
##### Core Package (mhd_surrogate_core)
- [ ] Implement initial data loading script in data.py to handle the .npz files.

- [ ] Add a basic utility function in utils.py (e.g., for normalization).

- [ ] Create a simple plotting function in data.py to visualize a single data snapshot.

##### Canonical KAE Model (mhd_canonical_kae)
- [ ] Define the KoopmanAutoencoder network architecture in model.py using PyTorch.

    - [ ] Create the 3D CNN Encoder.

    - [ ] Create the 3D CNN Decoder.

    - [ ] Implement the linear Koopman layer.

- [ ] Finalize the list of dependencies in the pyproject.toml files.

##### Experiment Setup (experiments/study_1_param_sweep)
- [ ] Write the main train_kae.py script.

    - [ ] Set up the training loop.

    - [ ] Implement the composite loss function (Reconstruction, Prediction, Linearity).

    - [ ] Add basic logging for training and validation loss.

- [ ] Create the requirements.txt file for the venv.
