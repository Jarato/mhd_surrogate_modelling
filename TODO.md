# Project TODO List
This document tracks the development tasks for the MHD Surrogate Modeling project.

- train.py (see Gemini webapp)

    - Final model training (with normalization param determination) on whole train-val.

    - CV for fine-grained selection between top models.

    - Try standardization instead of normalization to allow predictions outside of the seen realm.

### Stage 1: Baseline Implementation and HPC Pipeline

##### Canonical KAE Model (mhd_canonical_kae)
- [ ] Define the KoopmanAutoencoder network architecture in model.py using PyTorch.

    - ???

    -  normalization
