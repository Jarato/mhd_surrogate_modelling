# Project TODO List
This document tracks the development tasks for the MHD Surrogate Modeling project.

- train.py (see Gemini webapp)

    - Suggested further improvements (from Gemini):

        - Gradient Clipping: As a safety measure against unstable training, it's common to "clip" the gradients to a maximum value before the optimizer step. This prevents rare, large gradients from derailing the training process.

    - Final model training (with normalization param determination) on whole train-val.

    - CV for fine-grained selection between top models.

    - Try standardization instead of normalization to allow predictions outside of the seen realm.

### Stage 1: Baseline Implementation and HPC Pipeline

##### Canonical KAE Model (mhd_canonical_kae)
- [ ] Define the KoopmanAutoencoder network architecture in model.py using PyTorch.

    - ???

    -  normalization
