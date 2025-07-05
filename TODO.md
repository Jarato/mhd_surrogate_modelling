# Project TODO List
This document tracks the development tasks for the MHD Surrogate Modeling project.

- train.py (see Gemini webapp)

    - param sweep

    - evaluation

        - R^2 or other predictions for init points in training data

        - visualization of real data vs train and test predictions

        - total, recon, lin, pred error on test data

    - CV for fine-grained selection between top models

    - different random seeds to tune the final model a bit more for publishing

    - Try standardization instead of normalization to allow predictions outside of the seen realm.
