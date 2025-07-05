# Project TODO List
This document tracks the development tasks for the MHD Surrogate Modeling project.

- train.py (see Gemini webapp)

    - param sweep

        - discuss: should pred-loss be the new most important one?

        - Add [pred|lin|recon|eig]_loss to hparams such that we can compare easily for these

        - analyze

        - new overnight run

            - run with less channel? Or explore something else?

    - train

        - 

    - evaluation

        - visualization of real data vs train and test predictions

        - total, recon, lin, pred error on test data

        - eval in latent space?

    - CV for fine-grained selection between top models

    - different random seeds to tune the final model a bit more for publishing

    - Try standardization instead of normalization to allow predictions outside of the seen realm.
