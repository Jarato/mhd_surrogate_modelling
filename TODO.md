# Project TODO List
This document tracks the development tasks for the MHD Surrogate Modeling project.

- data:

    - getting data

        - local code for subsampling spacieally and channel wise

        - ask andreu to execute it, and provide it via some cloud

            - ask for access to supermuc

    - What border distance to take (Hartman layer = H/Ha)?

    - grad(P) is important, take this instead of P when available

    - two layers next to each other to enable gradient computations?

    - multiple boundary layers in general?

- theory:

    - think about learning transitions vs learning TS prediction

    - r^2 score, get more in depth

- param sweep

    - 

- train

    - data smoothing?

- evaluation

    -

- physics loss?!

- CV for fine-grained selection between top models

- different random seeds to tune the final model a bit more for publishing

- Try standardization instead of normalization to allow predictions outside of the seen realm.
