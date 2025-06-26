# MHD Surrogate Modeling

This repository contains the code and experiments for developing surrogate models for Magnetohydrodynamics (MHD) simulations, with a primary focus on deep learning techniques.

## Overview

The goal of this project is to research, develop, and evaluate various surrogate models capable of approximating the complex dynamics of MHD systems. By leveraging deep learning architectures, we aim to create computationally efficient alternatives to traditional numerical solvers.

The project is structured as a multi-package monorepo to ensure a clean separation of concerns, facilitate collaboration, and maintain high standards of reproducibility for different models and experiments.

## Project Structure

The repository is organized into several key directories:

```
mhd_surrogate_modeling/
├── .gitignore
├── README.md
│
├── packages/             # Installable Python packages
│   ├── mhd_surrogate_core/   # Foundational code (data, utils)
│   └── mhd_koopman_model/  # Koopman Autoencoder model
│
├── experiments/          # Scripts and configs for scientific studies
│   └── study_1_param_sweep/
│
└── notebooks/            # Jupyter notebooks for exploration
```

* **`packages/`**: This directory contains all the core Python source code, organized into independent, installable packages.
    * `mhd_surrogate_core`: A foundational library with no specific model dependencies. It handles common tasks like data loading, preprocessing, and plotting utilities.
    * `mhd_koopman_model`: A package containing a specific model implementation (e.g., a Koopman Autoencoder). It depends on `mhd_surrogate_core` and its own specific libraries (like PyTorch).
* **`experiments/`**: This is where scientific studies are conducted. Each subdirectory represents a self-contained experiment with its own environment definition (`requirements.txt`) and training scripts. This structure ensures that each experiment's environment is isolated and reproducible.
* **`notebooks/`**: Contains Jupyter notebooks for exploratory data analysis, visualization of results, and initial model prototyping.

## Getting Started

To set up a development environment for a specific experiment, follow these steps. This example uses `study_1_param_sweep`.

1.  **Navigate to the Experiment Directory:**
    ```
    cd experiments/study_1_param_sweep
    ```

2.  **Create a Virtual Environment:**
    This project uses `venv` for environment management to align with cluster environments.
    ```
    python -m venv venv
    ```

3.  **Activate the Environment:**
    * On macOS/Linux:
        ```
        source venv/bin/activate
        ```
    * On Windows:
        ```
        .\venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    The `requirements.txt` file specifies all necessary packages, including editable installs of the local `packages`. This ensures that any changes you make to the source code in the `packages/` directory are immediately available in your environment.
    ```
    pip install -r requirements.txt
    ```

## How to Run an Experiment

Once the environment is activated and the dependencies are installed, you can run the main training script for the experiment:

```bash
python train_kae.py
```

## Contributing

To add a new model or experiment to the project:

1.  **Create a New Model Package:** If developing a new model architecture, create a new subdirectory in `packages/` (e.g., `mhd_transformer_model`). Define its dependencies in its own `pyproject.toml` file, making sure it depends on `mhd_surrogate_core`.
2.  **Create a New Experiment Directory:** Add a new subdirectory in `experiments/`.
3.  **Define the Environment:** Create a `requirements.txt` file in your new experiment directory, listing any new third-party packages and editable installs of the required local packages.
4.  **Add Scripts:** Add your training and evaluation scripts to the experiment directory.

