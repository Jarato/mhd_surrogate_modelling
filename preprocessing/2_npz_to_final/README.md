Data Preprocessing Workflow
This guide outlines the two-step process for preparing the MHD simulation data for model training. The process first splits the raw data into a set for training/validation and a separate, contiguous test set. Then, it computes normalization statistics only from the training portion of the data.

Workflow Overview
The data flow is as follows:

raw_simulation.npz: Your initial, large, time-ordered dataset.

➡️ scripts/prepare_dataset.py: Splits the raw data chronologically.

Outputs:

train_val_set.npz: The earlier portion of data, to be used for training and validation.

test_set.npz: The later, contiguous portion of data, reserved for final model evaluation.

➡️ scripts/compute_normalization.py: Takes train_val_set.npz, internally splits it into training and validation sets, and computes statistics.

Output:

normalization_stats.npz: Contains the min/max values calculated from the training set only, along with the indices needed to replicate the exact train/validation split.

Step 1: Create Train/Val and Test Splits
First, use the prepare_dataset.py script to perform a chronological split on your raw data. This is crucial for time-series forecasting to prevent the model from being trained on data that occurs later in time than the test data.

Purpose: To create a training/validation set and a final, held-out test set for evaluating long-term rollouts.

Command:
Run the following command from the root of your project, replacing the paths and split ratio as needed.

python scripts/prepare_dataset.py \
    --data-path /path/to/your/raw_simulation.npz \
    --output-dir data/processed \
    --test-split 0.15

What it does:

Input: Takes your full time-series data (raw_simulation.npz).

Output: Produces two new files in the data/processed/ directory:

train_val_set.npz: Contains the first 85% of the time snapshots.

test_set.npz: Contains the final 15% of the time snapshots, kept in their original sequence.

Step 2: Compute Normalization Statistics
Next, use the compute_normalization.py script on the train_val_set.npz file generated in the previous step. This script identifies the training samples and calculates the per-channel minimum and maximum values using only those training samples.

Purpose: To calculate normalization statistics without any data leakage from the validation or test sets.

Command:
This command uses the output from Step 1 as its input. The --seed should match the one you'll use in your model training script to ensure the train/validation split is identical.

python scripts/compute_normalization.py \
    --data-path data/processed/train_val_set.npz \
    --output-path data/processed/normalization_stats.npz \
    --val-split 0.2 \
    --seed 42

What it does:

Input: Takes the train_val_set.npz created previously.

Process:

Internally splits the data into training (80%) and validation (20%) sets based on the --val-split and --seed.

Iterates through only the training set to find the absolute minimum and maximum values for each data channel.

Output: Saves a single file, data/processed/normalization_stats.npz, containing:

min_vals: An array of minimum values for each channel.

max_vals: An array of maximum values for each channel.

train_indices: The indices from train_val_set.npz that belong to the training set.

val_indices: The indices from train_val_set.npz that belong to the validation set.

Your training script will use this normalization_stats.npz file to properly normalize the data and to ensure it uses the exact same split for training and validation.