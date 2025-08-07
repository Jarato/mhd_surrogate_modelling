#!/bin/bash

# ====================================================================
# Hyperparameter Sweep Script for the Koopman Autoencoder
# ====================================================================

# --- Configuration ---
# This script should be run from the experiment directory, e.g.:
# cd experiments/study_2_q2d_model
# ./train.sh
# For logging use:
# ./train.sh 2>&1 | tee output/log.txt

TRAIN_SCRIPT="train.py"
DATA_PATH="/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/interp/prep2/train_val_set.npz"
NORM_STATS_PATH="/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/interp/prep2/normalization_stats.npz"

# --- Hyperparameters to Sweep ---
LEARNING_RATES=(1e-4)
LATENT_DIMS=(2048)
W_RECONS=(1.0)
W_PREDS=(1.0)
W_LINS=(1.0 0.1 0.01)
W_EIGS=(0.1)


# --- Experiment Execution ---
echo "Starting hyperparameter sweep..."

for lr in "${LEARNING_RATES[@]}"; do
for ld in "${LATENT_DIMS[@]}"; do
for wr in "${W_RECONS[@]}"; do
for wp in "${W_PREDS[@]}"; do
for wl in "${W_LINS[@]}"; do
for we in "${W_EIGS[@]}"; do

    RUN_NAME="lr_${lr}_ld_${ld}_wr_${wr}_wp_${wp}_wl_${wl}_we_${we}"
    BASE_OUTPUT_DIR="output"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
    
    # --- THE FIX IS HERE ---
    # Check if the output directory for this run already exists.
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "--------------------------------------------------"
        echo "SKIPPING: Output directory already exists for run ${RUN_NAME}"
        echo "--------------------------------------------------"
        continue # Skip to the next iteration of the loop
    fi
    
    echo "--------------------------------------------------"
    echo "RUNNING: LR=${lr}, LD=${ld}, W_Recon=${wr}, W_Pred=${wp}, W_Lin=${wl}, W_Eig=${we}"
    echo "Outputting to: ${OUTPUT_DIR}"
    echo "--------------------------------------------------"
    
    python "${TRAIN_SCRIPT}" \
      --data-path "${DATA_PATH}" \
      --norm-stats-path "${NORM_STATS_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --lr "${lr}" \
      --latent-dim "${ld}" \
      --w-recon "${wr}" \
      --w-pred "${wp}" \
      --w-lin "${wl}" \
      --w-eig "${we}" \
      --epochs 128 \
      --patience 20 \
      --lr-patience 8 \
      --clip-grad-value 25.0 \
      --batch-size 226 \
      --lr-factor 0.1

    if [ $? -ne 0 ]; then
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo "ERROR: Training failed for run ${RUN_NAME}"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    fi

done
done
done
done
done
done

echo "Hyperparameter sweep finished."