#!/bin/bash

# ====================================================================
# Hyperparameter Sweep Script for the Koopman Autoencoder
# ====================================================================

# --- Configuration ---
TRAIN_SCRIPT="train_kae.py"
DATA_PATH="/raid/skowronek/ha1000/train_val_set_dev_8.npz"
NORM_STATS_PATH="output/normalization_stats.npz"

# --- Hyperparameters to Sweep ---
LEARNING_RATES=(1e-4)
LATENT_DIMS=(128 256)
W_RECONS=(1.0)
W_PREDS=(1.0)
W_LINS=(1.0)
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
      --epochs 4 \
      --patience 16 \
      --lr-patience 4 \
      --batch-size 8
      
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
