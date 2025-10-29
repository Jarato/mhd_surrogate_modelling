#!/bin/bash

# This script runs the new prediction generation script.
#
# --- ACTION REQUIRED ---
# You MUST update these paths to match your environment.
# -----------------------

# Path to your *best_model.pth* file
MODEL_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_5_low_rank_tckae_2d/study_5.2_stable_svd/output/gamma_exploration_K16_val16/flat_kr16_M1_K16_Ktc0_gtc0_gi0.01_gfwd100.0_g_svdo1e-05_g_svdstab1e-05_bwd_False/best_model.pth"

# Path to your *test_set.npz* file
TEST_DATA_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/test_set.npz"

# Path to the *normalization_stats.npz* file.
# This MUST be the one generated during training (or compatible with it)
# and MUST contain 'mean_vals' and 'std_vals'.
NORM_STATS_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/normalization_stats.npz"

# Directory where the output .npz files will be saved.
# The analysis notebook will need to point to this directory.
OUTPUT_DIR="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_5_low_rank_tckae_2d/study_5.2_stable_svd/output/gamma_exploration_K16_val16/flat_kr16_M1_K16_Ktc0_gtc0_gi0.01_gfwd100.0_g_svdo1e-05_g_svdstab1e-05_bwd_False/eval/pred/"

echo "Running SVD prediction generation..."

python gen_prediction.py \
    --model-path "$MODEL_PATH" \
    --test-data-path "$TEST_DATA_PATH" \
    --norm-stats-path "$NORM_STATS_PATH" \
    --output-dir "$OUTPUT_DIR"

echo "Prediction generation complete. Files saved to $OUTPUT_DIR"
