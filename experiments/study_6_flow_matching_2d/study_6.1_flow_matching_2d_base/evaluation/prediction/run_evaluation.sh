#!/bin/bash
# experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/evaluation/prediction/run_evaluation.sh
#
# This script runs the Flow Matching prediction generation script.
#
# --- ACTION REQUIRED ---
# Update these paths to match your environment if they have changed.
# -----------------------

# Path to your Flow Matching *best_model.pth* file
MODEL_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/output/sweep_valrol/f64-128-256_t64_vroll8/best_model.pth"

# Path to your *test_set.npz* file (Verify this matches your training data source)
TEST_DATA_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/test_set.npz"

# Path to the *normalization_stats.npz* file.
NORM_STATS_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/normalization_stats.npz"

# Directory where the output .npz files will be saved.
# Updated to point to a generic 'eval/pred' folder relative to the model, 
# or you can hardcode a specific path like you did before.
OUTPUT_DIR="$(dirname "$MODEL_PATH")/eval/pred/"

# --- Flow Matching Specific Parameters ---
ODE_STEPS=8         # Higher = more accurate but slower (try 10, 20, 50)
SOLVER="midpoint"    # 'euler' or 'midpoint' (midpoint is generally better)
SEED=42              # Change this to test generation variance

echo "----------------------------------------------------------------"
echo "Running Flow Matching prediction generation..."
echo "Model: $(basename "$MODEL_PATH")"
echo "Solver: $SOLVER with $ODE_STEPS steps"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------------------------------"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

python gen_prediction.py \
    --model-path "$MODEL_PATH" \
    --test-data-path "$TEST_DATA_PATH" \
    --norm-stats-path "$NORM_STATS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --ode-steps $ODE_STEPS \
    --solver "$SOLVER" \
    --seed $SEED

echo "Prediction generation complete."