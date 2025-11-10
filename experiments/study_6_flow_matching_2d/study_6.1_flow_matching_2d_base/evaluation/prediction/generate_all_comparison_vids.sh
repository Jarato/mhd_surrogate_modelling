#!/bin/bash
# experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/evaluation/prediction/generate_all_comparison_vids.sh
#
# ==============================================================================
# Batch Comparison Video Generation Script for 2D Flow Matching Results
#
# This script automatically generates 3-panel comparison videos (Ground Truth,
# Prediction, Difference) for all specified velocity components.
# ==============================================================================

# --- USER CONFIGURATION ---

# Path to the Python script that generates the comparison video
VIDEO_SCRIPT_PATH="create_comparison_video.py"

# --- Input Paths ---
# Base directory where the evaluation output (prediction & difference files) is stored.
# Updated for Study 6.1 Flow Matching sweep results.
EVAL_DIR="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/output/sweep_valrol_steps8/f64-128-256_t64_vroll8/eval/pred/"

# Full path to the original ground truth test set
GROUND_TRUTH_NPZ="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/test_set.npz"

# --- Output Path ---
# Directory to save the output videos
OUTPUT_DIR="${EVAL_DIR}/comparison_videos/"

# --- Video Parameters ---
FPS=2
NUM_WORKERS=128
BASE_SIZE=20.0 # Base size (in inches) for the plot's width.
MIN_SIZE=4.0   # Minimum size (in inches) for a single plot's height.

# --- Color Scale for Main Plots (Ground Truth & Prediction) ---
# Tuned for Re16k data range.
COLOR_MAP="seismic"
VMINS="vx:-2.16 vz:-3"
VMAXS="vx:3.84 vz:3"
VCENTERS="vx:0.84 vz:0.0"

# --- Color Scale for Difference Plot ---
COLOR_MAP_DIFF="seismic"
VMINS_DIFF="vx:-3.0 vz:-3.0"
VMAXS_DIFF="vx:3.0 vz:3.0"
VCENTERS_DIFF="vx:0.0 vz:0.0"


# --- SCRIPT LOGIC ---

# Construct full paths to the prediction and difference files
PREDICTED_NPZ="${EVAL_DIR}/predicted_timeseries.npz"
DIFFERENCE_NPZ="${EVAL_DIR}/difference_timeseries.npz"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the channels and their display aliases to loop over
# channels=("vx" "vz")
# aliases=("u" "w")
channels=("vx")
aliases=("u")

# Loop through each velocity component
for i in "${!channels[@]}"; do
    channel=${channels[$i]}
    alias=${aliases[$i]}

    # Construct the output filename dynamically
    output_filename="comparison_vid_${alias}.mp4"
    output_path="${OUTPUT_DIR}/${output_filename}"

    # Construct the full command for the comparison script
    command="python $VIDEO_SCRIPT_PATH \
        --ground-truth-npz \"$GROUND_TRUTH_NPZ\" \
        --predicted-npz \"$PREDICTED_NPZ\" \
        --difference-npz \"$DIFFERENCE_NPZ\" \
        --output-path \"$output_path\" \
        --channel \"$channel\" \
        --channel-alias \"$alias\" \
        --fps $FPS \
        --num-workers $NUM_WORKERS \
        --base-size $BASE_SIZE \
        --min-size $MIN_SIZE \
        --cmap \"$COLOR_MAP\" \
        --vmins $VMINS \
        --vmaxs $VMAXS \
        --vcenters $VCENTERS \
        --cmap-diff \"$COLOR_MAP_DIFF\" \
        --vmins-diff $VMINS_DIFF \
        --vmaxs-diff $VMAXS_DIFF \
        --vcenters-diff $VCENTERS_DIFF"

    # Print the command to the console and then execute it
    echo "=============================================================================="
    echo "Executing command for channel '$alias' ($channel):"
    echo "Output: $output_path"
    echo "=============================================================================="
    # Using bash -c to handle the complex command string if needed, 
    # but direct eval also works.
    eval $command
    echo ""
done

echo "All comparison video generation tasks are complete."