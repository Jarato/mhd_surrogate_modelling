#!/bin/bash

# ==============================================================================
# Divergence Comparison Video Generation Script for 2D NPZ Data
#
# This script automatically generates a 3-panel comparison video for the
# fluid divergence field (True Reconstruction, Prediction, Difference).
#
# It is designed to work with the output of 'gen_latent_prediction.py' when
# the --calculate-divergence flag is used.
#
# USAGE:
# 1. Make this script executable:
#    chmod +x generate_divergence_video.sh
#
# 2. Modify the parameters in the "USER CONFIGURATION" section below.
#
# 3. Run the script:
#    ./generate_divergence_video.sh
# ==============================================================================

# --- USER CONFIGURATION ---

# Path to the Python script that generates the comparison video
VIDEO_SCRIPT_PATH="create_comparison_video.py"

# --- Input Paths ---
# Base directory where the divergence output is stored.
# This should point to the 'divergence' subfolder created by the analysis script.
EVAL_DIR="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_4_2d_tckae/output/tc_size_exploration/ld512_M16_K16_Ktc16_gtc1.0_bwd_False/eval/latent_space_analysis/combined_high_energy_modes/divergence/"

# The following paths are now constructed automatically based on EVAL_DIR
GROUND_TRUTH_NPZ="${EVAL_DIR}/true_timeseries.npz"
PREDICTED_NPZ="${EVAL_DIR}/predicted_timeseries.npz"
DIFFERENCE_NPZ="${EVAL_DIR}/difference_timeseries.npz"

# --- Output Path ---
# Directory to save the output video. A 'comparison_videos' subfolder will be created here.
OUTPUT_DIR="${EVAL_DIR}/comparison_videos/"

# --- Video Parameters ---
FPS=16
NUM_WORKERS=128
BASE_SIZE=20.0 # Base size (in inches) for the plot's width.
MIN_SIZE=4.0   # Minimum size (in inches) for a single plot's height.

# --- Color Scale for Main Plots (Ground Truth & Prediction) ---
# NOTE: Adjust these values based on the typical range of your divergence data.
COLOR_MAP="seismic"
# VMIN="-5.0"
# VMAX="5.0"
# VCENTER="0.0"

# --- Color Scale for Difference Plot ---
# NOTE: Adjust these values based on the typical range of your divergence error.
COLOR_MAP_DIFF="seismic"
# VMIN_DIFF="-5.0"
# VMAX_DIFF="5.0"
# VCENTER_DIFF="0.0"


# --- SCRIPT LOGIC ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# The channel name is fixed to "divergence" as saved by gen_latent_prediction.py
channel="divergence"
alias="Divergence"

# Construct the output filename dynamically
output_filename="comparison_vid_${alias}.mp4"
output_path="${OUTPUT_DIR}/${output_filename}"

# Construct the full command for the comparison script
# Note: We pass the single value VMIN/VMAX/VCENTER to the plural arguments of the script
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
    # --vmins \"$channel:$VMIN\" \
    # --vmaxs \"$channel:$VMAX\" \
    # --vcenters \"$channel:$VCENTER\" \
    --cmap-diff \"$COLOR_MAP_DIFF\" \
    # --vmins-diff \"$channel:$VMIN_DIFF\" \
    # --vmaxs-diff \"$channel:$VMAX_DIFF\" \
    # --vcenters-diff \"$channel:$VCENTER_DIFF\""

# Print the command to the console and then execute it
echo "=============================================================================="
echo "Executing command for channel '$alias':"
echo -e "$command"
echo "=============================================================================="
eval $command
echo "Divergence comparison video saved to $output_path"
echo ""

echo "Divergence video generation task is complete."
