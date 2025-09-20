#!/bin/bash

# ==============================================================================
# Batch Video Generation Script for Preprocessed 2D NPZ Data
#
# This script automatically generates videos for all specified velocity components
# (e.g., u, v, w).
#
# USAGE:
# 1. Make this script executable:
#    chmod +x generate_all_videos_from_npz_2d.sh
#
# 2. Modify the parameters in the "USER CONFIGURATION" section below.
#
# 3. Run the script from within the 'scripts' directory:
#    ./generate_all_videos_from_npz_2d.sh
# ==============================================================================

# --- USER CONFIGURATION ---

# Path to the Python script that generates the video
VIDEO_SCRIPT_PATH="create_video_from_npz.py"

# Path to the input 2D NPZ file
INPUT_NPZ="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_4_2d_tckae/output/no_grad_clipping/ld512_M8_K8_Ktc8_gtc1.0_bwd_False/eval/difference_timeseries.npz"

# Directory to save the output videos
OUTPUT_DIR="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_4_2d_tckae/output/no_grad_clipping/ld512_M8_K8_Ktc8_gtc1.0_bwd_False/eval/videos/"

# Video and color scale parameters
FPS=16
NUM_WORKERS=128
UNIT_LABEL="" # Set to "" or "m/s", etc.
COLOR_MAP="seismic" # E.g., coolwarm, bwr, seismic, plasma, viridis
BASE_SIZE=20.0 # The base size (in inches) for the longest dimension of the plot.
MIN_SIZE=4.0 # The minimum size (in inches) for the shortest dimension of the plot.


# Per-component min/max/center values for the color scale. Use space-separated "channel:value".
# Adjust these channels and values to match your 2D dataset.
VMINS="vx:-2.16 vz:-3"
VMAXS="vx:3.84 vz:3"
VCENTERS="vx:0.84 vz:0.0"


# --- SCRIPT LOGIC ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the channels and their display aliases.
# IMPORTANT: Adjust these to match the channels available in your 2D NPZ file.
channels=("vx" "vz")
aliases=("u" "w")

# Loop through each velocity component
for i in "${!channels[@]}"; do
    channel=${channels[$i]}
    alias=${aliases[$i]}

    # Construct the output filename dynamically
    output_filename="preprocessed_2d_video_${alias}.mp4"
    output_path="${OUTPUT_DIR}/${output_filename}"

    # Construct the full command for the 2D script
    command="python $VIDEO_SCRIPT_PATH \
        --input-npz \"$INPUT_NPZ\" \
        --output-path \"$output_path\" \
        --channel \"$channel\" \
        --channel-alias \"$alias\" \
        --unit-label \"$UNIT_LABEL\" \
        --fps $FPS \
        --num-workers $NUM_WORKERS \
        --vmins $VMINS \
        --vmaxs $VMAXS \
        --vcenters $VCENTERS \
        --cmap \"$COLOR_MAP\" \
        --base-size $BASE_SIZE \
        --min-size $MIN_SIZE"

    # Print the command to the console and then execute it
    echo "=============================================================================="
    echo "Executing command:"
    echo -e "$command"
    echo "=============================================================================="
    eval $command
    echo "Video saved to $output_path"
    echo ""
done

echo "All 2D video generation tasks are complete."

