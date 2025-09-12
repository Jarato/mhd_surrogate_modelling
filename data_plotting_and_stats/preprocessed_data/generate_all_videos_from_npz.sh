#!/bin/bash

# ==============================================================================
# Batch Video Generation Script for Preprocessed NPZ Data
#
# This script automatically generates videos for all three velocity components
# (u, v, w) across all three slice orientations (xz, xy, yz).
#
# USAGE:
# 1. Make this script executable:
#    chmod +x generate_all_videos_from_npz.sh
#
# 2. Modify the parameters in the "USER CONFIGURATION" section below.
#
# 3. Run the script from within the 'scripts' directory:
#    ./generate_all_videos_from_npz.sh
# ==============================================================================

# --- USER CONFIGURATION ---

# Path to the input NPZ file
INPUT_NPZ="/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1220_x1151_y5_z127_c3/T1220_x1151_y5_z127_c3.npz"

# Directory to save the output videos
OUTPUT_DIR="output"

# Video and color scale parameters
FPS=16
NUM_WORKERS=127
VMIN=-5
VMAX=6
UNIT_LABEL="" # Set to "" or "m/s", etc.

# Central slice indices.
# NOTE: You must know the dimensions of your preprocessed data to set these.
# Based on your filename "T1220_x1151_y5_z127_c3", we assume:
# - 5 points on the y-axis (indices 0-4), so central index is 2.
# - 127 points on the z-axis (indices 0-126), so central index is 63.
# - We don't need the x-index for a central slice, but it would be 1151/2 = 575.
CENTRAL_Y_INDEX=2
CENTRAL_Z_INDEX=63
CENTRAL_X_INDEX=575 # This corresponds to the middle of the original 2301 points with a stride of 2.

# --- SCRIPT LOGIC ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the loops
orientations=("xz" "xy" "yz")
channels=("vx" "vy" "vz")
aliases=("u" "v" "w")

# Loop through each slice orientation
for orientation in "${orientations[@]}"; do
    
    # Determine the correct slice index for the current orientation
    slice_index=0
    if [ "$orientation" == "xz" ]; then
        slice_index=$CENTRAL_Y_INDEX
    elif [ "$orientation" == "xy" ]; then
        slice_index=$CENTRAL_Z_INDEX
    elif [ "$orientation" == "yz" ]; then
        slice_index=$CENTRAL_X_INDEX
    fi

    # Loop through each velocity component
    for i in "${!channels[@]}"; do
        channel=${channels[$i]}
        alias=${aliases[$i]}

        # Construct the output filename dynamically
        output_filename="preprocessed_${orientation}_slice_${alias}.mp4"
        output_path="${OUTPUT_DIR}/${output_filename}"

        # Construct the full command
        command="python create_video_from_npz.py \
            --input-npz \"$INPUT_NPZ\" \
            --output-path \"$output_path\" \
            --slice-orientation \"$orientation\" \
            --slice-index $slice_index \
            --channel \"$channel\" \
            --channel-alias \"$alias\" \
            --unit-label \"$UNIT_LABEL\" \
            --fps $FPS \
            --num-workers $NUM_WORKERS \
            --vmin $VMIN \
            --vmax $VMAX"

        # Print the command to the console and then execute it
        echo "=============================================================================="
        echo "Executing command:"
        echo "$command"
        echo "=============================================================================="
        eval $command
        echo "Video saved to $output_path"
        echo ""
    done
done

echo "All video generation tasks are complete."
