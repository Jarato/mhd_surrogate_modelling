#!/bin/bash

# ==============================================================================
# Batch Video Generation Script for Raw DNS Snapshot Data
#
# This script automatically generates videos for all three velocity components
# (u, v, w) across all three slice orientations (xz, xy, yz) from raw binary
# snapshot files.
#
# USAGE:
# 1. Make this script executable:
#    chmod +x generate_all_videos_from_raw.sh
#
# 2. Modify the parameters in the "USER CONFIGURATION" section below.
#
# 3. Run the script from within the 'scripts' directory:
#    ./generate_all_videos_from_raw.sh
# ==============================================================================

# --- USER CONFIGURATION ---

# Path to the directory containing the raw snapshot files
SNAPSHOT_DIR="/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/raw/"
FILE_PREFIX="patt3d_vx3d_"

# Directory to save the output videos
OUTPUT_DIR="output/videos/"

# Grid and Time parameters
NX=2301
NY=481
NZ=121
TIME_START=609
TIME_END=737
# IMPORTANT: This must be the complete list of channels in the file, in order.
SOURCE_CHANNELS="vx vy vz T" 

# Video and color scale parameters
FPS=16
NUM_WORKERS=64
VMIN=-5
VMAX=6
UNIT_LABEL="" # Set to "" or "m/s", etc.
COLOR_MAP="coolwarm" # E.g., coolwarm, bwr, seismic, plasma, viridis

# Interpolation parameters
INTERP_Y=1024
INTERP_Z=1024

# Central slice indices based on the raw grid dimensions (NX, NY, NZ)
CENTRAL_X_INDEX=$((NX / 2)) # e.g., 2301 / 2 = 1150
CENTRAL_Y_INDEX=$((NY / 2)) # e.g., 481 / 2 = 240
CENTRAL_Z_INDEX=$((NZ / 2)) # e.g., 121 / 2 = 60

# --- SCRIPT LOGIC ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the loops for velocity components only
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
        output_filename="raw_${orientation}_slice_${alias}.mp4"
        output_path="${OUTPUT_DIR}/${output_filename}"

        # Construct the full command
        command="python create_video.py \
            --snapshot-dir \"$SNAPSHOT_DIR\" \
            --file-prefix \"$FILE_PREFIX\" \
            --output-path \"$output_path\" \
            --nx $NX --ny $NY --nz $NZ \
            --time-start $TIME_START --time-end $TIME_END \
            --source-channels $SOURCE_CHANNELS \
            --slice-orientation \"$orientation\" \
            --slice-index $slice_index \
            --channel \"$channel\" \
            --channel-alias \"$alias\" \
            --unit-label \"$UNIT_LABEL\" \
            --fps $FPS \
            --interp-y $INTERP_Y \
            --interp-z $INTERP_Z \
            --num-workers $NUM_WORKERS \
            --vmin $VMIN \
            --vmax $VMAX \
            --cmap \"$COLOR_MAP\""

        # Print the command to the console and then execute it
        echo "=============================================================================="
        echo "Executing command:"
        echo -e "$command"
        echo "=============================================================================="
        eval $command
        echo "Video saved to $output_path"
        echo ""
    done
done

echo "All video generation tasks are complete."

