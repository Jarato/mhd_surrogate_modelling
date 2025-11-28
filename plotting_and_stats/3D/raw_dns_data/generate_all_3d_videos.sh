#!/bin/bash

# ==============================================================================
# Batch 3D Video Generation Script
#
# This script automatically generates 3D volumetric videos for selected channels
# using the 'create_3d_video.py' script.
#
# USAGE:
# 1. Make this script executable:
#    chmod +x generate_all_3d_videos.sh
#
# 2. Modify the "USER CONFIGURATION" section below.
#
# 3. Run the script:
#    ./generate_all_3d_videos.sh
# ==============================================================================

# --- USER CONFIGURATION ---

# Path to the directory containing the raw snapshot files
SNAPSHOT_DIR="/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/raw/"
FILE_PREFIX="patt3d_vx3d_"

# Directory to save the output videos
OUTPUT_DIR="output/videos_3d/"

# Grid and Time parameters
NX=2301
NY=481
NZ=121
TIME_START=609
TIME_END=737

# IMPORTANT: This must be the complete list of channels existing in the binary files
SOURCE_CHANNELS="vx vy vz T" 

# --- CHANNEL SELECTION ---
# Space-separated list of channels you want to generate videos for.
# Options usually include: vx vy vz T
CHANNELS_TO_GENERATE="vx"

# Video parameters
FPS=16

# --- WORKER CONFIGURATION ---
# RAM CALCULATION:
# Each snapshot is approx 4.3 GB (Raw) + Overhead.
# With 128 GB RAM, max safe workers = ~128GB / 5GB per worker = ~25.
# We set this to 16 to be safe and leave room for the OS and VRAM overhead.
NUM_WORKERS=16 

COLOR_MAP="plasma" # E.g., viridis, plasma, coolwarm

# --- SCRIPT LOGIC ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting 3D video generation batch..."
echo "Target Channels: $CHANNELS_TO_GENERATE"
echo "Workers: $NUM_WORKERS"
echo "----------------------------------------"

# Loop through the user-selected channels
for channel in $CHANNELS_TO_GENERATE; do
    
    # Determine Alias and Unit Label based on the channel name
    # You can add more cases here if you have different variables
    alias="$channel"
    unit_label=""
    
    case "$channel" in
        "vx")
            alias="u"
            unit_label="m/s"
            ;;
        "vy")
            alias="v"
            unit_label="m/s"
            ;;
        "vz")
            alias="w"
            unit_label="m/s"
            ;;
        "T")
            alias="Temperature"
            unit_label="K"
            ;;
    esac

    # Construct the output filename
    output_filename="volumetric_3d_${alias}.mp4"
    output_path="${OUTPUT_DIR}/${output_filename}"

    # Construct the full command
    # Note: We do not use vmin/vmax here as the current 3D script 
    # auto-scales.
    command="python create_3d_video.py \
        --snapshot-dir \"$SNAPSHOT_DIR\" \
        --file-prefix \"$FILE_PREFIX\" \
        --output-path \"$output_path\" \
        --nx $NX --ny $NY --nz $NZ \
        --time-start $TIME_START --time-end $TIME_END \
        --source-channels $SOURCE_CHANNELS \
        --channel \"$channel\" \
        --channel-alias \"$alias\" \
        --unit-label \"$unit_label\" \
        --fps $FPS \
        --num-workers $NUM_WORKERS \
        --cmap \"$COLOR_MAP\""

    # Print and Execute
    echo "Processing Channel: $channel (Alias: $alias)"
    echo "Output: $output_path"
    echo "Executing..."
    
    eval $command
    
    status=$?
    if [ $status -eq 0 ]; then
        echo "✅ Successfully generated $output_filename"
    else
        echo "❌ Error generating $output_filename"
    fi
    echo "----------------------------------------"

done

echo "All 3D video generation tasks are complete."