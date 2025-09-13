#!/bin/bash

# ==============================================================================
# Batch Time Evolution Plot Generation Script for Raw DNS Data
#
# This script automatically generates time evolution plots for all three velocity
# components (u, v, w) across all three evolution axes (Time-vs-X, Time-vs-Y,
# Time-vs-Z) from raw binary snapshot files.
#
# USAGE:
# 1. Make this script executable:
#    chmod +x generate_all_time_evolution_plots.sh
#
# 2. Modify the parameters in the "USER CONFIGURATION" section below.
#
# 3. Run the script from within the 'scripts' directory:
#    ./generate_all_time_evolution_plots.sh
# ==============================================================================

# --- USER CONFIGURATION ---

# Path to the directory containing the raw snapshot files
SNAPSHOT_DIR="/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/raw/"
FILE_PREFIX="patt3d_vx3d_"

# Directory to save the output plot images
OUTPUT_DIR="output/figures"

# Grid and Time parameters
NX=2301
NY=481
NZ=121
TIME_START=609
TIME_END=737
# IMPORTANT: This must be the complete list of channels in the file, in order.
SOURCE_CHANNELS="vx vy vz T" 

# Plotting and color scale parameters
NUM_WORKERS=64
# Per-component min/max values for the color scale. Use space-separated "channel:value".
VMINS="vx:-2.16 vy:-3 vz:-3"
VMAXS="vx:3.84 vy:3 vz:3"
UNIT_LABEL="" # Set to "" or "m/s", etc.
COLOR_MAP="seismic" # E.g., coolwarm, bwr, seismic, plasma, viridis
# Per-component center values for the colormap. Use space-separated "channel:value".
VCENTERS="vx:0.84 vy:0.0 vz:0.0"

# Interpolation parameters
INTERP_Y=1024
INTERP_Z=1024

# Central indices for the lines to be plotted
CENTRAL_X_INDEX=$((NX / 2)) # e.g., 2301 / 2 = 1150
CENTRAL_Y_INDEX=$((NY / 2)) # e.g., 481 / 2 = 240
CENTRAL_Z_INDEX=$((NZ / 2)) # e.g., 121 / 2 = 60

# --- SCRIPT LOGIC ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the loops
plot_types=("time-z" "time-y" "time-x")
channels=("vx" "vy" "vz")
aliases=("u" "v" "w")

# Loop through each plot type
for plot_type in "${plot_types[@]}"; do
    
    # Determine the correct slice indices for the current plot type
    slice_indices=""
    if [ "$plot_type" == "time-z" ]; then
        slice_indices="$CENTRAL_X_INDEX $CENTRAL_Y_INDEX"
    elif [ "$plot_type" == "time-y" ]; then
        slice_indices="$CENTRAL_X_INDEX $CENTRAL_Z_INDEX"
    elif [ "$plot_type" == "time-x" ]; then
        slice_indices="$CENTRAL_Y_INDEX $CENTRAL_Z_INDEX"
    fi

    # Loop through each velocity component
    for i in "${!channels[@]}"; do
        channel=${channels[$i]}
        alias=${aliases[$i]}

        # Construct the full command
        command="python create_time_evolution_plots.py \
            --snapshot-dir \"$SNAPSHOT_DIR\" \
            --file-prefix \"$FILE_PREFIX\" \
            --output-dir \"$OUTPUT_DIR\" \
            --nx $NX --ny $NY --nz $NZ \
            --time-start $TIME_START --time-end $TIME_END \
            --source-channels $SOURCE_CHANNELS \
            --plot-type \"$plot_type\" \
            --slice-indices $slice_indices \
            --channel \"$channel\" \
            --channel-alias \"$alias\" \
            --unit-label \"$UNIT_LABEL\" \
            --interp-y $INTERP_Y \
            --interp-z $INTERP_Z \
            --num-workers $NUM_WORKERS \
            --vmins \"$VMINS\" \
            --vmaxs \"$VMAXS\" \
            --vcenters \"$VCENTERS\" \
            --cmap \"$COLOR_MAP\""

        # Print the command to the console and then execute it
        echo "=============================================================================="
        echo "Executing command:"
        echo -e "$command"
        echo "=============================================================================="
        eval $command
        output_filename="${plot_type}_${alias}.png"
        echo "Plot saved to ${OUTPUT_DIR}/${output_filename}"
        echo ""
    done
done

echo "All plot generation tasks are complete."

