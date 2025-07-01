#!/bin/bash

# ==============================================================================
# This script stages the final processed datasets for one or more Hartmann numbers
# from the permanent /cephfs storage to the temporary /raid scratch space
# for high-performance ML jobs.
#
# It creates the destination directory structure and skips any files that
# already exist in the destination to prevent unnecessary copying.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- User Configuration ---
# IMPORTANT: Set this variable to the absolute path of your top-level source data directory.
# This is the only line you should need to edit for the source path.
DATA_DIR="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data"

# The base path for your temporary high-speed storage.
DEST_BASE="/raid/skowronek"

# An array of the final dataset filenames to be copied for each Ha.
# Add or remove filenames here as needed for your jobs.
FILES_TO_COPY=(
    "vxyz_jxyz_p_f_du_dv_dw_uncompressed.npz"
)

# --- Script Configuration ---
# Get the absolute path of the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Sanity Checks ---
if [ "$DATA_DIR" == "/path/to/your/data_directory" ]; then
    echo "Error: Please edit this script and set the DATA_DIR variable to the correct path." >&2
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: The specified source data directory does not exist: $DATA_DIR" >&2
    exit 1
fi

# The base path for your permanent data storage (derived from DATA_DIR).
SRC_BASE="${DATA_DIR}"

# --- Argument Parsing ---
# Initialize an empty array to hold the Ha numbers to be processed.
HA_NUMBERS_TO_PROCESS=()

if [ "$1" == "--all" ]; then
    echo "Processing all available Hartmann numbers..."
    # Find all directories matching the pattern, extract the Ha number, and add to the list.
    for dir_path in "${SRC_BASE}/re1000_ha"*/ ; do
        # Check if it is a directory
        if [ -d "$dir_path" ]; then
            # Get just the directory name, e.g., "re1000_ha300"
            dir_name=$(basename "$dir_path")
            # Extract the number by removing the prefix
            ha_num=${dir_name##re1000_ha}
            HA_NUMBERS_TO_PROCESS+=( "$ha_num" )
        fi
    done
else
    # Use the command-line arguments as the list of Ha numbers.
    HA_NUMBERS_TO_PROCESS=("$@")
fi

# Ensure at least one Ha number is available to process.
if [ ${#HA_NUMBERS_TO_PROCESS[@]} -eq 0 ]; then
    echo "Usage: $0 <ha_number_1> [ha_number_2] ... | --all"
    echo "Example (single): $0 300"
    echo "Example (multiple): $0 300 500 700"
    echo "Example (all):    $0 --all"
    exit 1
fi

# --- Main Processing Loop ---
echo "Starting data staging process for Ha = ${HA_NUMBERS_TO_PROCESS[*]}..."

# Loop over all determined Hartmann numbers.
for HA_NUMBER in "${HA_NUMBERS_TO_PROCESS[@]}"; do
    echo ""
    echo "================================================="
    echo "Processing for Hartmann number (ha): $HA_NUMBER"
    echo "================================================="

    # --- Path Construction ---
    # Construct the full source and destination paths for the specified Ha.
    SRC_DIR="${SRC_BASE}/re1000_ha${HA_NUMBER}/3d/pkl"
    DEST_DIR="${DEST_BASE}/re1000_ha${HA_NUMBER}/3d"

    # --- Directory Creation ---
    # Safely create the destination directory and any parent directories if they
    # do not exist. The '-p' flag prevents errors if the directory is already there.
    echo "Ensuring destination directory exists: $DEST_DIR"
    mkdir -p "$DEST_DIR"

    # --- File Copy Loop ---
    echo "Staging files for Ha=${HA_NUMBER}..."

    for file_basename in "${FILES_TO_COPY[@]}"; do
        SRC_FILE="${SRC_DIR}/${file_basename}"
        DEST_FILE="${DEST_DIR}/${file_basename}"

        # First, check if the source file actually exists before trying to copy it.
        if [ ! -f "$SRC_FILE" ]; then
            echo "Warning: Source file not found, skipping: $SRC_FILE"
            continue
        fi
        
        # Check if the destination file already exists. If not, copy it.
        if [ ! -f "$DEST_FILE" ]; then
            echo "Copying '${file_basename}' to RAID..."
            # The '-v' flag (verbose) prints a confirmation of the copy operation.
            cp -v "$SRC_FILE" "$DEST_FILE"
        else
            echo "File '${file_basename}' already exists on RAID, skipping."
        fi
    done

    echo "Data staging for Ha=${HA_NUMBER} complete."

done

echo ""
echo "================================================="
echo "All staging tasks are complete."
echo "================================================="
