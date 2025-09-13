#!/bin/bash

# --- Configuration ---
# Define the directories and the scripts to be run within them.
# This makes the script easier to read and modify later.

# Task 1: Run the script for preprocessed data
DIR_1="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/data_plotting_and_stats/preprocessed_data"
SCRIPT_1="generate_all_videos_from_npz.sh"

# Task 2: Run the script for raw DNS data
DIR_2="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/data_plotting_and_stats/raw_dns_data"
SCRIPT_2="generate_all_videos_from_raw.sh"


# --- Execution ---

# Run the first script in its directory using a subshell
echo "--- 📂 Processing in: $DIR_1 ---"
(
  # Change directory, or exit the subshell on failure
  cd "$DIR_1" || exit 1
  
  echo "Executing $SCRIPT_1..."
  bash "./$SCRIPT_1"
)

# Run the second script in its directory using a subshell
echo "--- 📂 Processing in: $DIR_2 ---"
(
  # Change directory, or exit the subshell on failure
  cd "$DIR_2" || exit 1
  
  echo "Executing $SCRIPT_2..."
  bash "./$SCRIPT_2"
)

echo "✅ All tasks completed!"