#!/bin/bash

# --- Configuration ---
# Set the source directory.
# All files and subdirectories within this directory will be copied.
# Example: SOURCE_DIR="/home/user/documents"
SOURCE_DIR="/home/skowronek/Documents/PhD/nuclear_fusion_cooling/data/re1000_ha1000/3d/raid"

# Set the destination directory.
# The files and subdirectories from the source will be copied here.
# Example: DESTINATION_DIR="/home/user/backups"
DESTINATION_DIR="/raid/skowronek/ha1000"

# --- Script ---

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory '$SOURCE_DIR' not found."
  exit 1
fi

# Create the destination directory if it does not exist.
# The '-p' flag ensures that any missing parent directories are also created.
echo "Checking if destination directory '$DESTINATION_DIR' exists..."
mkdir -p "$DESTINATION_DIR"
echo "Destination directory is ready."

# Copy the files from the source to the destination.
# The '-r' flag stands for 'recursive', which is necessary to copy directories.
# The '-v' flag stands for 'verbose', which lists the files as they are copied.
echo "Copying files from '$SOURCE_DIR' to '$DESTINATION_DIR'..."
cp -rv "$SOURCE_DIR"/* "$DESTINATION_DIR"

echo "File copy completed successfully!"
