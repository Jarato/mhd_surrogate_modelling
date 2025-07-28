#!/bin/bash

#==============================================================================
# SLURM Directives for a Full Run on the TEST Partition
#==============================================================================

# -- Job Name --
#SBATCH --job-name=mhd_preprocess_prod_test

# -- Output and Error Files --
# The logs will be placed in a local 'output' directory.
# %x is the job name, %j is the job ID.
#SBATCH --output=./output/%x-%j.out
#SBATCH --error=./output/%x-%j.err

# -- Email Notifications --
# Sends you an email when the job begins, ends, or fails (ALL).
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.skowronek@tu-dortmund.de

# -- Wall Clock Time --
# The 'test' partition has a 30-minute maximum.
#SBATCH --time=00:30:00

# -- Resource Allocation (Optimized for Test Partition) --
# Requesting 14 CPU cores, the optimal number for an 80GB memory footprint.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14

# -- Memory Request --
# Requesting 80GB, the empirically determined maximum for this job on the test partition.
#SBATCH --mem=80G

# -- Project Account --
#SBATCH --account=pn67la

# -- Partition / Queue --
# Using the 'test' partition for a fast production run.
#SBATCH --partition=test

# -- Job Environment --
# Ensures the job starts with a clean, predictable environment.
#SBATCH --export=NONE
#SBATCH --get-user-env

# -- Job Re-queue Policy --
# Prevents the job from being automatically re-queued if the node fails.
#SBATCH --no-requeue

#==============================================================================
# Job Execution
#==============================================================================

echo "Job started on $(hostname) at $(date)"
echo "SLURM assigned me ${SLURM_CPUS_PER_TASK} CPUs on ${SLURM_NODELIST}."

# --- 1. Set up the Environment ---

source /etc/profile.d/modules.sh
export SLURM_EAR_LOAD_MPI_VERSION="intel"
module load python/3.8.11-extended
source $WORK/venvs/subsampling/bin/activate

# --- 2. Run the Application ---
cd $SLURM_SUBMIT_DIR

# Define paths for input and output data.
INPUT_DATA_DIR="/hppfs/work/pn67la/di97yic/01-Cold_Runs/01-Re16K_Ha325/"
OUTPUT_FILE="/hppfs/work/pn67la/di97hev/output/01-Cold_Runs/01-Re16K_Ha325.npz"

# --- Production Run ---
# Execute the Python script on all 575 timesteps.
# The number of workers is set to the number of CPUs allocated by SLURM (14).
echo "Running Python script for all 575 timesteps..."
python preprocess_dns_output.py \
    --nx 2301 --ny 481 --nz 121 \
    --input-dir ${INPUT_DATA_DIR} \
    --output-file ${OUTPUT_FILE} \
    --time-start 0 \
    --time-end 575 \
    --num-workers ${SLURM_CPUS_PER_TASK} \
    --source-channel-labels vx vy vz T \
    --channel-indices-to-keep 0 1 2 \
    --num-x-samples 24 \
    --y-indices-to-keep 10 55 80

echo "Job finished at $(date)"

