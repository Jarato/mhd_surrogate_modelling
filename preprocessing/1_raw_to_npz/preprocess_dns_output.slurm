#!/bin/bash

#==============================================================================
# SLURM Directives
#==============================================================================
# This section contains the directives for the SLURM scheduler.
# The '#' at the beginning is for comments, but '#SBATCH' is a special
# directive that SLURM reads.

# -- Job Name --
# Specifies a name for your job. This is the name you'll see in the queue.
#SBATCH --job-name=mhd_preprocess_medium_test

# -- Output and Error Files --
# Specifies where to save the standard output and error of your job.
# The logs will be placed in a local 'output' directory.
# %x is the job name, %j is the job ID.
#SBATCH --output=./output/%x-%j.out
#SBATCH --error=./output/%x-%j.err

# -- Email Notifications --
# Sends you an email when the job begins, ends, or fails (ALL).
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.skowronek@tu-dortmund.de

# -- Wall Clock Time --
# Specifies the maximum time your job is allowed to run.
# 30 minutes should be sufficient for this test.
#SBATCH --time=00:30:00

# -- Resource Allocation for a Large, Memory-Bound Job --
# This requests 18 CPU cores on a single node. We request more memory than
# the default per-core to accommodate the large files.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# -- Memory Request --
# Requesting 90GB, which is a safe amount for 18 workers processing ~4GB files.
#SBATCH --mem=75GB

# -- Project Account --
# Specifies which project account to charge the compute time to.
#SBATCH --account=pn67la

# -- Partition / Queue --
# The 'micro' partition is appropriate for single-node, multi-core jobs.
# The 'general' partition likely has a multi-node minimum requirement.
#SBATCH --partition=test

# -- Job Environment --
# This ensures the job starts with a clean, predictable environment.
#SBATCH --export=NONE
#SBATCH --get-user-env

# -- Job Re-queue Policy --
# This prevents the job from being automatically re-queued if the node fails.
#SBATCH --no-requeue

#==============================================================================
# Job Execution
#==============================================================================
# This section contains the commands that will be executed on the compute node.

echo "Job started on $(hostname) at $(date)"
echo "SLURM assigned me ${SLURM_CPUS_PER_TASK} CPUs on ${SLURM_NODELIST}."

# --- 1. Set up the Environment ---

# -- Initialize Module System --
source /etc/profile.d/modules.sh

# -- Energy Aware Runtime (EAR) Environment Variable --
export SLURM_EAR_LOAD_MPI_VERSION="intel"

# -- Load Python Module --
module load python/3.8.11-extended

# -- Activate Virtual Environment (venv) --
source $WORK/venvs/subsampling/bin/activate


# --- 2. Run the Application ---
cd $SLURM_SUBMIT_DIR

# Define paths for input and output data.
INPUT_DATA_DIR="/hppfs/work/pn67la/di97yic/01-Cold_Runs/01-Re16K_Ha325/"
OUTPUT_FILE="/hppfs/work/pn67la/di97hev/output/01-Cold_Runs/01-Re16K_Ha325.npz"

# --- Medium Test Run ---
# Execute the Python script on 72 timesteps.
# The number of workers is set to the number of CPUs allocated by SLURM.
echo "Running Python script for a 72-timestep test..."
python preprocess_dns_output.py \
    --nx 2301 --ny 481 --nz 121 \
    --input-dir ${INPUT_DATA_DIR} \
    --output-file ${OUTPUT_FILE} \
    --time-start 0 \
    --time-end 200 \
    --num-workers 14 \
    --source-channel-labels vx vy vz T \
    --channel-indices-to-keep 0 1 2 \
    --num-x-samples 24 \
    --y-indices-to-keep 10 55 80


echo "Job finished at $(date)"

