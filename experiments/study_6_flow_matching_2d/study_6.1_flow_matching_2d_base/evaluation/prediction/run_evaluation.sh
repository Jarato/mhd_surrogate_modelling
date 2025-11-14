#!/bin/bash
#
# ==============================================================================
# Master Evaluation Pipeline Script (Prediction + Video)
#
# This script combines prediction generation (gen_prediction.py) and
# comparison video generation (create_comparison_video.py) into a
# single, configurable workflow.
#
# --- USAGE ---
# Run all steps:
#   ./run_evaluation.sh
#
# Run prediction generation ONLY:
#   ./run_evaluation.sh --prediction-only
#
# Run video generation ONLY (assumes predictions already exist):
#   ./run_evaluation.sh --video-only
#
# Run all, but skip prediction:
#   ./run_evaluation.sh --skip-prediction
#
# Run all, but skip video:
#   ./run_evaluation.sh --skip-video
# ==============================================================================

# --- ACTION REQUIRED: CONFIGURE YOUR PATHS & PARAMETERS HERE ---

# --- Core Paths ---
# Path to your Flow Matching *best_model.pth* file
# MODEL_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/output/sweep_valrol_steps8/f64-128-256_t64_vroll8/best_model.pth"

# Path to your *test_set.npz* file (Used for prediction input AND video ground truth)
# TEST_DATA_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/test_set.npz"

# Path to the *normalization_stats.npz* file.
# NORM_STATS_PATH="/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/normalization_stats.npz"

# --- Config File ---
# This file provides the paths above.
CONFIG_FILE="evaluation.conf"

# --- Script Paths ---
# Assumes scripts are in the same directory. Update if needed.
PRED_SCRIPT_PATH="gen_prediction.py"
VIDEO_SCRIPT_PATH="create_comparison_video.py"

# --- Step 1: Prediction Parameters ---
ODE_STEPS=16        # Higher = more accurate but slower (try 10, 20, 50)
SOLVER="midpoint"  # 'euler' or 'midpoint' (midpoint is generally better)
SEED=42            # Change this to test generation variance

# --- Step 2: Video Parameters ---
# Define the channels and their display aliases to loop over
# Example for two channels:
# channels=("vx" "vz")
# aliases=("u" "w")
channels=("vx")
aliases=("u")

# Video settings
FPS=2
NUM_WORKERS=10
BASE_SIZE=20.0 # Base size (in inches) for the plot's width.
MIN_SIZE=4.0   # Minimum size (in inches) for a single plot's height.

# Color Scale for Main Plots (Ground Truth & Prediction)
COLOR_MAP="seismic"
VMINS="vx:-2.16 vz:-3"
VMAXS="vx:3.84 vz:3"
VCENTERS="vx:0.84 vz:0.0"

# Color Scale for Difference Plot
COLOR_MAP_DIFF="seismic"
VMINS_DIFF="vx:-3.0 vz:-3.0"
VMAXS_DIFF="vx:3.0 vz:3.0"
VCENTERS_DIFF="vx:0.0 vz:0.0"

# --- END OF CONFIGURATION ---
# (No edits should be needed below this line)
# ------------------------------------------------------------------------------


# --- 1. SCRIPT LOGIC: LOAD CONFIG & PARSE FLAGS ---

# Source the configuration file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found!"
    echo "Please create '$CONFIG_FILE' and set MODEL_PATH, TEST_DATA_PATH, and NORM_STATS_PATH."
    exit 1
fi
source "$CONFIG_FILE"
echo "Loaded configuration from $CONFIG_FILE"

RUN_PREDICTION=true
RUN_VIDEO=true

# Handle exclusive "only" flags first
if [[ "$1" == "--prediction-only" ]]; then
    RUN_VIDEO=false
    echo "Running in PREDICTION ONLY mode."
elif [[ "$1" == "--video-only" ]]; then
    RUN_PREDICTION=false
    echo "Running in VIDEO ONLY mode."
else
    # Handle individual skip flags
    for arg in "$@"; do
        case $arg in
            --skip-prediction)
            RUN_PREDICTION=false
            echo "Flag found: Will skip prediction generation."
            shift
            ;;
            --skip-video)
            RUN_VIDEO=false
            echo "Flag found: Will skip video generation."
            shift
            ;;
        esac
    done
fi


# --- 2. SCRIPT LOGIC: DERIVE PATHS ---

# Directory where the output .npz files will be saved.
EVAL_OUTPUT_DIR="$(dirname "$MODEL_PATH")/eval/pred/"

# Directory to save the output videos
VIDEO_OUTPUT_DIR="${EVAL_OUTPUT_DIR}/comparison_videos/"

# Construct full paths to the prediction and difference files
PREDICTED_NPZ="${EVAL_OUTPUT_DIR}/predicted_timeseries.npz"
DIFFERENCE_NPZ="${EVAL_OUTPUT_DIR}/difference_timeseries.npz"


# --- 3. SCRIPT LOGIC: RUN PREDICTION STEP ---
if [ "$RUN_PREDICTION" = true ]; then
    echo "=============================================================================="
    echo "STEP 1: Running Flow Matching prediction generation..."
    echo "=============================================================================="
    echo "Model:    $(basename "$MODEL_PATH")"
    echo "Data:     $(basename "$TEST_DATA_PATH")"
    echo "Solver:   $SOLVER with $ODE_STEPS steps"
    echo "Output:   $EVAL_OUTPUT_DIR"
    echo "------------------------------------------------------------------------------"

    # Ensure output directory exists
    mkdir -p "$EVAL_OUTPUT_DIR"

    python "$PRED_SCRIPT_PATH" \
        --model-path "$MODEL_PATH" \
        --test-data-path "$TEST_DATA_PATH" \
        --norm-stats-path "$NORM_STATS_PATH" \
        --output-dir "$EVAL_OUTPUT_DIR" \
        --ode-steps $ODE_STEPS \
        --solver "$SOLVER" \
        --seed $SEED

    # Check if prediction generation failed
    if [ $? -ne 0 ]; then
        echo "ERROR: Prediction generation (Step 1) failed. Aborting."
        exit 1
    fi
    echo "Prediction generation complete."
else
    echo "=============================================================================="
    echo "STEP 1: Skipping prediction generation (as requested)."
    echo "=============================================================================="
fi


# --- 4. SCRIPT LOGIC: RUN VIDEO GENERATION STEP ---
if [ "$RUN_VIDEO" = true ]; then
    echo ""
    echo "=============================================================================="
    echo "STEP 2: Running comparison video generation..."
    echo "=============================================================================="

    # Sanity check: Ensure the required .npz files exist before proceeding
    if [ ! -f "$PREDICTED_NPZ" ] || [ ! -f "$DIFFERENCE_NPZ" ] || [ ! -f "$TEST_DATA_PATH" ]; then
        echo "ERROR: Cannot generate videos. One or more required files are missing."
        echo "Check for Ground Truth: $TEST_DATA_PATH"
        echo "Check for Prediction:   $PREDICTED_NPZ"
        echo "Check for Difference:   $DIFFERENCE_NPZ"
        if [ "$RUN_PREDICTION" = false ]; then
            echo "Hint: You skipped the prediction step. Run without --skip-prediction or --video-only to generate these files."
        fi
        exit 1
    fi

    # Create the output directory if it doesn't exist
    mkdir -p "$VIDEO_OUTPUT_DIR"
    echo "Video Output: $VIDEO_OUTPUT_DIR"

    # Loop through each velocity component
    for i in "${!channels[@]}"; do
        channel=${channels[$i]}
        alias=${aliases[$i]}

        # Construct the output filename dynamically
        output_filename="comparison_vid_${alias}.mp4"
        output_path="${VIDEO_OUTPUT_DIR}/${output_filename}"

        # Construct the full command for the comparison script
        command="python $VIDEO_SCRIPT_PATH \
            --ground-truth-npz \"$TEST_DATA_PATH\" \
            --predicted-npz \"$PREDICTED_NPZ\" \
            --difference-npz \"$DIFFERENCE_NPZ\" \
            --output-path \"$output_path\" \
            --channel \"$channel\" \
            --channel-alias \"$alias\" \
            --fps $FPS \
            --num-workers $NUM_WORKERS \
            --base-size $BASE_SIZE \
            --min-size $MIN_SIZE \
            --cmap \"$COLOR_MAP\" \
            --vmins $VMINS \
            --vmaxs $VMAXS \
            --vcenters $VCENTERS \
            --cmap-diff \"$COLOR_MAP_DIFF\" \
            --vmins-diff $VMINS_DIFF \
            --vmaxs-diff $VMAXS_DIFF \
            --vcenters-diff $VCENTERS_DIFF"

        # Print the command to the console and then execute it
        echo "------------------------------------------------------------------------------"
        echo "Executing command for channel '$alias' ($channel):"
        echo "Output: $output_path"
        echo "------------------------------------------------------------------------------"
        eval $command
        echo ""
    done

    echo "All comparison video generation tasks are complete."

else
    echo ""
    echo "=============================================================================="
    echo "STEP 2: Skipping video generation (as requested)."
    echo "=============================================================================="
fi

echo "Master evaluation script finished."
