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
#
# Force re-generation of all samples:
#   ./run_evaluation.sh --force-rerun
#
# Run combined multi-sample video instead of individual videos:
#   ./run_evaluation.sh --video-mode combined
# ==============================================================================

# --- ACTION REQUIRED: CONFIGURE YOUR PATHS & PARAMETERS HERE ---

# --- Config File ---
# This file provides the paths to MODEL_PATH, TEST_DATA_PATH, NORM_STATS_PATH
CONFIG_FILE="evaluation.conf"

# --- Script Paths ---
# Assumes scripts are in the same directory. Update if needed.
PRED_SCRIPT_PATH="gen_prediction.py"
VIDEO_SCRIPT_PATH="create_comparison_video.py"

# --- Step 1: Prediction Parameters ---
ODE_STEPS=16        # Higher = more accurate but slower (try 10, 20, 50)
SOLVER="midpoint"  # 'euler' or 'midpoint' (midpoint is generally better)
BASE_SEED=42       # Base seed. Samples will use BASE_SEED, BASE_SEED+1, ...
NUM_SAMPLES=5      # <<< NEW: Number of samples to generate
FORCE_RERUN=false  # <<< NEW: Set to true to always regenerate samples

# --- Step 2: Video Parameters ---
VIDEO_MODE="combined" # <<< NEW: 'individual' or 'combined'
VIDEO_GRID_COLS=2       # <<< NEW: Number of columns for 'combined' video

# Define the channels and their display aliases to loop over
# Example for two channels:
# channels=("vx" "vz")
# aliases=("u" "w")
channels=("vx")
aliases=("u")

# Video settings
FPS=4
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
            --force-rerun)
            FORCE_RERUN=true
            echo "Flag found: Will force-rerun all predictions."
            shift
            ;;
            --video-mode)
            VIDEO_MODE="$2"
            echo "Flag found: Setting video mode to $2."
            shift # past argument
            shift # past value
            ;;
            --video-grid-cols)
            VIDEO_GRID_COLS="$2"
            echo "Flag found: Setting video grid columns to $2."
            shift # past argument
            shift # past value
            ;;
        esac
    done
fi


# --- 2. SCRIPT LOGIC: DERIVE PATHS ---

# Directory where the output sample subdirectories will be saved.
EVAL_OUTPUT_DIR="$(dirname "$MODEL_PATH")/eval/pred/"

# Base directory to save the output videos
VIDEO_OUTPUT_DIR="${EVAL_OUTPUT_DIR}/comparison_videos/"

# NOTE: PREDICTED_NPZ and DIFFERENCE_NPZ are now defined
# inside the video loop, as they are sample-specific.


# --- 3. SCRIPT LOGIC: RUN PREDICTION STEP ---
if [ "$RUN_PREDICTION" = true ]; then
    echo "=============================================================================="
    echo "STEP 1: Running Flow Matching prediction generation..."
    echo "=============================================================================="
    echo "Model:    $(basename "$MODEL_PATH")"
    echo "Data:     $(basename "$TEST_DATA_PATH")"
    echo "Solver:   $SOLVER with $ODE_STEPS steps"
    echo "Samples:  $NUM_SAMPLES (starting from seed $BASE_SEED)"
    echo "Output:   $EVAL_OUTPUT_DIR"
    echo "------------------------------------------------------------------------------"

    # Ensure output directory exists
    mkdir -p "$EVAL_OUTPUT_DIR"

    # --- NEW: Construct force flag ---
    FORCE_FLAG=""
    if [ "$FORCE_RERUN" = true ]; then
        FORCE_FLAG="--force-rerun"
    fi

    python "$PRED_SCRIPT_PATH" \
        --model-path "$MODEL_PATH" \
        --test-data-path "$TEST_DATA_PATH" \
        --norm-stats-path "$NORM_STATS_PATH" \
        --output-dir "$EVAL_OUTPUT_DIR" \
        --ode-steps $ODE_STEPS \
        --solver "$SOLVER" \
        --seed $BASE_SEED \
        --num-samples $NUM_SAMPLES \
        $FORCE_FLAG

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
    echo "STEP 2: Running comparison video generation (Mode: $VIDEO_MODE)..."
    echo "=============================================================================="

    # --- MODE 1: Individual 3-Panel Videos (GT, Pred, Diff) for each sample ---
    if [ "$VIDEO_MODE" = "individual" ]; then
        echo "Generating individual 3-panel video for each of $NUM_SAMPLES sample(s)..."
        
        # Loop over each generated sample
        for ((s=0; s < $NUM_SAMPLES; s++)); do
            SAMPLE_STR=$(printf "sample_%02d" $s)
            echo ""
            echo "------------------------------------------------------------------------------"
            echo "--- Generating videos for $SAMPLE_STR ---"
            echo "------------------------------------------------------------------------------"

            # Define paths for this specific sample
            SAMPLE_DIR="${EVAL_OUTPUT_DIR}/${SAMPLE_STR}/"
            PREDICTED_NPZ="${SAMPLE_DIR}/predicted_timeseries.npz"
            DIFFERENCE_NPZ="${SAMPLE_DIR}/difference_timeseries.npz"

            # Sanity check: Ensure the required .npz files exist before proceeding
            if [ ! -f "$PREDICTED_NPZ" ] || [ ! -f "$DIFFERENCE_NPZ" ] || [ ! -f "$TEST_DATA_PATH" ]; then
                echo "ERROR: Cannot generate videos for $SAMPLE_STR. One or more required files are missing."
                echo "Check for Ground Truth: $TEST_DATA_PATH"
                echo "Check for Prediction:   $PREDICTED_NPZ"
                echo "Check for Difference:   $DIFFERENCE_NPZ"
                if [ "$RUN_PREDICTION" = false ]; then
                    echo "Hint: You skipped the prediction step. Run without --skip-prediction or --video-only to generate these files."
                fi
                continue # Skip to the next sample
            fi

            # Create the output directory for this sample's videos
            SAMPLE_VIDEO_OUTPUT_DIR="${VIDEO_OUTPUT_DIR}/${SAMPLE_STR}/"
            mkdir -p "$SAMPLE_VIDEO_OUTPUT_DIR"
            echo "Video Output: $SAMPLE_VIDEO_OUTPUT_DIR"

            # Loop through each velocity component
            for i in "${!channels[@]}"; do
                channel=${channels[$i]}
                alias=${aliases[$i]}

                # Construct the output filename dynamically
                output_filename="comparison_vid_${alias}.mp4"
                output_path="${SAMPLE_VIDEO_OUTPUT_DIR}/${output_filename}"

                # Construct the full command for the comparison script
                command="python $VIDEO_SCRIPT_PATH \
                    --mode individual \
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
                echo "---"
                echo "Executing command for channel '$alias' ($channel):"
                echo "Output: $output_path"
                echo "---"
                eval $command
                echo ""
            done
        done # End of sample loop
    
    # --- MODE 2: Combined Multi-Panel Video (GT, S0, S1, S2...) ---
    elif [ "$VIDEO_MODE" = "combined" ]; then
        echo "Generating one combined video for all $NUM_SAMPLES sample(s)..."
        
        # This mode runs ONCE, but still needs to loop over channels
        for i in "${!channels[@]}"; do
            channel=${channels[$i]}
            alias=${aliases[$i]}

            # Output path for the *combined* video
            output_filename="combined_vid_${alias}.mp4"
            output_path="${VIDEO_OUTPUT_DIR}/${output_filename}"

            # Sanity check: Ensure GT data exists
            if [ ! -f "$TEST_DATA_PATH" ]; then
                 echo "ERROR: Cannot generate combined video. Ground Truth file missing: $TEST_DATA_PATH"
                 exit 1
            fi
            # Sanity check: Ensure prediction base dir exists
            if [ ! -d "$EVAL_OUTPUT_DIR" ]; then
                echo "ERROR: Cannot generate combined video. Base prediction directory not found: $EVAL_OUTPUT_DIR"
                exit 1
            fi
            
            # Create the output directory (just the base video dir)
            mkdir -p "$VIDEO_OUTPUT_DIR"

            # Construct the command for the *combined* mode
            command="python $VIDEO_SCRIPT_PATH \
                --mode combined \
                --ground-truth-npz \"$TEST_DATA_PATH\" \
                --base-pred-dir \"$EVAL_OUTPUT_DIR\" \
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
                --video-grid-cols $VIDEO_GRID_COLS"
            
            # Note: Diff-related args are omitted as they don't apply here

            echo "------------------------------------------------------------------------------"
            echo "Executing command for combined video (channel '$alias'):"
            echo "Output: $output_path"
            echo "------------------------------------------------------------------------------"
            eval $command
            echo ""
        done # End of channel loop
    else
        echo "ERROR: Unknown VIDEO_MODE: '$VIDEO_MODE'. Use 'individual' or 'combined'."
    fi
    
    echo "All comparison video generation tasks are complete."

else
    echo ""
    echo "=============================================================================="
    echo "STEP 2: Skipping video generation (as requested)."
    echo "=============================================================================="
fi

echo "Master evaluation script finished."