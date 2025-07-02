#!/bin/bash
# Script to run parallel training with multiple ARC problems

# Default values
PROBLEMS="53 87 129 140 150 155 179 241 322 339 346 355 380 385"
NUM_TRAJECTORIES=10000
NUM_PROCESSES=2
OUTPUT_DIR="trajectories_output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --problems)
            PROBLEMS="$2"
            shift 2
            ;;
        --num_trajectories)
            NUM_TRAJECTORIES="$2"
            shift 2
            ;;
        --num_processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "ARC Parallel Training Configuration"
echo "========================================="
echo "Problems: $PROBLEMS"
echo "Trajectories per problem: $NUM_TRAJECTORIES"
echo "Number of processes: $NUM_PROCESSES"
echo "Output directory: $OUTPUT_DIR"
echo "========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the parallel training
python3 main_parallel.py \
    --problems $PROBLEMS \
    --num_trajectories $NUM_TRAJECTORIES \
    --num_processes $NUM_PROCESSES \
    --output_dir $OUTPUT_DIR \
    --save_trajectories \
    --checkpoint_interval 1000

echo "Training completed!"