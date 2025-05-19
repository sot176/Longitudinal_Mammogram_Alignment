#!/bin/bash

# Define placeholder variables for paths
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
OUTPUT_DIR_PATH="PATH_TO_OUTPUT_DIRECTORY"
TRAINING_ID="YOUR_TRAINING_ID"
DATASET="CSAW"  # or "EMBED"

# Create directory if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"

# Run the Python script with the specified arguments
python3 src/train/main_train_mammoregnet.py \
--data_root "$DATA_ROOT_PATH" \
--path_out_dir "$OUTPUT_DIR_PATH" \
--id_training "$TRAINING_ID" \
--dataset "$DATASET" \

