#!/bin/bash

# Define placeholder variables for paths
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
OUTPUT_DIR_PATH="PATH_TO_OUTPUT_DIRECTORY"
TEST_FOLDER_PATH="PATH_TO_TEST_FOLDER"
TRAINING_ID="YOUR_TRAINING_ID"
NUM_EPOCHS=99
DATASET="CSAW"  # or "EMBED"

# Create directory if it doesn't exist
mkdir -p "$TEST_FOLDER_PATH"

# Run the Python script with the specified arguments
python3 src/evaluate/main_test_mammoregnet.py \
--data_root "$DATA_ROOT_PATH" \
--path_out_dir "$OUTPUT_DIR_PATH" \
--path_test_folder "$TEST_FOLDER_PATH" \
--id_training "$TRAINING_ID" \
--num_epoch "$NUM_EPOCHS" \
--dataset "$DATASET" \


