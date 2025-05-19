#!/bin/bash

# Define placeholder variables for paths
CSV_FILE_PATH="PATH_TO_CSV_FILE"
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
OUTPUT_DIR_PATH="PATH_TO_OUTPUT_DIRECTORY"
TRAINING_ID="YOUR_TRAINING_ID"
DATASET="CSAW"  # or "EMBED"

# Create directory if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"

# Run the Python script with the specified arguments
python3 src/train/main_train_risk_prediction.py \
--csv_file "$CSV_FILE_PATH"  \
--data_root "$DATA_ROOT_PATH"  \
--path_out_dir "$OUTPUT_DIR_PATH" \
--id_training "$TRAINING_ID" \
--use_scheduler "True" \
--accumulation_steps 1 \
--augmentations "False" \
--use_img_alignment "False" \
--no_feat_Alignment "True" \
--use_reg_loss "False" \
--dataset "$DATASET" \
--seed 2023 \
