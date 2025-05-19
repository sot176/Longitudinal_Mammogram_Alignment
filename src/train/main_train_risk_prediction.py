import argparse
import os
import random

import kornia.augmentation as K
import torch
from torch.utils.data import DataLoader

from src.dataloaders.risk_prediction.dataset_csaw import  BreastCancerRiskDatasetCSAWCC
from src.dataloaders.risk_prediction.dataset_embed import  BreastCancerRiskDataset
from src.train.train_risk_prediction import (train_val_jointly,
                                             train_val_jointly_img_alignment)



def parse_arguments():
    parser = argparse.ArgumentParser(description="Training config for breast cancer risk prediction")

    # Paths and dataset
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with dataset info")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset images")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Output directory for saving models and logs")
    parser.add_argument("--id_training", type=int, required=True, help="Unique training run ID")
    parser.add_argument("--dataset", type=str, default="EMBED", help="Dataset to use (EMBED or CSAW)")

    # Training settings
    parser.add_argument("--augmentations", type=str, required=True, help="Enable data augmentation if 'True'")
    parser.add_argument("--use_scheduler", type=str, required=True, help="Use learning rate scheduler if 'True'")
    parser.add_argument("--use_img_alignment", type=str, default="False", help="Enable image-level alignment if 'True'")
    parser.add_argument("--use_img_feat_alignment", type=str, default="False", help="Enable image-feature alignment if 'True'")
    parser.add_argument("--no_feat_Alignment", type=str, default="False", help="Disable feature alignment if 'True'")
    parser.add_argument("--use_reg_loss", type=str, default="False", help="Use regularization loss if 'True'")
    parser.add_argument("--lambda_regu", type=float, default=0.2, help="Weight for regularization loss")

    parser.add_argument("--patience_lr_scheduler", default=5, type=int, help="Patience epochs for LR scheduler")
    parser.add_argument("--patience", default=15, type=int, help="Patience epochs for early stopping")
    parser.add_argument("--accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--lr_decay", default=0.5, type=float, help="Learning rate decay factor")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")

    # DataLoader params
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for training and validation")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--schuffle", default=True, type=bool, help="Shuffle training data")
    parser.add_argument("--pin_memory", default=True, type=bool, help="Use pin_memory in DataLoader")

    # Reproducibility
    parser.add_argument("--seed", default=2023, type=int, help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup image augmentations for training if enabled
    if args.augmentations == "True":
        transforms_img_train = torch.nn.Sequential(
            K.RandomRotation(degrees=10),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        )
        transforms_img_val = None
    else:
        transforms_img_train = None
        transforms_img_val = None

    print(f"Train augmentations: {transforms_img_train}")

    # Load datasets based on selection
    if args.dataset.upper() == "CSAW":
        train_dataset = BreastCancerRiskDatasetCSAWCC(
            args.csv_file, args.data_root, "train", transforms=transforms_img_train
        )
        validation_dataset = BreastCancerRiskDatasetCSAWCC(
            args.csv_file, args.data_root, "val", transforms=transforms_img_val
        )
    else:
        train_dataset = BreastCancerRiskDataset(
            args.csv_file, args.data_root, "train", transforms=transforms_img_train
        )
        validation_dataset = BreastCancerRiskDataset(
            args.csv_file, args.data_root, "val", transforms=transforms_img_val
        )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.schuffle,
        pin_memory=args.pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    # Define paths for model checkpoint and logging
    model_path = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"
    log_path = f"train_risk_prediction_training_id_{args.id_training}.log"
    path_out_model = os.path.join(args.path_out_dir, model_path)
    path_logger = os.path.join(args.path_out_dir, log_path)

    print(f"Training on device: {device}")
    print(f"Model will be saved to: {path_out_model}")
    print(f"Log will be saved to: {path_logger}")

    # Choose appropriate training loop based on image alignment flag
    if args.use_img_alignment == "True":
        train_val_jointly_img_alignment(
            train_loader,
            validation_loader,
            device,
            args.learning_rate,
            args.weight_decay,
            args.num_epochs,
            path_logger,
            path_out_model,
            args.id_training,
            args.use_scheduler,
            args.path_out_dir,
            args.accumulation_steps,
            args.patience_lr_scheduler,
            args.patience,
            args.use_img_feat_alignment,
            args.lr_decay,
            args.dataset,
        )
    else:
        train_val_jointly(
            train_loader,
            validation_loader,
            device,
            args.learning_rate,
            args.weight_decay,
            args.num_epochs,
            path_logger,
            path_out_model,
            args.id_training,
            args.use_scheduler,
            args.path_out_dir,
            args.accumulation_steps,
            args.patience_lr_scheduler,
            args.patience,
            args.use_reg_loss,
            args.lambda_regu,
            args.lr_decay,
            args.no_feat_Alignment,
        )

if __name__ == "__main__":
    main()
