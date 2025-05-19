import argparse
import os
import random

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.dataloaders.MammoRegNet.dataset_csaw import CSAWCCRegistrationDataset
from src.dataloaders.MammoRegNet.dataset_embed import EMBEDRegistrationDataset
from src.train.train_mammoregnet import train_val


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for registration model.")

    # Paths and dataset
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Output directory for models and logs")
    parser.add_argument("--id_training", type=int, required=True, help="Training ID used for naming outputs")
    parser.add_argument("--dataset", type=str, choices=["EMBED", "CSAW"], required=True, help="Dataset to use")

    # Training settings
    parser.add_argument("--augmentations", action="store_true", help="Use data augmentations")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--max_iterations", type=int, default=75, help="Maximum iterations for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for optimizer")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")

    # DataLoader params
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--schuffle", default=True, type=bool)
    parser.add_argument("--pin_memory", default=True, type=bool)

    # Reproducibility
    parser.add_argument("--seed", default=2023, type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compose transforms if augmentations are enabled
    transforms_img = (
        A.Compose([
            A.Normalize(mean=[0.176, 0.176, 0.176], std=[0.218, 0.218, 0.218]),
            ToTensorV2(),
        ])
        if args.augmentations else None
    )

    # Load datasets
    if args.dataset == "EMBED":
        train_dataset = EMBEDRegistrationDataset(args.data_root, "train", transforms=transforms_img)
        validation_dataset = EMBEDRegistrationDataset(args.data_root, "val", transforms=transforms_img)
    else:
        train_dataset = CSAWCCRegistrationDataset(args.data_root, "train", transforms=transforms_img)
        validation_dataset = CSAWCCRegistrationDataset(args.data_root, "val", transforms=transforms_img)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    # Setup output paths
    model_filename = f"model_registration_training_id_{args.id_training}_last_epoch.pth"
    log_filename = f"train_registration_training_id_{args.id_training}.log"
    path_out_model = os.path.join(args.path_out_dir, model_filename)
    path_logger = os.path.join(args.path_out_dir, log_filename)

    print(f"Training on device: {device}")
    print(f"Model will be saved to: {path_out_model}")
    print(f"Log will be saved to: {path_logger}")

    # Start training and validation
    train_val(
        train_loader,
        validation_loader,
        device,
        args.learning_rate,
        args.weight_decay,
        args.num_epochs,
        path_logger,
        path_out_model,
        args.path_out_dir,
        args.id_training,
        args.use_scheduler,
        args.max_iterations,
    )


if __name__ == "__main__":
    main()
