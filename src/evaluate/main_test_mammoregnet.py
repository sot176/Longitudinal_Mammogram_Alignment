import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

from src.dataloaders.MammoRegNet.dataset_csaw import CSAWCCRegistrationDataset
from src.dataloaders.MammoRegNet.dataset_embed import EMBEDRegistrationDataset
from src.evaluate.test_mammoregnet import test


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test script for mammogram registration.")

    # Paths and I/O
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Directory where model outputs will be saved.")
    parser.add_argument("--path_test_folder", type=str, required=True, help="Folder to store test results.")

    # Training and model settings
    parser.add_argument("--id_training", type=int, required=True, help="Unique training ID for model identification.")
    parser.add_argument("--num_epoch", type=int, required=True,
                        help="Number of training epochs used (to select model).")
    parser.add_argument("--dataset", type=str, choices=["EMBED", "CSAW"], required=True, help="Dataset name.")

    # DataLoader configuration
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader worker threads.")
    parser.add_argument("--shuffle", type=bool, default=False, help="Whether to shuffle the test data.")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Use pinned memory for DataLoader.")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed for reproducibility.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    # use GPU in deterministic mode for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"  # use the GPU if available

    # Determine model filename
    model_filename = (
        f"model_registration_training_id_{args.id_training}_last_epoch.pth"
        if args.num_epoch == 99
        else f"best_model_registration_id-{args.id_training}.pth"
    )

    model_path = os.path.join(args.path_out_dir, model_filename)
    log_filename = f"test_registration_training_id_{args.id_training}.log"
    log_path = os.path.join(args.path_test_folder, log_filename)

    print(f"Model path: {model_path}")
    print(f"Log path: {log_path}")

    # Dataset
    print("Creating test dataset...")
    dataset_cls = EMBEDRegistrationDataset if args.dataset == "EMBED" else CSAWCCRegistrationDataset
    test_dataset = dataset_cls(args.data_root, "test", transforms=None)

    # Dataloader setup
    print("Creating test dataloader...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    print("Running registration on test set...")
    test(test_loader, dev, model_path, log_path, args.path_test_folder)


if __name__ == "__main__":
    main()
