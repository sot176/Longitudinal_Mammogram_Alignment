import argparse
import random
from torch.utils.data import DataLoader

from src.dataloaders.risk_prediction.dataset_csaw import  BreastCancerRiskDatasetCSAWCC
from src.dataloaders.risk_prediction.dataset_embed import  BreastCancerRiskDataset
from src.evaluate.test_risk_prediction import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test script for breast cancer risk prediction.")

    # Paths and setup
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with data info.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root.")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Directory for saved models.")
    parser.add_argument("--path_test_folder", type=str, required=True, help="Output folder for test results.")

    # Training ID and epochs
    parser.add_argument("--id_training", type=int, required=True, help="ID of training run.")
    parser.add_argument("--num_epoch", type=int, required=True, help="Number of epochs (used to decide model path).")

    # Dataset
    parser.add_argument("--dataset", type=str, choices=["CSAW", "EMBED"], required=True, help="Dataset to use.")

    # Model config flags
    parser.add_argument("--use_img_alignment", type=str, help="Use image alignment model.")
    parser.add_argument("--use_img_feat_alignment_way1", type=str, help="Use image alignment deformationfield in feature space model.")
    parser.add_argument("--no_feat_alignment", type=str, help="Flag or path for no-feature-alignment model.")
    parser.add_argument("--early_stop", type=str, help="Use early stopping model instead of best or last.")

    # Loader settings
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--schuffle", default=False, type=bool)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--seed", default=2023, type=int)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    print("Creating test dataset...")
    dataset_cls = BreastCancerRiskDatasetCSAWCC if args.dataset == "CSAW" else BreastCancerRiskDataset
    test_dataset = dataset_cls(args.csv_file, args.data_root, "test", transforms=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
    )

    # Determine model path
    if args.early_stop:
        model_filename = f"early_stopping_risk_prediction_id-{args.id_training}.pth"
    elif args.num_epoch == 99:
        model_filename = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"
    else:
        model_filename = f"best_model_risk_prediction_id-{args.id_training}.pth"

    path_model = os.path.join(args.path_out_dir, model_filename)
    path_logger = os.path.join(args.path_test_folder, f"test_risk_prediction_training_id_{args.id_training}.log")

    print(f"Model path: {path_model}")
    print(f"Logger path: {path_logger}")

    # Run appropriate test method
    if args.use_img_alignment == "True":
        test_img_alignment_risk_pred_combined_train(
            test_loader,
            dev,
            path_model,
            args.path_test_folder,
            path_logger,
            args.use_img_feat_alignment_way1,
            args.dataset,
        )
    else:
        test_jointly_feat_alignment_risk(
            test_loader,
            dev,
            path_model,
            args.path_test_folder,
            path_logger,
            args.no_feat_alignment,
        )

if __name__ == "__main__":
    main()
