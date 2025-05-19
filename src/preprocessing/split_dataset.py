import os
import random
import shutil
from collections import defaultdict
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data_and_copy_images_risk(
    source_dir,
    train_dir,
    val_dir,
    test_dir,
):
    """
    Split the DataFrame by patient ID and copy images into train, val, and test directories.

    Parameters:
    - df: DataFrame with columns 'patient_id' and 'image_path'
    - train_dir, val_dir, test_dir: Target directories for train, validation, and test images
    - train_size, val_size, test_size: Proportions for splitting the data
    """

    # Ensure target directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    val_size = 0.2  # Validation size (percentage of the subset)
    test_size = 0.3  # Test size (percentage of the subset)

    # Extract patient IDs from filenames
    filenames = [
        f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
    ]
    patient_files = defaultdict(list)

    for file in filenames:
        parts = file.split("_")
        if len(parts) < 4:
            print(f"Skipping invalid file: {file}")
            continue

        patient_id = parts[0]
        date_str = parts[3].split(".")[0]
        try:
            exam_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Skipping invalid date in file: {file}")
            continue
        # Group by patient ID
        patient_files[patient_id].append(file)

    # Get all patient ids
    patient_ids = list(patient_files.keys())
    print(f"Total number of patients: {len(patient_ids)}")

    # Split all available patients into train, val, and test sets
    train_ids, temp_ids = train_test_split(
        patient_ids, test_size=(val_size + test_size), random_state=42
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(test_size / (val_size + test_size)), random_state=42
    )

    print(
        f"Train IDs: {len(train_ids)}, Validation IDs: {len(val_ids)}, Test IDs: {len(test_ids)}"
    )

    # Copy files to their respective directories
    for group, ids, target_dir in [
        ("Train", train_ids, train_dir),
        ("Validation", val_ids, val_dir),
        ("Test", test_ids, test_dir),
    ]:
        print(f"\nCopying {group} images...")
        for patient_id in ids:
            for file in patient_files[patient_id]:
                src_path = os.path.join(source_dir, file)
                dest_path = os.path.join(target_dir, file)
                shutil.copy(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")


def split_data_and_copy_images_registration_dataset(
    source_dir,
    train_dir,
    val_dir,
    test_dir,
):
    """
    Split the DataFrame by patient ID and copy images into train, val, and test directories.
    Parameters:
    - df: DataFrame with columns 'patient_id' and 'image_path'
    - train_dir, val_dir, test_dir: Target directories for train, validation, and test images
    - train_size, val_size, test_size: Proportions for splitting the data
    """

    # Ensure target directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    subset_size = 1000
    val_size = 0.25  # Validation size (percentage of the subset)
    test_size = 0.25  # Test size (percentage of the subset)
    max_files_per_group = 2  # Max images per (laterality, view) group for a patient

    # Extract patient IDs from filenames
    filenames = [
        f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
    ]
    patient_files = defaultdict(lambda: defaultdict(list))

    for file in filenames:
        parts = file.split("_")
        if len(parts) < 4:
            print(f"Skipping invalid file: {file}")
            continue

        patient_id = parts[0]
        laterality = parts[1]
        view = parts[2]
        date_str = parts[3].split(".")[0]
        try:
            exam_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Skipping invalid date in file: {file}")
            continue
        # Group by patient ID, laterality, and view
        key = (laterality, view)
        patient_files[patient_id][key].append((exam_date, file))

    # Limit files per group for each patient
    limited_patient_files = {}
    for patient_id, groups in patient_files.items():
        limited_patient_files[patient_id] = []
        for key, images in groups.items():
            if len(images) < max_files_per_group:
                continue
            selected_images = random.sample(images, max_files_per_group)
            limited_patient_files[patient_id].extend(
                [file for _, file in selected_images]
            )

    # Select a subset of patients
    patient_ids = list(limited_patient_files.keys())
    if len(patient_ids) < subset_size:
        print(
            f"Warning: Requested subset size ({subset_size}) exceeds available patients ({len(patient_ids)})."
        )
        subset_size = len(patient_ids)

    subset_patient_ids = random.sample(patient_ids, subset_size)

    # Split the subset into train, val, and test sets
    train_ids, temp_ids = train_test_split(
        subset_patient_ids, test_size=(val_size + test_size), random_state=42
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(test_size / (val_size + test_size)), random_state=42
    )

    print(
        f"Train IDs: {len(train_ids)}, Validation IDs: {len(val_ids)}, Test IDs: {len(test_ids)}"
    )

    # Copy files to their respective directories
    for group, ids, target_dir in [
        ("Train", train_ids, train_dir),
        ("Validation", val_ids, val_dir),
        ("Test", test_ids, test_dir),
    ]:
        print(f"\nCopying {group} images...")
        for patient_id in ids:
            for file in limited_patient_files[patient_id]:
                src_path = os.path.join(source_dir, file)
                dest_path = os.path.join(target_dir, file)
                shutil.copy(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")


def split_data_and_copy_images(
    source_dir,
    train_dir,
    val_dir,
    test_dir,
    val_size=0.25,
    test_size=0.25,
    subset_size=1000,
):
    """
    Splits the dataset by patient ID, selecting a subset of patients (default: 1000).
    Randomly selects two images per laterality/view per patient and copies them to train, validation, and test directories.

    Parameters:
    - source_dir: Directory containing the images.
    - train_dir, val_dir, test_dir: Directories where train, val, and test images will be copied.
    - train_size, val_size, test_size: Proportions for dataset splitting.
    - subset_size: Number of patients to include in the dataset.
    """

    # Ensure target directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    max_files_per_group = 2  # Max images per (laterality, view) group for a patient

    # Extract patient-specific filenames
    filenames = [
        f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
    ]
    patient_files = defaultdict(lambda: defaultdict(list))

    for file in filenames:
        parts = file.split("_")
        if len(parts) < 5:  # Ensure valid filename structure
            print(f"Skipping invalid file: {file}")
            continue

        patient_id, date, laterality, view, _ = parts[:5]  # Extract relevant parts

        # Group images by patient, laterality, and view
        key = (laterality, view)
        patient_files[patient_id][key].append(file)

    # Limit to two images per (laterality, view) for each patient
    limited_patient_files = {}
    for patient_id, groups in patient_files.items():
        limited_patient_files[patient_id] = []
        for key, images in groups.items():
            if len(images) < max_files_per_group:
                continue  # Skip if fewer than 2 images exist
            selected_images = random.sample(images, max_files_per_group)
            limited_patient_files[patient_id].extend(selected_images)

    # Get all patient IDs with selected images
    patient_ids = list(limited_patient_files.keys())

    # Ensure subset size does not exceed available patients
    if len(patient_ids) < subset_size:
        print(
            f"Warning: Requested subset size ({subset_size}) exceeds available patients ({len(patient_ids)}). Using all available patients."
        )
        subset_size = len(patient_ids)

    # Select a random subset of patients
    subset_patient_ids = random.sample(patient_ids, subset_size)

    # Split into train, validation, and test sets
    train_ids, temp_ids = train_test_split(
        subset_patient_ids, test_size=(val_size + test_size), random_state=42
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(test_size / (val_size + test_size)), random_state=42
    )

    print(
        f"Train Patients: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}"
    )

    # Function to copy files
    def copy_files(patient_list, target_dir, group_name):
        print(f"\nCopying {group_name} images...")
        for patient_id in patient_list:
            for file in limited_patient_files[patient_id]:
                src_path = os.path.join(source_dir, file)
                dest_path = os.path.join(target_dir, file)
                shutil.copy(src_path, dest_path)
                print(f"Copied {file} to {target_dir}")

    # Copy images to respective directories
    copy_files(train_ids, train_dir, "Train")
    copy_files(val_ids, val_dir, "Validation")
    copy_files(test_ids, test_dir, "Test")


def split_data_and_copy_images_csaw_risk(
    df,
    source_dir,
    train_dir,
    val_dir,
    test_dir,
):
    """
    Splits the dataset by patient ID and copies all images for each patient into train, validation, and test directories.

    Parameters:
    - source_dir: Directory containing the images.
    - train_dir, val_dir, test_dir: Directories where train, val, and test images will be copied.
    - train_size, val_size, test_size: Proportions for dataset splitting (default: train 50%, val 20%, test 30%).
    """

    # Ensure target directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    df = pd.read_csv(df)
    # Iterate through the dataframe and copy images based on their split group
    for _, row in df.iterrows():
        image_filename = row["file_name"]
        split_group = row["split_group"]

        # Determine the target directory based on the split group
        if split_group == "train":
            target_dir = train_dir
        elif split_group == "val":
            target_dir = val_dir
        elif split_group == "test":
            target_dir = test_dir
        else:
            continue  # Skip if the split group is not one of the expected values

            # Define source and destination file paths
        src_path = os.path.join(source_dir, image_filename)
        dest_path = os.path.join(target_dir, image_filename)

        # Copy the image to the target directory
        shutil.copy(src_path, dest_path)
        print(f"Copied {image_filename} to {target_dir}")
