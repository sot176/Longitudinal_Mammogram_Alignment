import os
import random
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def imgunit16(img):
    mammogram_scaled = (
        (img.astype(np.float32) - img.min()) / (img.max() - img.min()) * 65535
    )
    return mammogram_scaled


class BreastCancerRiskDataset(Dataset):

    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        """
        Args:
            csv_file (str): Path to the CSV file containing patient data.
            image_dir (str): Directory containing mammogram images.
            transform (callable, optional): Optional transform to be applied on an image.
            n_years (int): Number of years for risk prediction (default: 5).
        """
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode
        self.image_data = self._load_image_data()
        # Precompute the patient-view pairs once during initialization
        self.patient_view_pairs = []

        for patient_id, view_images in self.image_data.items():
            if (len(view_images) > 1
                ):  # Only consider patient-view pairs with multiple images
                for i in range(len(view_images) - 1):
                    self.patient_view_pairs.append(
                        (patient_id, view_images, i))

        # Store the length of the dataset (number of valid patient-view pairs)
        self.num_elements = len(self.patient_view_pairs)

    def map_density(self, value):
        if value == 1:
            return "A"
        elif value == 2:
            return "B"
        elif value == 3:
            return "C"
        elif value == 4:
            return "D"
        else:
            return "NA"

    def _load_image_data(self):
        """
        Load image data into a dictionary grouped by patient ID and view type.
        """
        image_data = defaultdict(list)

        # Regex pattern to extract the necessary information from the filename
        pattern = r"(\d+)_([A-Z]+_[A-Z]+)_(\d{4})-\d{2}-\d{2}_.*\.png"

        # Iterate over the files in the image directory
        for filename in os.listdir(
                os.path.join(self.data_dir, self.mode).replace("\\", "/")):
            # Match the filename to extract the relevant info
            match = re.match(pattern, filename)
            if match:
                patient_id = match.group(1)
                view = match.group(2)
                year = int(match.group(3))

                # Store the filename with patient_id and view as the key
                image_data[(patient_id, view)].append((filename, year))

        return image_data

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        if not self.patient_view_pairs:
            raise ValueError("No valid patient-view pairs generated.")
        if idx < 0 or idx >= len(self.patient_view_pairs):
            raise ValueError(
                f"Index {idx} is out of range. Valid range: 0 to {len(self.patient_view_pairs) - 1}"
            )

            # Access the patient-view pair
        patient_id, view_images, image_idx = self.patient_view_pairs[idx]

        # Helper function to extract datetime from filename
        def extract_date_from_filename(filename):
            date_str = filename.split("_")[3]  # e.g., "2019-07-13"
            return datetime.strptime(date_str, "%Y-%m-%d")

        # Sort images by date (ascending)
        view_images.sort(key=lambda x: x[1])

        # Check at least two images to form a valid pair
        if len(view_images) < 2:
            raise ValueError(
                f"Not enough images for patient {patient_id} and view {view_images[0][1]}."
            )

        # Randomly select an image index ensuring a previous image exists
        selected_idx = random.randint(1, len(view_images) - 1)

        selected_image = view_images[selected_idx]  # current image (filename, date)
        previous_image = view_images[selected_idx - 1]  # previous image (filename, date)

        # Extract filenames
        current_image_file = selected_image[0]
        previous_image_file = previous_image[0]

        # Extract years from filenames
        current_year = extract_date_from_filename(current_image_file)
        previous_year = extract_date_from_filename(previous_image_file)

        # Compute time gap capped at 5 years
        time_gap = abs(current_year.year - previous_year.year)
        time_gap = min(time_gap, 5)

        # Construct full paths
        current_image_path = os.path.join(self.data_dir, self.mode, current_image_file)
        previous_image_path = os.path.join(self.data_dir, self.mode, previous_image_file)

        # Load images and normalize
        current_image_pil = Image.open(current_image_path)
        previous_image_pil = Image.open(previous_image_path)

        current_image_np = np.array(current_image_pil)
        current_image_np = imgunit16(current_image_np)
        current_image_np = (current_image_np - 7047.99) / 12005.5
        current_image = (torch.from_numpy(current_image_np).unsqueeze(0).to(torch.float16))
        previous_image_np = np.array(previous_image_pil)
        previous_image_np = imgunit16(previous_image_np)
        previous_image_np = (previous_image_np - 7047.99) / 12005.5

        previous_image = (torch.from_numpy(previous_image_np).unsqueeze(0).to(torch.float16))

        if self.transform is not None:
            current_image = self.transform(current_image.to(torch.float32))
            previous_image = self.transform(previous_image.to(torch.float32))
            current_image = current_image.squeeze(0)
            previous_image = previous_image.squeeze(0)

        # Extract patient ID, laterality, and view from the current and prior image filenames
        parts = current_image_file.split("_")
        patient_id = parts[0]
        image_laterality = parts[1]
        view = parts[2]

        parts_prior = previous_image_file.split("_")
        patient_id_prior = parts_prior[0]
        image_laterality_prior = parts_prior[1]
        view_prior = parts_prior[2]

        # Ensure 'study_date_anon' and study dates are datetime objects
        self.data["study_date_anon"] = pd.to_datetime(self.data["study_date_anon"], format="%Y-%m-%d")
        study_date = pd.to_datetime(current_year, format="%Y-%m-%d")
        study_date_prior = pd.to_datetime(previous_year, format="%Y-%m-%d")

        # Find corresponding rows in the CSV for current and prior images
        matching_row = self.data[
            (self.data["patient_id"].astype(str) == str(patient_id)) &
            (self.data["ImageLateralityFinal"].astype(str) == str(image_laterality)) &
            (self.data["view"].astype(str) == str(view)) &
            (self.data["study_date_anon"] == study_date)
            ]

        matching_row_prior = self.data[
            (self.data["patient_id"].astype(str) == str(patient_id_prior)) &
            (self.data["ImageLateralityFinal"].astype(str) == str(image_laterality_prior)) &
            (self.data["view"].astype(str) == str(view_prior)) &
            (self.data["study_date_anon"] == study_date_prior)
            ]

        # Extract relevant clinical variables from matched rows
        years_last_followup_prior = matching_row_prior["years_last_followup"].values[0]
        years_last_followup = matching_row["years_last_followup"].values[0]

        time_to_cancer = matching_row["Time_to_Cancer_Years"].values[0]
        time_to_cancer_prior_img = matching_row_prior["Time_to_Cancer_Years"].values[0]

        density = matching_row["density"].values[0]
        density = self.map_density(density)

        if time_to_cancer == 0:
            time_to_cancer = 1
        if pd.isna(time_to_cancer):  # Check if the value is NaN
            time_to_cancer = 6
        if time_to_cancer_prior_img == 0:
            time_to_cancer_prior_img = 1
        if pd.isna(time_to_cancer_prior_img):  # Check if the value is NaN
            time_to_cancer_prior_img = 6

        years_last_followup = int(years_last_followup)
        time_to_cancer = int(time_to_cancer)
        time_to_cancer_prior_img = int(time_to_cancer_prior_img)

        time_to_cancer = time_to_cancer - 1

        time_to_cancer_prior_img = time_to_cancer_prior_img - 1
        any_cancer = time_to_cancer < 5

        # Initialize the sequence with zeros
        target = np.zeros(6)
        target_prior = np.zeros(6)

        # If the patient has cancer, mark the event year and later with 1
        if any_cancer:
            time_at_event = int(time_to_cancer)
            event_observed = 1
            time_at_event_prior = int(time_to_cancer_prior_img)
            target[time_at_event:] = 1
            target_prior[time_at_event_prior:] = 1
            target[-1] = 0
            target_prior[-1] = 0
        else:
            # If no cancer, use the last follow-up year
            if years_last_followup == 0:
                years_last_followup = 1
            time_at_event = int(min(years_last_followup, 5))
            time_at_event = time_at_event - 1
            event_observed = 0
            time_at_event_prior = int(min(years_last_followup_prior, 5))
            time_at_event_prior = time_at_event_prior - 1
            target[-1] = 1
            target_prior[-1] = 1

        y_mask = np.array([1] * (time_at_event + 1) + [0] *
                          (6 - (time_at_event + 1)))
        y_mask_prior = np.array([1] * (time_at_event_prior + 1) + [0] *
                                (6 - (time_at_event_prior + 1)))
        y_mask = y_mask[:6]
        y_mask_prior = y_mask_prior[:6]
        # Create a mask for valid time points (before the last event or follow-up)
        event_times = time_at_event

        data = {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": current_image_file,
            "previous_image_id": previous_image_file,
            "event_observed": event_observed,
            "event_times": event_times,
            "time_gap": time_gap,
            "y_mask": y_mask,
            "y_mask_prior": y_mask_prior,
            "target": target,
            "target_prior": target_prior,
            "density": density,
        }
        return data

