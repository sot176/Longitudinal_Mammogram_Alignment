import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

def imgunit16(img):
    mammogram_scaled = ((img.astype(np.float32) - img.min()) /
                        (img.max() - img.min()) * 65535)
    return mammogram_scaled


class BreastCancerRiskDatasetCSAWCC(Dataset):

    def __init__(self, csv_file, image_dir, mode, transforms=None, n_years=5):
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.csv_data = self.data.set_index("file_name").to_dict(
            orient="index")

        self.data_dir = image_dir
        self.transform = transforms
        self.n_years = n_years
        self.mode = mode
        self.image_data = self._load_image_data()
        self.patient_view_pairs = []

        for patient_id, view_images in self.image_data.items():
            if len(view_images) > 1:
                for i in range(len(view_images) - 1):
                    self.patient_view_pairs.append(
                        (patient_id, view_images, i))

        self.num_elements = len(self.patient_view_pairs)

    def map_density(self, value):
        mapping = {
            "0-33": "A",
            "33-67": "B",
            "67-100": "C",
        }
        return mapping.get(value, "NA")

    def _load_image_data(self):
        """
        Load image data into a dictionary grouped by Patient ID, Laterality, and View.
        This ensures that prior and current images are from the same patient, same laterality, and same view.
        """
        image_data = defaultdict(list)
        csv_data = self.csv_data

        image_dir_path = os.path.join(self.data_dir,
                                      self.mode).replace("\\", "/")
        for filename in os.listdir(image_dir_path):
            if filename in csv_data:
                file_info = csv_data[filename]
                patient_id = str(file_info["patient_id"])
                laterality = file_info["laterality"]  # Include laterality
                view = file_info["viewposition"]
                exam_year = file_info["exam_year"]
                years_to_cancer = file_info["years_to_cancer"]
                years_last_followup = file_info["years_to_last_followup"]
                density = file_info["density"]
                # Group by (patient_id, laterality, view)
                image_data[(patient_id, laterality, view)].append(
                    (filename, exam_year, years_to_cancer, years_last_followup,
                     density))

        # Sort images by exam_year in ascending order (oldest first)
        for key in image_data:
            image_data[key].sort(key=lambda x: x[1])

        return image_data

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        # Check if patient-view pairs exist and index is valid
        if not self.patient_view_pairs:
            raise ValueError("No valid patient-view pairs generated.")
        if idx < 0 or idx >= len(self.patient_view_pairs):
            raise ValueError(
                f"Index {idx} is out of range. Valid range: 0 to {len(self.patient_view_pairs) - 1}"
            )

        # Access patient-view pair
        patient_id, view_images, image_idx = self.patient_view_pairs[idx]

        # Sort images by year (ascending)
        view_images.sort(key=lambda x: x[1])

        # Ensure at least two images for valid current-prior pair
        if len(view_images) < 2:
            raise ValueError(
                f"Not enough images for patient {patient_id} and view {view_images[0][1]}."
            )

        # Randomly select an image index ensuring a previous image exists
        selected_idx = random.randint(1, len(view_images) - 1)
        selected_image = view_images[selected_idx]  # Current image info
        previous_image = view_images[selected_idx - 1]  # Prior image info

        # Unpack image metadata
        (current_image_file, current_year, time_to_cancer, years_last_followup, density) = selected_image
        (previous_image_file, previous_year, time_to_cancer_prior, years_last_followup_prior, _) = previous_image

        # Calculate time gap and clamp to max 5 years
        time_gap = abs(current_year - previous_year)
        time_gap = min(time_gap, 5)

        # Build image paths
        current_image_path = os.path.join(self.data_dir, self.mode, current_image_file)
        previous_image_path = os.path.join(self.data_dir, self.mode, previous_image_file)

        # Load images using PIL
        current_image_pil = Image.open(current_image_path)
        previous_image_pil = Image.open(previous_image_path)

        # Convert images to numpy arrays and normalize to [0, 1]
        current_image_np = np.array(current_image_pil).astype(np.float32) / 255.0
        previous_image_np = np.array(previous_image_pil).astype(np.float32) / 255.0

        # Scale images to 16-bit range
        current_image_np *= 65535
        previous_image_np *= 65535

        # Convert to tensors with shape (1, H, W) and half precision
        current_image = torch.from_numpy(current_image_np).unsqueeze(0).to(torch.float16)
        previous_image = torch.from_numpy(previous_image_np).unsqueeze(0).to(torch.float16)

        # Apply transforms if specified, convert to float32 and squeeze batch dim
        if self.transform is not None:
            current_image = self.transform(current_image.to(torch.float32)).squeeze(0)
            previous_image = self.transform(previous_image.to(torch.float32)).squeeze(0)

        if int(time_to_cancer) == 0:
            time_to_cancer = 1
        if int(time_to_cancer_prior) == 0:
            time_to_cancer_prior = 1

        # Map density using provided mapping function
        density = self.map_density(density)

        # Initialize target arrays
        target = np.zeros(6)
        target_prior = np.zeros(6)
        years_last_followup = int(years_last_followup)
        time_to_cancer = int(time_to_cancer) - 1
        time_to_cancer_prior = int(time_to_cancer_prior) - 1

        any_cancer = time_to_cancer < 5
        # If the patient has cancer, mark the event year and later with 1
        if any_cancer:
            time_at_event = int(time_to_cancer)
            event_observed = 1
            time_at_event_prior = int(time_to_cancer_prior)
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

        return {
            "current_image": current_image,
            "previous_image": previous_image,
            "current_image_id": current_image_file,
            "previous_image_id": previous_image_file,
            "event_observed": event_observed,
            "event_times": time_at_event,
            "time_gap": time_gap,
            "y_mask": y_mask,
            "y_mask_prior": y_mask_prior,
            "target": target,
            "target_prior": target_prior,
            "density": density,
        }
