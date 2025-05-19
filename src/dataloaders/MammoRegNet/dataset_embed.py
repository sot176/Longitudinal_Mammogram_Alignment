import os
import random
import re
from collections import defaultdict
from datetime import datetime

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import rgb_to_grayscale

class EMBEDRegistrationDataset(Dataset):

    def __init__(self, data_dir, mode, transforms=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transforms = transforms
        self.image_data = self._load_image_data()

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
        """
        Return the number of image pairs available for registration.
        """
        """
            Return the number of valid image pairs available for registration.
            """

        self.num_elements = sum(
            len(images) - 1 for images in self.image_data.values())

        return self.num_elements

    def __getitem__(self, idx):
        """
        Return a fixed and moving image pair for the registration model.
        """
        # Ensure you are getting the correct patient-view pair
        patient_view_pairs = []

        for patient_id, view_images in self.image_data.items():
            if (len(view_images) > 1
                ):  # Only consider patient-view pairs with multiple images
                # For each patient-view pair, generate image pairs
                for i in range(len(view_images) - 1):
                    patient_view_pairs.append((patient_id, view_images, i))

        # Get the patient_id, images list, and selected index
        patient_id, view_images, image_idx = patient_view_pairs[idx]

        # Extract the date from the filename and sort images by date (year, month, day)
        # Assuming filename pattern: patient_id_view_YYYY-MM-DD_imageid_label.png
        def extract_date_from_filename(filename):
            # Extract date part from the filename, assuming format: patient_id_view_YYYY-MM-DD_imageid_label.png
            date_str = filename.split("_")[3]  # e.g., "2019-07-13"
            return datetime.strptime(date_str,
                                     "%Y-%m-%d")  # Convert to datetime object

        # Sort images by extracted date (year, month, day)
        view_images.sort(key=lambda x: extract_date_from_filename(x[0])
                         )  # Sorting by full date

        # The most recent image (fixed) and a random previous one (moving)
        fixed_image_data = view_images[-1]  # Most recent image
        moving_image_data = random.choice(
            view_images[:-1])  # Random previous image

        # Get file paths
        fixed_image_path = os.path.join(self.data_dir, self.mode,
                                        fixed_image_data[0])
        moving_image_path = os.path.join(self.data_dir, self.mode,
                                         moving_image_data[0])

        img_fix_id = str(fixed_image_data[0])
        img_mov_id = str(moving_image_data[0])

        # Open images
        fixed_image = Image.open(fixed_image_path)
        moving_image = Image.open(moving_image_path)
        img_fix = np.array(fixed_image)
        img_fixed = torch.from_numpy(img_fix.astype(np.float32)).unsqueeze(0)
        img_fixed = img_fixed.to(torch.float32)
        img_mov = np.array(moving_image)
        img_moving = torch.from_numpy(img_mov.astype(np.float32)).unsqueeze(0)
        img_moving = img_moving.to(torch.float32)

        # Apply any transformations
        if self.transforms is not None:
            img_fixed = img_fixed.squeeze(0)
            img_moving = img_moving.squeeze(0)
            img_fixed = cv2.cvtColor(np.array(img_fixed), cv2.COLOR_GRAY2RGB)
            img_moving = cv2.cvtColor(np.array(img_moving), cv2.COLOR_GRAY2RGB)
            img_fixed = self.transforms(image=img_fixed)["image"]
            img_moving = self.transforms(image=img_moving)["image"]
            img_moving = rgb_to_grayscale(img_moving, num_output_channels=1)
            img_fixed = rgb_to_grayscale(img_fixed, num_output_channels=1)


        data = {
            "img_fix": img_fixed,
            "img_mov": img_moving,
            "img_fix_id": img_fix_id,
            "img_mov_id": img_mov_id,
        }

        return data

