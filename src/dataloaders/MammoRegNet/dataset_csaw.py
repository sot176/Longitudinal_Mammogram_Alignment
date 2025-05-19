import itertools
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import rgb_to_grayscale

class CSAWCCRegistrationDataset(Dataset):

    def __init__(self, data_dir, mode, transforms=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        if self.mode == "train" or self.mode == "val" or self.mode == "test":
            elements = [
                f for f in os.listdir(
                    os.path.join(self.data_dir, self.mode).replace("\\", "/"))
            ]
            combinations = list(itertools.combinations(elements, 2))
            self.combinations_same_patient = []
            for i in range(len(combinations)):
                if (combinations[i][0][:5] == combinations[i][1][:5]
                        and combinations[i][0][15:].split(".")[0][:-2]
                        == combinations[i][1][15:].split(".")[0][:-2]):
                    self.combinations_same_patient.append(combinations[i])
            self.num_elements = len(self.combinations_same_patient)

        return self.num_elements

    def __getitem__(self, index):
        # Ensure you are getting the correct patient-view pair
        if self.mode == "train":
            img_pairs_list = random.sample(self.combinations_same_patient,
                                           self.num_elements)
            img_dir = os.path.join(self.data_dir, self.mode).replace("\\", "/")
            img_pair = img_pairs_list[index]

            # Load images
            img_fix_path = os.path.join(img_dir,
                                        str(img_pair[0])).replace("\\", "/")
            img_mov_path = os.path.join(img_dir,
                                        str(img_pair[1])).replace("\\", "/")
            img_fix = np.array(Image.open(img_fix_path))
            img_fixed = torch.from_numpy(img_fix).unsqueeze(0)
            img_fixed = img_fixed.to(torch.float32)
            img_mov = np.array(Image.open(img_mov_path))
            img_moving = torch.from_numpy(img_mov).unsqueeze(0)
            img_moving = img_moving.to(torch.float32)

            img_mov_id = str(img_pair[1])
            img_fix_id = str(img_pair[0])

            data = {
                "img_fix": img_fixed,
                "img_mov": img_moving,
                "img_fix_id": img_fix_id,
                "img_mov_id": img_mov_id,
            }

        if self.mode == "val" or self.mode == "test":
            img_pairs_list = random.sample(self.combinations_same_patient,
                                           self.num_elements)

            img_dir = os.path.join(self.data_dir, self.mode).replace("\\", "/")

            img_pair = img_pairs_list[index]

            img_fix_path = os.path.join(img_dir,
                                        str(img_pair[0])).replace("\\", "/")
            img_mov_path = os.path.join(img_dir,
                                        str(img_pair[1])).replace("\\", "/")

            # Load images and segmentations
            img_fix = np.array(Image.open(img_fix_path))
            img_fixed = torch.from_numpy(img_fix).unsqueeze(0)
            img_fixed = img_fixed.to(torch.float32)
            img_mov = np.array(Image.open(img_mov_path))
            img_moving = torch.from_numpy(img_mov).unsqueeze(0)
            img_moving = img_moving.to(torch.float32)
            img_mov_id = str(img_pair[1])
            img_fix_id = str(img_pair[0])
            data = {
                "img_fix": img_fixed,
                "img_mov": img_moving,
                "img_fix_id": img_fix_id,
                "img_mov_id": img_mov_id,
            }

        if self.transforms is not None:
            if self.mode == "train" or self.mode == "test" and self.use_seg == "True":
                img_fixed = img_fixed.squeeze(0)
                img_moving = img_moving.squeeze(0)

                img_fixed = cv2.cvtColor(np.array(img_fixed),
                                         cv2.COLOR_GRAY2RGB)
                img_moving = cv2.cvtColor(np.array(img_moving),
                                          cv2.COLOR_GRAY2RGB)
                img_fixed = self.transforms(image=img_fixed)["image"]
                img_moving = self.transforms(image=img_moving)["image"]

                img_moving = rgb_to_grayscale(img_moving,
                                              num_output_channels=1)
                img_fixed = rgb_to_grayscale(img_fixed, num_output_channels=1)
                data = {
                    "img_fix": img_fixed,
                    "img_mov": img_moving,
                    "img_fix_id": img_fix_id,
                    "img_mov_id": img_mov_id,
                }

            if self.mode == "val" or self.mode == "test" and self.use_seg == "False":
                img_fixed = img_fixed.squeeze(0)
                img_moving = img_moving.squeeze(0)
                img_fixed = cv2.cvtColor(np.array(img_fixed),
                                         cv2.COLOR_GRAY2RGB)
                img_moving = cv2.cvtColor(np.array(img_moving),
                                          cv2.COLOR_GRAY2RGB)
                img_fixed = self.transforms(image=img_fixed)["image"]
                img_moving = self.transforms(image=img_moving)["image"]
                img_moving = rgb_to_grayscale(img_moving,
                                              num_output_channels=1)
                img_fixed = rgb_to_grayscale(img_fixed, num_output_channels=1)
                data = {
                    "img_fix": img_fixed,
                    "img_mov": img_moving,
                    "img_fix_id": img_fix_id,
                    "img_mov_id": img_mov_id,
                }

        return data
