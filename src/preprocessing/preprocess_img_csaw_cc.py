import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers import apply_windowing


def resize_with_alignment(image, target_width, target_height, align="L"):
    """
    Resize the image to maximize the fit within target dimensions while maintaining aspect ratio,
    with minimal padding to preserve the breast tissue size.

    Arguments:
        image: Input image.
        target_width: Target width (e.g., 1024 pixels).
        target_height: Target height (e.g., 512 pixels).
        align: Alignment for the resized image ("left" or "right").

    Returns:
        Resized image with minimal padding to fit the target size, aligned to the specified side.
    """
    original_height, original_width = image.shape[:2]

    # Compute the scaling factor to maximize fit
    # Compute the scaling factor to maximize fit
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    # Create a black canvas of the target size
    result = np.zeros((target_height, target_width), dtype=np.uint8)

    if align == "Left":  # Align to the left
        y_offset = (target_height - new_height) // 2
        x_offset = 0
    elif align == "Right":  # Align to the right
        y_offset = (target_height - new_height) // 2
        x_offset = target_width - new_width

    result[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return result


def preprocess_mammogram_with_largest_contour(csv_file, output_path):
    """
    Preprocesses DICOM mammography images listed in a CSV file by keeping only the largest contour
    (assumed to be the breast tissue), resizing to the target size, and saving as an 8-bit PNG.

    Parameters:
    csv_file (str): Path to the CSV file containing DICOM file paths and associated metadata.
    output_path_uncompressed (str): Directory to save the uncompressed DICOM files.
    output_path (str): Directory to save the processed 8-bit PNG files.
    """
    # Ensure the output directories exist
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # Read the CSV file to get the DICOM file paths and metadata
    df = pd.read_csv(csv_file)
    # Print the current working directory

    # Iterate over each DICOM file listed in the CSV
    for _, row in df.iterrows():
        dicom_path = row["anon_dicom_path_local"]  # Path to the DICOM file
        laterality = row["laterality"]  # Laterality (e.g., 'L' or 'R')

        # Ensure the file exists before processing
        if not os.path.exists(dicom_path):
            print(f"Skipping {dicom_path}, file not found.")
            continue

        # Create the output DICOM path
        dicom_filename = os.path.basename(dicom_path)
        # Load the DICOM file
        dicom_img = pydicom.dcmread(dicom_path, force=True)
        img_array = dicom_img.pixel_array

        # The DICOM PhotometricInterpretation attribute determines whether the minimum pixel value is black or white
        if dicom_img.PhotometricInterpretation == "MONOCHROME1":
            img_array = np.amax(img_array) - img_array
        else:
            img_array = img_array

        img = apply_windowing(img_array, dicom_img)

        # Normalize the mammogram
        img = (img.astype(float) - img.min()) / (img.max() - img.min())
        img_gray = (img * 255).astype(np.uint8)
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )

        rect_areas = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for i in range(1, len(contours)):
            (x, y, w, h) = cv2.boundingRect(contours[i])
            area = w * h
            rect_areas.append(area)
            # if area <= 100:
            if laterality == "Right":
                if x <= 1600 or x + w <= 1600 and y <= 700:
                    img_gray[y : y + h, x : x + w] = 0
            if laterality == "Left":
                if (
                    x >= img_gray.shape[1] - 1600
                    or x + w >= img_gray.shape[1] - 1600
                    and y <= 700
                ):
                    img_gray[y : y + h, x : x + w] = 0

        resized_image = resize_with_alignment(
            img_gray, 512, 1024, laterality
        )  # resize(breast_only, ( 1024, 512), anti_aliasing=True)
        final_image_pil = Image.fromarray(resized_image.astype(np.uint8))

        filename = os.path.splitext(dicom_filename)[0] + ".png"

        png_path = os.path.join(output_path, filename)
        final_image_pil.save(png_path)
        print(f"Processed {dicom_filename} and saved as {png_path}")
