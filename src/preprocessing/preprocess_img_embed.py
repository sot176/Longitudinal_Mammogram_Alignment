import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers import apply_windowing


def min_max_windowing(dicom_data):
    """Apply min-max windowing to the DICOM pixel data."""
    pixel_array = dicom_data.pixel_array.astype(np.float32)
    min_val, max_val = pixel_array.min(), pixel_array.max()
    windowed_image = 255 * (pixel_array - min_val) / (max_val - min_val)
    return windowed_image.astype(np.uint8)


def apply_voi_lut(dicom_data):
    """Apply VOI LUT if available in DICOM metadata."""
    if "VOILUTSequence" in dicom_data:
        return pydicom.pixel_data_handlers.apply_voi_lut(
            dicom_data.pixel_array, dicom_data
        )
    return dicom_data.pixel_array


def decompress_dicom(file_path):
    try:
        # Attempt to read the DICOM file
        dataset = pydicom.dcmread(file_path)

        # Check if the file contains PixelData (which indicates it is an image)
        if "PixelData" not in dataset:
            print(f"Skipping {file_path}: No Pixel Data found.")
            return None  # Skip this file by returning None

        # Decompress the image if PixelData is found
        dataset.decompress()  # Decompress if the file has PixelData
        print(f"Decompressed {file_path}")
        return dataset  # Return the decompressed DICOM dataset

    except (pydicom.errors.InvalidDicomError, AttributeError, KeyError) as e:
        # Handle cases where the file is not a valid DICOM or has missing metadata
        print(
            f"Skipping {file_path}: Invalid DICOM file or missing required metadata. Error: {e}"
        )
        return None  # Return None to indicate skipping this file

    except Exception as e:
        # Handle any other unforeseen errors
        print(f"Skipping {file_path}: Unexpected error - {e}")
        return None  # Return None to skip this file


def check_and_flip_dicom(dicom_img):
    # Load the DICOM file

    # Check View position (left or right breast)
    laterality = dicom_img.ImageLaterality  # View position
    if laterality is None:
        raise ValueError("Image Laterality (0020,0062) is missing.")

    # Check Patient Orientation
    patient_orientation = dicom_img.PatientOrientation  # Patient Orientation
    print(
        "Patient orientation",
        patient_orientation[0],
        patient_orientation[1],
        "laterality",
        laterality,
    )
    # For simplicity, assume flipping horizontally is necessary if the laterality is "R"
    # and nipple points left, or "L" and nipple points right.

    # Initialize flipping logic
    flipHorz = False
    if patient_orientation[0] == "P":
        flipHorz = True
    else:
        flipHorz = False

    if laterality == "R":
        if patient_orientation[0] == "A":
            if patient_orientation[1] == "FL" or patient_orientation[1] == "L":
                flipHorz = True
            else:
                flipHorz = False
        else:
            flipHorz = False

    return flipHorz


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

    if align == "L":  # Align to the left
        y_offset = (target_height - new_height) // 2
        x_offset = 0
    elif align == "R":  # Align to the right
        y_offset = (target_height - new_height) // 2
        x_offset = target_width - new_width

    result[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return result


def preprocess_mammogram_with_largest_contour(csv_file, output_path, positive_group):
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
        if positive_group:
            patient_id = row["patient_id"]  # Patient ID
            study_date = row["study_date_anon"]  # Study Date
            laterality = row[
                "ImageLateralityFinal"
            ]  # Laterality (e.g., 'L' or 'R')
            dicom_path = row["file_path_dcm"]  # Path to the DICOM file
            view = row["view"]  # View position (e.g., 'MLO' or 'CC')
            label = row["Label"]  # Cancer label ('Cancer' or 'No Cancer')

        else:
            dicom_path = row["anon_dicom_path"]  # Path to the DICOM file
            patient_id = row["empi_anon"]  # Patient ID
            study_date = row["study_date_anon"]  # Study Date
            laterality = row[
                "ImageLateralityFinal"
            ]  # Laterality (e.g., 'L' or 'R')
            view = row["ViewPosition"]  # View position (e.g., 'MLO' or 'CC')

        # Split and extract
        last_part = dicom_path.split("/")[-1]  # Get the last part of the path
        numbers = last_part.split(".")[-2]  # Get the part before .dcm
        last_4_numbers = numbers[-4:]

        # Ensure the file exists before processing
        if not os.path.exists(dicom_path):
            print(f"Skipping {dicom_path}, file not found.")
            continue

        # Create the output DICOM path
        dicom_filename = os.path.basename(dicom_path)

        # Load the DICOM file
        # dicom_img = pydicom.dcmread(dicom_path,  force=True)
        dicom_img = decompress_dicom(dicom_path)
        if dicom_img is not None:
            img_array = dicom_img.pixel_array.astype(float)
            # Check if a horizontal flip is necessary
            horz = check_and_flip_dicom(dicom_img)
            if horz:
                img_array = np.fliplr(img_array)

            img = apply_windowing(img_array, dicom_img)
            img = (img.astype(float) - img.min()) / (img.max() - img.min()) * 255

            image = (img).astype(np.uint8)

            # Apply a binary threshold to create a binary image for contour detection
            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
            binary_image = binary_image.astype(np.uint8)

            # Detect contours
            contours, _ = cv2.findContours(
                binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Find the largest contour, assuming it corresponds to the breast tissue
                largest_contour = max(contours, key=cv2.contourArea)

                # Create a mask with only the largest contour
                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.drawContours(
                    mask, [largest_contour], -1, (255), thickness=cv2.FILLED
                )

                # Apply the mask to the image to keep only the breast area
                breast_only = cv2.bitwise_and(image, mask)

                # Resize the image to the target size (512x1024 in this case)
                resized_image = resize_with_alignment(
                    breast_only, 512, 1024, laterality
                )  # resize(breast_only, ( 1024, 512), anti_aliasing=True)
                resized_image = (
                    np.maximum(resized_image, 0) / resized_image.max()
                ) * 65536

                # Format the filename as 'patient_id_laterality_view_study_date-label.png'
                study_date_str = pd.to_datetime(study_date).strftime(
                    "%Y-%m-%d"
                )  # Format study date to YYYY-MM-DD

                if positive_group:
                    filename = f"{patient_id}_{laterality}_{view}_{study_date_str}_{last_4_numbers}_{label}.png"
                else:
                    filename = f"{patient_id}_{laterality}_{view}_{study_date_str}_{last_4_numbers}.png"

                # Output path for PNG image
                png_path = os.path.join(output_path, filename)
                final_image = np.uint16(resized_image)
                final_image = Image.fromarray(final_image)

                # Save the processed image as an 8-bit PNG
                final_image.save(png_path)

                print(f"Processed {dicom_filename} and saved as {png_path}")
            else:
                print(f"No contours found in {dicom_filename}, skipping.")
