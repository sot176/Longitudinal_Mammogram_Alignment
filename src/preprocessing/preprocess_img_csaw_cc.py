import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers import apply_windowing



def resize_with_alignment(image, target_width, target_height, align="L"):
    """
    Resize image to fit target dimensions with preserved aspect ratio and minimal padding.

    Parameters:
        image (np.array): Input grayscale image.
        target_width (int): Width of the output image.
        target_height (int): Height of the output image.
        align (str): "Left" or "Right" to align the content accordingly.

    Returns:
        np.array: Resized and padded image.
    """
    original_height, original_width = image.shape[:2]

    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    result = np.zeros((target_height, target_width), dtype=np.uint8)

    y_offset = (target_height - new_height) // 2
    x_offset = 0 if align == "Left" else target_width - new_width

    result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return result


def preprocess_mammogram_with_largest_contour(csv_file, output_path):
    """
    Process mammogram DICOMs by retaining the main breast contour, resizing, and saving as PNG.

    Parameters:
        csv_file (str): Path to the CSV containing metadata and DICOM file paths.
        output_path (str): Directory where processed PNGs will be saved.
    """
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(csv_file)

    RIGHT_MASK_X_THRESHOLD = 1600
    LEFT_MASK_X_THRESHOLD = 1600
    MASK_Y_THRESHOLD = 700

    for _, row in df.iterrows():
        dicom_path = row["anon_dicom_path_local"]
        laterality = row["laterality"]  # Expected values: "Left" or "Right"

        if not os.path.exists(dicom_path):
            print(f"Skipping {dicom_path}, file not found.")
            continue

        dicom_filename = os.path.basename(dicom_path)
        dicom_img = pydicom.dcmread(dicom_path, force=True)
        img_array = dicom_img.pixel_array

        if dicom_img.PhotometricInterpretation == "MONOCHROME1":
            img_array = np.amax(img_array) - img_array

        img = apply_windowing(img_array, dicom_img)
        img = (img.astype(float) - img.min()) / (img.max() - img.min())
        img_gray = (img * 255).astype(np.uint8)

        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for i in range(1, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            if laterality == "Right" and (x <= RIGHT_MASK_X_THRESHOLD or (x + w <= RIGHT_MASK_X_THRESHOLD and y <= MASK_Y_THRESHOLD)):
                img_gray[y:y + h, x:x + w] = 0

            elif laterality == "Left" and (x >= img_gray.shape[1] - LEFT_MASK_X_THRESHOLD or (x + w >= img_gray.shape[1] - LEFT_MASK_X_THRESHOLD and y <= MASK_Y_THRESHOLD)):
                img_gray[y:y + h, x:x + w] = 0

        resized_image = resize_with_alignment(img_gray, 512, 1024, laterality)
        final_image_pil = Image.fromarray(resized_image.astype(np.uint8))

        filename = os.path.splitext(dicom_filename)[0] + ".png"
        png_path = os.path.join(output_path, filename)
        final_image_pil.save(png_path)

        print(f"Processed {dicom_filename} and saved as {png_path}")