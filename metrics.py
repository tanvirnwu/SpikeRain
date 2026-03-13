import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def calculate_metrics_from_folders(gt_folder, modified_folder, resize_dim=(512, 512)):
    # Check if folders exist
    if not os.path.exists(gt_folder):
        print(f"Error: Ground truth folder '{gt_folder}' does not exist.")
        return
    if not os.path.exists(modified_folder):
        print(f"Error: Modified folder '{modified_folder}' does not exist.")
        return

    # Initialize accumulators for metrics
    total_psnr, total_ssim, total_mse = 0.0, 0.0, 0.0
    count = 0

    # Process all images in folders
    for gt_file in os.listdir(gt_folder):
        gt_path = os.path.join(gt_folder, gt_file)
        modified_path = os.path.join(modified_folder, gt_file)

        # Check if both ground truth and modified images exist
        if os.path.isfile(gt_path) and os.path.isfile(modified_path):
            # Load images
            gt_image = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            modified_image = cv2.imread(modified_path, cv2.IMREAD_COLOR)
            
            # Resize images to the specified dimensions
            gt_image_resized = cv2.resize(gt_image, resize_dim)
            modified_image_resized = cv2.resize(modified_image, resize_dim)
            
            # Convert images to grayscale for SSIM calculation
            gt_image_gray = cv2.cvtColor(gt_image_resized, cv2.COLOR_BGR2GRAY)
            modified_image_gray = cv2.cvtColor(modified_image_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            psnr_value = peak_signal_noise_ratio(gt_image_resized, modified_image_resized)
            ssim_value, _ = structural_similarity(gt_image_gray, modified_image_gray, full=True)
            mse_value = mean_squared_error(gt_image_resized, modified_image_resized)
            
            # Accumulate metrics
            total_psnr += psnr_value
            total_ssim += ssim_value
            total_mse += mse_value
            count += 1

    # Compute averages
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_mse = total_mse / count
    else:
        avg_psnr, avg_ssim, avg_mse = 0.0, 0.0, 0.0

    # Print results in the requested format
    print(f"{avg_psnr:.2f} & {avg_ssim:.4f} & {avg_mse:.2f} \\\\")


# Example usage
dehazed_image = "/home/tanvir/projects/img_restoration/sota/ESDNet/output/"
ground_truth = "/home/tanvir/datasets/img_restoration/detraining/rain1400/test/target"
calculate_metrics_from_folders(ground_truth, dehazed_image)