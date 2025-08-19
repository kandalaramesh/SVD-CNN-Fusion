import os
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from cv2.ximgproc import guidedFilter  # Ensure OpenCV contrib module is installed

# ✅ Bilateral Filter for MRI (preserves edges while reducing Gaussian noise)
def apply_bilateral_denoise(image):
    """Denoise MRI image using Bilateral Filter."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# ✅ Improved Non-Local Means + Guided Filtering for PET (Poisson noise)
def apply_poisson_denoise(image):
    """Denoise PET image using Non-Local Means and Guided Filtering."""
    sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))  
    
    # Reduce h value to prevent excessive smoothing
    denoised = denoise_nl_means(
        image, 
        h=0.5 * sigma_est,  # Reduced from 0.6 to 0.5 to retain details
        fast_mode=True, 
        patch_size=3,  # Keeping smaller patches for fine preservation
        patch_distance=3,  # Reduced further for localized denoising
        channel_axis=-1
    )
    
    denoised = (255 * denoised).astype('uint8')  # Scale back to [0, 255]
    
    # **Use Guided Filter instead of Median Blur for better edge preservation**
    denoised = guidedFilter(denoised, denoised, radius=4, eps=0.01)

    return denoised

# 🔄 Process and Save Images
def process_and_save_images(input_folder, output_folder, noise_type):
    """Process images, apply denoising based on noise type, and save results."""
    os.makedirs(output_folder, exist_ok=True)

    for image_name in sorted(os.listdir(input_folder)):
        if image_name.endswith('.png'):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Could not read {image_path}")
                continue

            # Apply the correct denoising filter based on noise type
            if noise_type == "gaussian":
                denoised_image = apply_bilateral_denoise(image)
            elif noise_type == "poisson":
                denoised_image = apply_poisson_denoise(image)
            else:
                print(f"Unknown noise type for {image_name}")
                continue

            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, denoised_image)
            print(f"Denoised and saved: {output_path}")

# 📂 Define Input and Output Folders
folders = {
    "MR_T1": {
        "input": r"D:\codes\brain_atlas\MR_T1",
        "output": r"D:\codes\SVD_CNN\original_denoise\MR_T1",
        "noise_type": r"gaussian"
    },
    "MR_T2": {
        "input": r"D:\codes\brain_atlas\MR_T2",
        "output": r"D:\codes\SVD_CNN\original_denoise\MR_T2",
        "noise_type": r"gaussian"
    },
    "PET": {
        "input": r"D:\codes\brain_atlas\PET",
        "output": r"D:\codes\SVD_CNN\original_denoise\PET",
        "noise_type": r"poisson"
    }
}

# 🔄 Run the Denoising Process
for category, paths in folders.items():
    process_and_save_images(paths["input"], paths["output"], paths["noise_type"])
