import os
import cv2
import numpy as np

def create_directories(base_path, categories):
    """Create directories for storing noisy images."""
    for category in categories:
        os.makedirs(os.path.join(base_path, category), exist_ok=True)

def add_gaussian_noise(image):
    """Add Gaussian noise to an image."""
    mean = 0
    stddev = 0.5
    gaussian_noise = np.random.normal(mean, stddev, image.shape).astype('float32')
    noisy_image = np.clip(image.astype('float32') + gaussian_noise, 0, 255).astype('uint8')
    return noisy_image

def process_images(input_folder, output_folder, noise_function):
    """Apply noise to images and save them to the corresponding output folder."""
    os.makedirs(output_folder, exist_ok=True)

    for image_name in sorted(os.listdir(input_folder)):
        if image_name.endswith('.png'):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read {image_path}")
                continue

            noisy_image = noise_function(image)

            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, noisy_image)
            print(f"Processed {output_path}")

# Input and Output folders
input_folders = {
    "MR_T1": "D:/codes/Backup_dataset/brain_atlas/MR_T1",
    "MR_T2": "D:/codes/Backup_dataset/brain_atlas/MR_T2"
}

output_folders = {
    "MR_T1": "D:/codes/noises_datasets/gaussian/MR_T1",
    "MR_T2": "D:/codes/noises_datasets/gaussian/MR_T2"
}

# Apply Gaussian noise
for category, input_folder in input_folders.items():
    output_folder = output_folders[category]
    process_images(input_folder, output_folder, add_gaussian_noise)
