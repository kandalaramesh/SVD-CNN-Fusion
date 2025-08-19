import os
import cv2
import numpy as np
from skimage.util import random_noise

def add_photon_noise(image):
    """Add Photon noise (Poisson noise) to an image."""
    if image.max() > 1:
        image = image / 255.0  # Normalize to [0, 1] for Poisson noise
    noisy_image = random_noise(image, mode='poisson')
    noisy_image = (255 * noisy_image).astype('uint8')  # Scale back to [0, 255]
    return noisy_image

def process_images(input_folder, output_folder, noise_function):
    """Apply noise to images and save them to the corresponding output folder."""
    os.makedirs(output_folder, exist_ok=True)

    for image_name in sorted(os.listdir(input_folder)):
        if image_name.endswith('.png'):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Could not read {image_path}")
                continue

            noisy_image = noise_function(image)

            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, noisy_image)
            print(f"Processed {output_path}")

# Input and Output folders
input_folder = "D:/codes/Backup_dataset/brain_atlas/PET"
output_folder = "D:/codes/noises_datasets/photon/PET"

# Apply Photon noise to PET dataset
process_images(input_folder, output_folder, add_photon_noise)
