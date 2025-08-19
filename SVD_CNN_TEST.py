# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# import os
# import time  # ⏱ For processing time measurement
# from torchvision.models import vgg19

# # ✅ Load trained model
# model_path = r"D:\codes\SVD_CNN\trained_vgg19.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class CustomVGG19(nn.Module):
#     def __init__(self):
#         super(CustomVGG19, self).__init__()
#         self.vgg19 = vgg19(pretrained=False)
#         self.vgg19.classifier = nn.Identity()  # Remove classifier, keep feature extraction layers

#     def forward(self, x):
#         return self.vgg19(x)

# # Load trained model
# model = CustomVGG19().to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()

# # ✅ Ensure grayscale images are converted to RGB
# def preprocess_image(image):
#     if len(image.shape) == 2 or image.shape[0] == 1:  # Grayscale check
#         image = np.stack([image] * 3, axis=-1)  # Convert 1-channel to 3-channel

#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     return transform(image).unsqueeze(0).to(device)

# # ✅ Extract Features Using Trained VGG19
# def extract_features(image):
#     image = preprocess_image(image)
#     with torch.no_grad():
#         features = model(image)
#     return features.cpu().numpy()

# # ✅ SVD Decomposition
# def decompose_svd(image):
#     U, S, Vt = np.linalg.svd(image, full_matrices=False)
#     L = np.dot(U, np.dot(np.diag(S), Vt))  # Low-frequency
#     H = image - L  # High-frequency
#     return L, H

# # ✅ Adaptive Fusion Weights
# def calculate_weights(L1, L2):
#     energy_L1, energy_L2 = np.sum(L1**2), np.sum(L2**2)
#     alpha = energy_L1 / (energy_L1 + energy_L2)
#     beta = energy_L2 / (energy_L1 + energy_L2)
#     return alpha, beta

# # ✅ Fusion Function with YUV Conversion
# def fuse_images(mri_image, pet_image):
#     # Convert PET image to YUV
#     pet_yuv = cv2.cvtColor(pet_image, cv2.COLOR_BGR2YUV)
#     Y_pet, U_pet, V_pet = cv2.split(pet_yuv)

#     # Convert MRI to grayscale
#     Y_mri = cv2.cvtColor(mri_image, cv2.COLOR_BGR2GRAY)

#     # Apply SVD to both Y channels
#     L_mri, H_mri = decompose_svd(Y_mri)
#     L_pet, H_pet = decompose_svd(Y_pet)

#     # Calculate fusion weights
#     alpha, beta = calculate_weights(L_mri, L_pet)

#     # Fuse the Y component
#     L_fused = alpha * L_mri + beta * L_pet
#     H_fused = (extract_features(H_mri).mean() + extract_features(H_pet).mean()) / 2 * H_pet
#     Y_fused = np.clip(L_fused + H_fused, 0, 255).astype(np.uint8)

#     # Merge back with U and V channels
#     fused_yuv = cv2.merge((Y_fused, U_pet, V_pet))

#     # Convert back to RGB
#     fused_rgb = cv2.cvtColor(fused_yuv, cv2.COLOR_YUV2BGR)
#     return fused_rgb

# # ✅ Process and Fuse Images
# input_folders = {
#     'T1': r"D:\codes\SVD_CNN\original_denoise\MR_T1",
#     'T2': r"D:\codes\SVD_CNN\original_denoise\MR_T2",
#     'PET': r"D:\codes\SVD_CNN\original_denoise\PET"
# }
# output_folders = {
#     'T1': r"D:\codes\SVD_CNN\fused_images_trained\T1_PET",
#     'T2': r"D:\codes\SVD_CNN\fused_images_trained\T2_PET"
# }

# # Ensure output directories exist
# os.makedirs(output_folders['T1'], exist_ok=True)
# os.makedirs(output_folders['T2'], exist_ok=True)
# # MR_T1 + PET fusion starts
# start_time_t1 = time.time()

# # ✅ Process T1-PET Fusion
# for image_name in os.listdir(input_folders['T1']):
#     mri_path = os.path.join(input_folders['T1'], image_name)
#     pet_path = os.path.join(input_folders['PET'], image_name)

#     img_mri = cv2.imread(mri_path)
#     img_pet = cv2.imread(pet_path)

#     if img_mri is None or img_pet is None:
#         print(f"Skipping {image_name}, missing file.")
#         continue

#     fused_img = fuse_images(img_mri, img_pet)
#     cv2.imwrite(os.path.join(output_folders['T1'], f"fused_{image_name}"), fused_img)
#     print(f"✅ Fused T1-PET image saved: {image_name}")
# # MR_T1 + PET fusion ends
# end_time_t1 = time.time()
# print(f"\n✅ MR_T1 + PET fusion completed in {end_time_t1 - start_time_t1:.2f} seconds.\n")

# # MR_T2 + PET fusion starts
# start_time_t2 = time.time()

# # ✅ Process T2-PET Fusion
# for image_name in os.listdir(input_folders['T2']):
#     mri_path = os.path.join(input_folders['T2'], image_name)
#     pet_path = os.path.join(input_folders['PET'], image_name)

#     img_mri = cv2.imread(mri_path)
#     img_pet = cv2.imread(pet_path)

#     if img_mri is None or img_pet is None:
#         print(f"Skipping {image_name}, missing file.")
#         continue

#     fused_img = fuse_images(img_mri, img_pet)
#     cv2.imwrite(os.path.join(output_folders['T2'], f"fused_{image_name}"), fused_img)
#     print(f"✅ Fused T2-PET image saved: {image_name}")
# # MR_T2 + PET fusion ends
# end_time_t2 = time.time()
# print(f"\n✅ MR_T2 + PET fusion completed in {end_time_t2 - start_time_t2:.2f} seconds.\n")

# print("✅ All fusion processing completed successfully!")
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import time  # ⏱ For processing time measurement
from torchvision.models import vgg19

# ✅ Load trained model
model_path = r"D:\codes\SVD_CNN\trained_vgg19_bayesopt.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomVGG19(nn.Module):
    def __init__(self):
        super(CustomVGG19, self).__init__()
        self.vgg19 = vgg19(pretrained=False)
        self.vgg19.classifier = nn.Identity()  # Remove classifier, keep feature extraction layers

    def forward(self, x):
        return self.vgg19(x)

# Load trained model
model = CustomVGG19().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ Ensure grayscale images are converted to RGB
def preprocess_image(image):
    if len(image.shape) == 2 or image.shape[0] == 1:  # Grayscale check
        image = np.stack([image] * 3, axis=-1)  # Convert 1-channel to 3-channel

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# ✅ Extract Features Using Trained VGG19
def extract_features(image):
    image = preprocess_image(image)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()

# ✅ SVD Decomposition
def decompose_svd(image):
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    L = np.dot(U, np.dot(np.diag(S), Vt))  # Low-frequency
    H = image - L  # High-frequency
    return L, H

# ✅ Adaptive Fusion Weights
def calculate_weights(L1, L2):
    energy_L1, energy_L2 = np.sum(L1**2), np.sum(L2**2)
    alpha = energy_L1 / (energy_L1 + energy_L2)
    beta = energy_L2 / (energy_L1 + energy_L2)
    return alpha, beta

# ✅ Fusion Function with YUV Conversion
def fuse_images(mri_image, pet_image):
    # Convert PET image to YUV
    pet_yuv = cv2.cvtColor(pet_image, cv2.COLOR_BGR2YUV)
    Y_pet, U_pet, V_pet = cv2.split(pet_yuv)

    # Convert MRI to grayscale
    Y_mri = cv2.cvtColor(mri_image, cv2.COLOR_BGR2GRAY)

    # Apply SVD to both Y channels
    L_mri, H_mri = decompose_svd(Y_mri)
    L_pet, H_pet = decompose_svd(Y_pet)

    # Calculate fusion weights
    alpha, beta = calculate_weights(L_mri, L_pet)

    # Fuse the Y component
    L_fused = alpha * L_mri + beta * L_pet
    H_fused = (extract_features(H_mri).mean() + extract_features(H_pet).mean()) / 2 * H_pet
    Y_fused = np.clip(L_fused + H_fused, 0, 255).astype(np.uint8)

    # Merge back with U and V channels
    fused_yuv = cv2.merge((Y_fused, U_pet, V_pet))

    # Convert back to RGB
    fused_rgb = cv2.cvtColor(fused_yuv, cv2.COLOR_YUV2BGR)
    return fused_rgb

# ✅ Process and Fuse Images
input_folders = {
    'T1': r"D:\codes\SVD_CNN\original_denoise\MR_T1",
    'T2': r"D:\codes\SVD_CNN\original_denoise\MR_T2",
    'PET': r"D:\codes\SVD_CNN\original_denoise\PET"
}
output_folders = {
    'T1': r"D:\codes\SVD_CNN\fused_images_trained\train-Bayesian\T1_PET",
    'T2': r"D:\codes\SVD_CNN\fused_images_trained\train-Bayesian\T2_PET"
}

# Ensure output directories exist
os.makedirs(output_folders['T1'], exist_ok=True)
os.makedirs(output_folders['T2'], exist_ok=True)

# ✅ Ask user for number of image pairs
try:
    num_images = int(input("Enter the number of image pairs to fuse (max 94): "))
except ValueError:
    print("Invalid input. Defaulting to 116.")
    num_images = 94

# MR_T1 + PET fusion starts
start_time_t1 = time.time()

# ✅ Process T1-PET Fusion
for idx, image_name in enumerate(sorted(os.listdir(input_folders['T1']))):
    if idx >= num_images:
        break

    mri_path = os.path.join(input_folders['T1'], image_name)
    pet_path = os.path.join(input_folders['PET'], image_name)

    img_mri = cv2.imread(mri_path)
    img_pet = cv2.imread(pet_path)

    if img_mri is None or img_pet is None:
        print(f"Skipping {image_name}, missing file.")
        continue

    fused_img = fuse_images(img_mri, img_pet)
    cv2.imwrite(os.path.join(output_folders['T1'], f"fused_{image_name}"), fused_img)
    print(f"✅ Fused T1-PET image saved: {image_name}")
# MR_T1 + PET fusion ends
end_time_t1 = time.time()
print(f"\n✅ MR_T1 + PET fusion completed in {end_time_t1 - start_time_t1:.2f} seconds.\n")

# MR_T2 + PET fusion starts
start_time_t2 = time.time()

# ✅ Process T2-PET Fusion
for idx, image_name in enumerate(sorted(os.listdir(input_folders['T2']))):
    if idx >= num_images:
        break

    mri_path = os.path.join(input_folders['T2'], image_name)
    pet_path = os.path.join(input_folders['PET'], image_name)

    img_mri = cv2.imread(mri_path)
    img_pet = cv2.imread(pet_path)

    if img_mri is None or img_pet is None:
        print(f"Skipping {image_name}, missing file.")
        continue

    fused_img = fuse_images(img_mri, img_pet)
    cv2.imwrite(os.path.join(output_folders['T2'], f"fused_{image_name}"), fused_img)
    print(f"✅ Fused T2-PET image saved: {image_name}")
# MR_T2 + PET fusion ends
end_time_t2 = time.time()
print(f"\n✅ MR_T2 + PET fusion completed in {end_time_t2 - start_time_t2:.2f} seconds.\n")

print("✅ All fusion processing completed successfully!")
