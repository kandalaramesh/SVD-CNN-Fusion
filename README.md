# SVD-CNN Fusion: A Hybrid Medical Image Integration Model

This repository contains the official implementation of the **SVD-CNN Fusion** model proposed in our paper:

**"SVD-CNN Fusion: A Hybrid Medical Image Integration Model for Color-Retentive Reconstruction"**

📌 **Purpose:**  
To perform robust and color-preserving fusion of MRI and PET brain images using a hybrid approach combining:
- **SVD (Singular Value Decomposition)** for low-frequency structural fusion.
- **VGG19-based CNN** for high-frequency feature-aware integration.
- **YUV color space** for preserving PET chrominance and MRI structure.
- **Denoising module** for Gaussian and Poisson noise correction.

---

## 🧠 Key Features

- Dual-modality support: T1-PET and T2-PET fusion
- Preprocessing with controlled noise injection (Gaussian for MRI, Poisson for PET)
- Hybrid denoising using bilateral, NLM, and guided filtering
- Energy-based low-frequency fusion + deep feature fusion for high-frequency components
- Fast inference (<0.6s/image on CPU)
- Evaluated on Harvard Brain Image Dataset

---

## 📁 Folder Structure

```
SVD_CNN_Fusion/
│
├── noise_images.py              # Add Gaussian noise to MRI images
├── poisson_noise.py            # Add Poisson noise to PET images
├── denoise_MRI_PET.py          # Hybrid denoising module (MRI: bilateral, PET: NLM + guided)
├── SVD_CNN_TRAIN.py            # Train VGG19 on modality-specific augmentations
├── SVD_CNN_TEST.py             # Final fusion pipeline for T1-PET and T2-PET
├── trained_vgg19.pth           # [Optional] Trained VGG19 model (to be added manually or linked)
├── requirements.txt
└── README.md
```

---

## 🗂 Dataset Preparation

Due to license restrictions, we cannot distribute full datasets. However:

- ✅ Public dataset source:
  - MRI & PET Images: [Harvard Brain Atlas](https://www.med.harvard.edu/aanlib/)
    [Harvard Fusion Dataset](https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets)

**Input Image Folders**
```
D:/codes/brain_atlas/
├── MR_T1/
├── MR_T2/
└── PET/
```

**Output after Denoising**
```
D:/codes/SVD_CNN/original_denoise/
├── MR_T1/
├── MR_T2/
└── PET/
```

**Final Fusion Output**
```
D:/codes/SVD_CNN/fused_images_trained/
├── T1_PET/
└── T2_PET/
```

---

## 🚀 Running the Code

### 1. Add Noise (Simulate Clinical Degradations)
```bash
python noise_images.py         # Adds Gaussian noise to MR_T1 and MR_T2
python poisson_noise.py        # Adds Poisson noise to PET images
```

### 2. Apply Hybrid Denoising
```bash
python denoise_MRI_PET.py      # Denoises MRI and PET images using appropriate filters
```

### 3. Train the VGG19 Feature Extractor
```bash
python SVD_CNN_TRAIN.py
```

> 🔒 If GPU unavailable, training will automatically fall back to CPU.

### 4. Fuse MRI + PET Images
```bash
python SVD_CNN_TEST.py
```

---

## 📊 Evaluation Metrics Used

- PSNR, MSE, SSIM, CC (with ground truth)
- Entropy (EN), Std. Dev. (SD), Avg. Gradient (AG)
- LPIPS (Learned Perceptual Image Patch Similarity)

---

## 🔗 Citation

If you use this code, please cite:

```bibtex
@article{ramesh2025svdcnn,
  title={SVD-CNN Fusion: A Hybrid Medical Image Integration Model for Color-Retentive Reconstruction},
  author={Kandala, SSVV Ramesh and Selva Kumar, S},
  journal={PLOS ONE},
  year={2025}
}
```

---

## ⚖ License

This code is released for **research purposes only**. Please contact the authors if you need extended access.

---

## 📬 Contact

For queries, reach out to:
- SSVV Ramesh Kandala, VIT-AP University
- selvakumar.s@vitap.ac.in
