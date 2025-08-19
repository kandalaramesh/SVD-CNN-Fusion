import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torchvision.io import read_image
from google.colab import drive

# 1️⃣ **Mount Google Drive**
# drive.mount('/content/drive')

# 2️⃣ **Define Dataset Path**
dataset_path = "/content/drive/MyDrive/ASFE-Fusion-main"

# 3️⃣ **Check If Dataset Exists**
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# 4️⃣ **Define Custom Dataset Class**
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = read_image(img_path).float() / 255.0  # Normalize

        # Convert grayscale to RGB
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)
        return image

# 5️⃣ **Define Stronger Augmentations**
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 6️⃣ **Load MRI & PET Datasets**
mri_dataset = CustomImageDataset(image_dir=os.path.join(dataset_path, "MRI"), transform=train_transform)
pet_dataset = CustomImageDataset(image_dir=os.path.join(dataset_path, "PET"), transform=train_transform)

train_dataset = torch.utils.data.ConcatDataset([mri_dataset, pet_dataset])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print(f"✅ Dataset loaded successfully! Total images: {len(train_dataset)}")

# 7️⃣ **Define Custom VGG19 Model**
class CustomVGG19(nn.Module):
    def __init__(self):
        super(CustomVGG19, self).__init__()
        self.vgg19 = models.vgg19(weights="IMAGENET1K_V1")  # Pretrained weights
        self.vgg19.classifier = nn.Identity()

    def forward(self, x):
        return self.vgg19(x)

# 8️⃣ **Set Device (Use GPU if available)**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model
model = CustomVGG19().to(device)

# 🔥 **Improved Feature Loss**
class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, feature1, feature2):
        return self.loss_fn(feature1, feature2)

criterion = FeatureLoss()

# 🔥 **Reduce Learning Rate**
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Reduced LR

# 🔥 **Use Learning Rate Scheduler**
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

# 🔥 **Clip Gradients to Prevent Vanishing Updates**
def clip_gradient(optimizer, clip_value=1.0):
    for group in optimizer.param_groups:
        for param in group['params']:
            torch.nn.utils.clip_grad_norm_(param, clip_value)

# 9️⃣ **Train the Model**
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs in train_loader:
        inputs = inputs.to(device)

        optimizer.zero_grad()
        
        # 🔥 Use Stronger Data Augmentation (Rotation, Jitter)
        transformed_inputs = torch.rot90(inputs, k=1, dims=[2, 3])  # Rotate 90 degrees

        # Extract VGG features
        original_features = model(inputs)
        transformed_features = model(transformed_inputs)

        # Compute loss
        loss = criterion(original_features, transformed_features)

        loss.backward()
        
        # Apply Gradient Clipping
        clip_gradient(optimizer)

        optimizer.step()
        running_loss += loss.item()

    # Update learning rate
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 🔟 **Save Trained Model**
model_path = "/content/drive/MyDrive/ASFE-Fusion-main/model/trained_vgg19.pth"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved at {model_path}")
