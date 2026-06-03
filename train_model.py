import os
import torch
import torch.nn as nn
import torch.optim as optim
import tifffile as tiff
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# ======================
# Dataset Class
# ======================
class BuildingDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = [f for f in os.listdir(path) if "_image.tif" in f]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        label_name = img_name.replace("_image.tif", "_label.tif")

        img = tiff.imread(os.path.join(self.path, img_name))
        mask = tiff.imread(os.path.join(self.path, label_name))

        # normalize
        img = img[:, :, :3] / 255.0
        mask = (mask > 0).astype(np.float32)

        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# ======================
# U-Net Model
# ======================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.middle = self.conv_block(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = self.conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        m = self.middle(self.pool(e2))

        d1 = self.up1(m)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return torch.sigmoid(self.final(d2))

# ======================
# Training
# ======================
train_path = r"C:\Users\ramya\OneDrive\Desktop\building-project\train"

dataset = BuildingDataset(train_path)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# to store loss values for graph
loss_history = []

# ======================
# TRAIN LOOP
# ======================
for epoch in range(20):
    total_loss = 0

    for img, mask in loader:
        mask = mask.unsqueeze(1)

        pred = model(img)
        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss_history.append(total_loss)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "building_unet.pth")
    print("Model saved successfully!")

# ======================
# METRICS
# ======================
# flatten masks
y_true = mask.cpu().numpy().flatten()
y_pred = (pred > 0.5).cpu().numpy().astype(int).flatten()

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)

# ======================
# GRAPH 1: Training Loss vs Epoch
# ======================
epochs = list(range(1, len(loss_history) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.show()

# ======================
# GRAPH 2: Performance Metrics Bar Graph
# ======================
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metric_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

plt.figure(figsize=(8, 5))
plt.bar(metric_names, metric_values)
plt.xlabel("Metrics")
plt.ylabel("Percentage (%)")
plt.title("Performance Metrics of U-Net Model")
plt.ylim(0, 100)
plt.show()

# ======================
# TEST ON ONE IMAGE
# ======================
img, mask = dataset[0]
img_input = img.unsqueeze(0)

pred = model(img_input).detach().numpy()[0, 0]

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(mask.numpy(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pred > 0.5, cmap='gray')

plt.show()
