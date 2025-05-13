import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import SpineDataset
from model import UNet
from loss import dice_loss

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "D:/Spine_Segmentation/DATASET/total_png_dataset"
json_dir = "D:/Spine_Segmentation/DATASET/total_json"
batch_size = 4
learning_rate = 1e-4
num_epochs = 50
save_path = "D:/Spine_Segmentation/weights/unet_dice.pth"

# 데이터 로딩
dataset = SpineDataset(root_dir=root_dir, json_dir=json_dir, transform=None)
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델, 최적화 설정
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = dice_loss(outputs, masks)
            val_loss += loss.item() * images.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")
