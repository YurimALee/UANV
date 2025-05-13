import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dcm
import cv2
import os

# DICOM 파일 읽기 함수
def read_dcm(img_path, target_size=(128, 128)):
    # DICOM 파일 읽기
    dcm_data = dcm.read_file(img_path, force=True)
    
    try:
        # 이미지의 픽셀 값 추출
        image = dcm_data.pixel_array.squeeze()
        
        # 이미지가 MONOCHROME2 형식이 아닌 경우 픽셀 값을 반전시킴
        if dcm_data.PhotometricInterpretation != "MONOCHROME2":
            image = np.invert(image)
    except Exception as e:
        print(f"Error reading DICOM file {img_path}: {str(e)}")
        return None
    
    # 이미지 크기 조정
    image = cv2.resize(image, target_size)
    
    # Min-Max 정규화 수행
    cut = image
    image = (image - cut.min()) / (cut.max() - cut.min()) * 255
    
    # Torch Tensor로 변환하여 반환
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # 차원 추가하여 (1, H, W)로 변환
    return image_tensor

# UNet++ 모델 정의
class UNetPlusPlus(nn.Module):
    def __init__(self):
        super(UNetPlusPlus, self).__init__()
        # Encoder 부분
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder 부분
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 최종 출력은 확률 값이므로 Sigmoid 사용
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 최종 출력은 확률 값이므로 Sigmoid 사용
        )
    
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        
        x1 = self.decoder1(x2)
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        
        x2 = torch.cat([x1, x2], dim=1)
        x2 = self.decoder2(x2)
        
        return x2

# 데이터셋 정의 (CustomDataset 클래스를 사용하여 read_dcm 함수로 DICOM 파일을 불러옴)
class CustomDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_dcm(img_path)
        if img is None:
            return None
        
        return img

# 학습 파라미터 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
epochs = 10
learning_rate = 0.001

# 데이터 경로 설정 (적절히 수정 필요)
data_dir = "./YRL_Spine_data"
img_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dcm')]

# 데이터셋 인스턴스 생성
dataset = CustomDataset(img_paths)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 모델 초기화 및 손실 함수, 최적화기 설정
model = UNetPlusPlus().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습
model.train()
for epoch in range(epochs):
    for batch_idx, data in enumerate(dataloader):
        inputs = data.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# 학습 완료 후 모델 저장 (필요시)
# torch.save(model.state_dict(), 'unetplusplus_model.pth')
