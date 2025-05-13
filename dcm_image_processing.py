import numpy as np
import pydicom as dcm
import cv2

def read_dcm(img_path, target_size=(256, 256)):
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
    image = (image - image.min()) / (image.max() - image.min()) * 255
    
    return image.astype(np.uint8)  # 반환값은 uint8 형식의 이미지로 변환
