# Required Libraries
import pandas as pd
import json
import numpy as np
import pydicom as dcm
import os
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from easydict import EasyDict as edict
from PIL import Image, ImageDraw
import glob

# Install necessary libraries
!pip install pydicom scikit-image scikit-learn torch opencv-python matplotlib

# Set Data Directory
data_dir = './YRL_Spine_data/'
current_dir = os.getcwd()

# List all JSON files in data directory
data_list = glob.glob('./YRL_Spine_data/*.json')
print(data_list[3])

# Function to read DICOM file and preprocess image
def read_dcm(img_path):
    """Read DICOM image and preprocess"""
    dcm_data = dcm.read_file(img_path, force=True)
    print(img_path)
    
    try:
        if dcm_data.PhotometricInterpretation != "MONOCHROME2":
            image = np.invert(dcm_data.pixel_array.squeeze())
        else:
            image = dcm_data.pixel_array.squeeze()
    except:
        dcm_data.file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian
        try:
            if dcm_data.PhotometricInterpretation != "MONOCHROME2":
                image = np.invert(dcm_data.pixel_array.squeeze())
            else:
                image = dcm_data.pixel_array.squeeze()
        except:
            image = dcm_data.pixel_array.squeeze()
        
        print(img_path)
    
    # Min-Max Normalization
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image

# Function to load JSON and extract lesion points
def load_json_and_points(file):
    """Load JSON and extract lesion points"""
    with open(file) as json_reader:
        json_data = edict(json.load(json_reader))
    
    lesion_points = []
    for lesion in json_data.study[0].series[0].lesion:
        points = lesion['points'].split('|')
        points_list = []
        for point in points:
            x, y = map(float, point.split(','))
            points_list.append([round(x), round(y)])
        lesion_points.append(np.asarray(points_list))
    
    return lesion_points

# Extract DICOM image size and spacing info
def get_image_and_spacing(img_file):
    """Get DICOM image and pixel spacing"""
    dcm_data = dcm.read_file(img_file)
    spacing = dcm_data.PixelSpacing[0]
    img_array = dcm_data.pixel_array.squeeze()
    return img_array, spacing

# Create mask from polygon points
def create_mask(image_size, polygon_points):
    """Create binary mask from polygon points"""
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, outline=1, fill=1)
    del draw
    return np.array(mask)

# Visualize image with polygon overlay
def visualize_image_with_mask(image, polygon_points):
    """Visualize the DICOM image with lesion points overlaid as polygons"""
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for points in polygon_points:
        polygon_patch = patches.Polygon(points, closed=True, edgecolor='r', linewidth=1, facecolor='none')
        ax.add_patch(polygon_patch)
    plt.show()

# Process all images and generate masks
def process_data(data_list):
    """Process all DICOM images and generate masks for each"""
    mask_data = []
    for file in data_list:
        img_file = file.replace('.json', '.dcm')
        img, spce = get_image_and_spacing(img_file)
        
        # Load lesion points from the corresponding JSON
        all_points = load_json_and_points(file)
        
        # Store image and lesion points
        one_image = {
            'image': img_file,
            'points': all_points
        }
        mask_data.append(one_image)
    
    return mask_data

# Save processed data as JSON
def save_processed_data(mask_data, output_file):
    """Save the processed data to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(mask_data, f, indent=4)

# Main processing logic
mask_data = process_data(data_list)
save_processed_data(mask_data, current_dir + '/Preprocessed_data4.json')

# Visualize example image with mask
example_data = mask_data[0]
image_name = example_data['image']
img_array, _ = get_image_and_spacing(image_name)
lesion_points = example_data['points']
visualize_image_with_mask(img_array, lesion_points)
