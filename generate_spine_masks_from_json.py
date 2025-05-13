import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pydicom as dcm
import json
import os
import glob
import skimage
import cv2
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from easydict import EasyDict as edict
from PIL import Image, ImageDraw

data_dir = './YRL_Spine_data/'
data_list = glob.glob('./YRL_Spine_data/*.json')
current_dir = os.getcwd()

data_list[0].replace('.json','.dcm')

def read_dcm(img_path):
#     dcm_data = read_dcm(img_path)
    dcm_data = dcm.read_file(img_path, force=True)
    # print(dcm_data.file_meta)
    print(img_path)
    try:
        if dcm_data.PhotometricInterpretation != "MONOCHROME2":
            # print(img_path, dcm_data.PhotometricInterpretation)
            image = np.invert(dcm_data.pixel_array.squeeze())
        else:
            # print(img_path, dcm_data.PhotometricInterpretation)
            image = dcm_data.pixel_array.squeeze()
    except: # ValueError:
        dcm_data.file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian
        try:
            if dcm_data.PhotometricInterpretation != "MONOCHROME2":
                img = np.invert(dcm_data.pixel_array.squeeze())
            else:
                img = dcm_data.pixel_array.squeeze()
        except:
            img = dcm_data.pixel_array.squeeze()
        print(img_path)
        # ValueError: The length of the pixel data in the dataset (~~~ bytes) doesn't match the expected length (~~~ bytes).
        # The dataset may be corrupted or there may be an issue with the pixel data handler.
        # print(f"ERROR: {img_path}")
        # return

    cut = image
    # Min-Max normalization
    image = (image - cut.min()) / (cut.max() - cut.min()) * 255
    # Additional preprocess
#     converted_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
#     converted_image = img_Contrast(converted_image)
#     converted_image = cv2.convertScaleAbs(converted_image, beta=25)
    return image # converted_image

dcm_image = read_dcm(data_list[3].replace('.json','.dcm'))
# dcm_data = dcm.read_file(data_list[2].replace('.json', '.dcm'))
# dcm_image = dcm_data.pixel_array()
np.savetxt('./data_images/test.txt', dcm_image)
# np.savetxt(current_dir + 'test.txt', dcm_image)

with open(data_list[3]) as json_reader:
    one_json = edict(json.load(json_reader))

points = one_json.study[0].series[0].lesion[0].points.split('|')
ppoints=[]
for point in points:
    x = round(float(point.split(',')[0]))
    y = round(float(point.split(',')[1]))
    ppoints.append([x,y])
ppoints  = np.asarray(ppoints)
spce = 0.314

all_points = []
for i in range(len(one_json.study[0].series[0].lesion)):
    one_lesion = one_json.study[0].series[0].lesion[i]
    points = one_lesion['points'].split('|')
    ppoints=[]
    for point in points:
        x = round(float(point.split(',')[0])/spce)
        y = round(float(point.split(',')[1])/spce)
        ppoints.append([x,y])
    all_points.append(ppoints)

all_points

all_points2 = []
for i in range(len(one_json.study[0].series[0].lesion)):
    one_lesion = one_json.study[0].series[0].lesion[i]
    points = one_lesion['points'].split('|')
    ppoints2=[]
    for point in points:
        x = round(float(point.split(',')[0])/spce)
        y = round(float(point.split(',')[1])/spce)
        ppoints2.append([[x,y]])
    all_points2.append(ppoints2)

one_dcm = dcm.read_file(data_list[3].replace('.json', '.dcm'))
one_img = one_dcm.pixel_array.squeeze()
spce = one_dcm.PixelSpacing[0]
one_img.shape

def create_mask(image_size, polygon_points):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, outline=1, fill=1)
    del draw
    return np.array(mask)

# Create a figure and axis
fig, ax = plt.subplots()
# Display the mask image
ax.imshow(np.zeros([1116, 500]), cmap='gray')
# Add the polygon patch to the axis
for i in range(len(all_points)):
    polygon_patch = patches.Polygon(all_points[i]*7, closed=True, edgecolor='w', linewidth=1, facecolor='white')
    ax.add_patch(polygon_patch)
# Show the plot
plt.show()

# Create a figure and axis
fig, ax = plt.subplots()
# Display the mask image
ax.imshow(np.zeros([3000, 1772]), cmap='gray')
# Add the polygon patch to the axis
for i in range(len(edges)):
    polygon_patch = patches.Polygon(edges[i], closed=True, edgecolor='red', linewidth=0.5, facecolor='none')
    ax.add_patch(polygon_patch)
# Show the plot
plt.show()

def get_deg(adot):
  [tl, bl, tr, br]= adot
  rad1 = math.atan2(tl[1]-tr[1], tl[0]-tr[0])
  ang1 = (rad1*180)/math.pi
  rad2 = math.atan2(bl[1]-br[1], bl[0]-br[0])
  ang2 = (rad2*180)/math.pi
  ang = ang1 - ang2
  return int(to_pos(ang))

for adot in edges:
    print(get_deg(adot))

def draw_edges(img_dir, edges);
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Display the mask image
    h, w =img_dir.shape()
    ax.imshow(np.zeros([h, w]), cmap='gray')
    # Add the polygon patch to the axis
    for i in range(len(edges)):
        edge = edges[i]
        degree = get_deg(edge)
        for j in range(len(edge)):
            if i == 0:
              f_color = 'red'
            elif i == 1:
              f_color = 'orange'
            elif i == 2:
              f_color = 'yellow'
            elif i == 3:
              f_color = 'cyan'
            elif i == 4:
              f_color = 'green'
            elif i == 5:
              f_color = 'blue'
            elif i == 6:
              f_color = 'purple'
            elif i == 7:
              f_color = 'brown'
            else:
              f_color = 'white'
            plt.scatter(hull[j][0], w-10, facecolor=f_color)
