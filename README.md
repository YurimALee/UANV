# UANV
UNet-based Attention Network for Vertebral Compression Fracture Angle Measurement

## Overview
Scoliosis is a prevalent spinal condition characterized by the abnormal lateral curvature of the spine, which can result in deformities. In clinical settings, assessing the severity of scoliosis is crucial for diagnosis and treatment. One common method for curvature estimation is the Cobb angle, which measures the angle between two lines drawn perpendicular to the upper and lower endplates of the affected vertebrae. However, manual measurement of the Cobb angle is time-consuming and subject to interobserver and intraobserver variations.

This repository presents UANV (UNet-based Attention Network for Vertebral Compression Fracture Angle), a deep convolutional neural network (CNN)-based model for automating vertebra angle measurements using lateral spinal X-ray images. The model incorporates an attention mechanism to capture the detailed shape of each vertebra and records the edges of each vertebra to accurately calculate the vertebral angles.

## Key Features
Automated Vertebra Angle Measurement: The model predicts the vertebral angles directly from lateral spinal X-rays, reducing the need for manual measurements.

Attention Mechanism: The attention module enables the model to focus on critical regions, improving the accuracy of vertebrae angle estimation.

Deep Learning Approach: Utilizes a UNet-based architecture to process and analyze X-ray images for curvature estimation.

## Model Architecture
The model leverages a UNet-based architecture with an attention mechanism to focus on the important regions of the X-ray image. The network is trained to segment each vertebra and detect its boundaries, allowing for accurate calculation of the vertebral angle.

## Requirements
To run this model, you will need the following libraries:
torch
torchvision
pydicom
opencv-python
numpy
matplotlib

