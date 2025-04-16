# VisionAI Size Measurement API

This repository contains a RESTful API built with Django and integrated with cutting-edge AI models for size measurement. The system processes images of a person (front and side views) and calculates body measurements, helping to determine accurate clothing sizes.

## Features
- Uses **YOLOv8m-pose** for pose estimation.
- Calculates measurements such as height, chest, waist, sleeve, and neck.
- Dynamic size categorization based on provided measurements.
- Handles both male and female size calculations.

## Requirements
- Python 3.8+
- Django 4.x
- Required Python packages (see `requirements.txt`):
  - `opencv-python`
  - `torch`
  - `insightface`
  - `ultralytics`
  - `Pillow`
  - `djangorestframework`

## Installation
1. Clone this repository:
   ```bash
   https://github.com/Mehedi-Bin-Hafiz/Body-Size-Measurements-for-Clothing-E-commerce.git
   cd Body-Size-Measurements-for-Clothing-E-commerce
