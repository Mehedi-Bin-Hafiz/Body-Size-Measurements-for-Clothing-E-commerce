# Size Measurements Vision AI: Core Innovation & Math Formulas

This project is built on Python's Django framework to create an AI-driven REST API for precise body size estimation. The innovation lies in combining image processing, pose estimation, and mathematical formulas to extract real-world measurements from images. The project is designed for estimating T-shirt sizes based on uploaded front and side images.

## Core Innovation

### Pose Detection with YOLOv8
- Pretrained `yolov8m-pose.pt` model for keypoint detection.
- Extracts pixel coordinates of body landmarks (e.g., shoulders, elbows, and waist).

### Face Detection
- Utilizes OpenCV's Haar Cascade Classifier to determine face width in pixels.

### Distance Estimation
- Calculates the subject's real-world distance from the camera using focal length and face width.

### Pixel-to-Centimeter Conversion
- Converts pixel measurements to real-world dimensions based on distance.

### Mathematical Formulas for Measurements
- Custom formulas for calculating chest, waist, sleeve length, and height.

### T-Shirt Size Recommendation
- Maps measurements to size categories (S, M, L, etc.).

## Mathematical Formulas Used

### 1. Distance Estimation
The real-world distance \( D \) is calculated using the formula:

\[
D = \frac{W \cdot F}{P}
\]

Where:
- \( W \): Real width of the human face (constant, e.g., 14 cm).
- \( F \): Camera's focal length (calculated using known face images).
- \( P \): Face width in pixels detected by Haar Cascade.

### 2. Pixel-to-Centimeter Conversion
Conversion factor \( K \):

\[
K = \frac{W}{P}
\]

### 3. Height Calculation
From the shoulder and hip keypoints detected by YOLOv8:

\[
H = |y_{\text{shoulder}} - y_{\text{hip}}|
\]

Where:
- \( y_{\text{shoulder}} \) and \( y_{\text{hip}} \): Vertical pixel coordinates of the respective keypoints.

### 4. Chest and Waist Measurements
Using keypoints for the shoulders and waist:

\[
\text{Chest} = |x_{\text{shoulder1}} - x_{\text{shoulder2}}|
\]

\[
\text{Waist} = |x_{\text{waist1}} - x_{\text{waist2}}|
\]

### 5. T-Shirt Size Classification
Predefined thresholds for height, chest, and waist measurements are used to determine sizes. For example:
- **Small (S)**: Chest < 90 cm, Height < 160 cm
- **Medium (M)**: \( 90 \, \text{cm} \leq \text{Chest} < 100 \, \text{cm}, \, 160 \, \text{cm} \leq \text{Height} < 170 \, \text{cm} \)
- **Large (L)**: Chest ≥ 100 cm, Height ≥ 170 cm
