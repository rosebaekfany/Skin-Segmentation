# Enhanced Skin Segmentation and Color Tone Classification

## Overview
This project enhances MediaPipe’s multi-class selfie segmentation model by integrating additional HSV and YCbCr thresholding techniques to improve human skin segmentation and color tone classification. Our approach combines preprocessing steps such as Gaussian blur and histogram equalization with advanced image processing techniques to enhance segmentation accuracy and classify skin tones using predefined monk color values.

## Project Structure
The repository includes the following Python scripts and a detailed documentation of the methodology in `skin_segmentation.pdf`:

- `mediapipe_color_space.py` - Applies color space transformations and MediaPipe segmentation.
- `mediapipe_color_space_combine.py` - Combines color space transformations with MediaPipe outputs.
- `mediapipe_color_space_ycbcr.py` - Specializes in YCbCr color space for segmentation.
- `mediapipe_color_space_ycbcr_combin.py` - Applies combined thresholding and MediaPipe models in the YCbCr color space.
- `mediapipe_gussian_hist.py` - Utilizes Gaussian blur and histogram equalization preprocessing.
- `Mediapipe_only.py` - Basic MediaPipe segmentation without additional color processing.
- `tone_classifier.py` - Classifies skin tones based on monk colors using the mean squared error method.

## Setup and Installation

### Requirements
- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- MediaPipe
- matplotlib (for visualizations)

### Installation
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/yourgithub/repo.git
cd repo
pip install -r requirements.txt
