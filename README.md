# Enhanced Skin Segmentation and Color Tone Classification

Course project for **Digital Image Processing**: improving deep-learning-based skin segmentation with classical image-processing methods.

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

## Notebooks and data

- `project.ipynb`, `test on our dataset.ipynb` — experiments on the Pratheepan dataset and on our own images
- `Pratheepan_Dataset/`, `Ground_Truth/` — input images and ground-truth masks
- `Pratheepan_*_masks/`, `mediapipe-mask/` — masks produced by each method
- `skin_segmentation.pdf` — report

## Related

This work was extended in my bachelor's thesis: [SkinMap](https://github.com/rosebaekfany/SkinMap-A-Novel-Weighted-Full-Body-Skin-Segmentation) (CVPR 2025 DemoDiv Workshop).
