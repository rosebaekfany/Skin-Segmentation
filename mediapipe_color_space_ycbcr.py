import cv2
import numpy as np
import tensorflow as tf
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
import os
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

model_path = "Path_to_model"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
def apply_skin_color_threshold(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0,30, 90], dtype=np.uint8)
    upper_skin = np.array([255, 255, 255], dtype=np.uint8)
    skin_color_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    return skin_color_mask

BG_COLOR = (192, 192, 192) 

def rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


def create_ycbcr_mask(image):
    ycbcr_image = rgb_to_ycbcr(image)
    y, cb, cr = ycbcr_image[:,:,0], ycbcr_image[:,:,1], ycbcr_image[:,:,2]
    mask = (cr > 135) & (cb > 85) & (y > 80) & (cr <= (1.5862*cb)+20) & (cr >= (0.3448*cb)+76.2069) & \
           (cr >= (-4.5652*cb)+234.5652) & (cr <= (-1.15*cb)+301.75) & (cr <= (-2.2857*cb)+432.85)
    return mask.astype(np.uint8)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def apply_histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def calculate_metrics(pred_mask, true_mask):
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    iou = jaccard_score(true_flat, pred_flat, average='binary')
    precision = precision_score(true_flat, pred_flat, average='binary')
    recall = recall_score(true_flat, pred_flat, average='binary')
    f1 = f1_score(true_flat, pred_flat, average='binary')
    dice = (2 * precision * recall) / (precision + recall)  # Dice coefficient

    return iou, precision, recall, f1, dice

def evaluate_model_on_dataset(image_dir, mask_dir, output_ycbcr_dir):
    os.makedirs(output_ycbcr_dir, exist_ok=True)

    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png') or image_file.endswith('.jpeg'):
            image_path = os.path.join(image_dir, image_file)
            mask_filename = os.path.splitext(image_file)[0] + '.png'  # Assuming mask format is .png
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                print(f"Mask not found for image {image_file}")
                continue

            image = cv2.imread(image_path)
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_mask = true_mask > 0  # Convert to binary mask
            
            # Create mask
            ycbcr_mask = create_ycbcr_mask(image)

            # Save mask
            cv2.imwrite(os.path.join(output_ycbcr_dir, mask_filename), ycbcr_mask * 255)

            # Evaluate mask
            iou = jaccard_score(true_mask.flatten(), ycbcr_mask.flatten(), average='binary')
            precision = precision_score(true_mask.flatten(), ycbcr_mask.flatten(), average='binary')
            recall = recall_score(true_mask.flatten(), ycbcr_mask.flatten(), average='binary')
            f1 = f1_score(true_mask.flatten(), ycbcr_mask.flatten(), average='binary')

            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            # print(f"YCbCr - Image: {image_file}, IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    print(f"Mean IoU over dataset for YCbCr: {np.mean(iou_scores):.4f}")
    print(f"Mean Precision over dataset for YCbCr: {np.mean(precision_scores):.4f}")
    print(f"Mean Recall over dataset for YCbCr: {np.mean(recall_scores):.4f}")
    print(f"Mean F1 Score over dataset for YCbCr: {np.mean(f1_scores):.4f}")

image_dir = "images_address"  
mask_dir = "masks_address"
output_mask_dir = "out_put_saving_address" 
evaluate_model_on_dataset(image_dir, mask_dir, output_mask_dir)