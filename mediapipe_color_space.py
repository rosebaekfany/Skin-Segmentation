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

def evaluate_model_on_dataset(image_dir, mask_dir, output_mask_dir):
    os.makedirs(output_mask_dir, exist_ok=True)
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    dice_scores = []

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png') or image_file.endswith('.jpeg'):
            image_path = os.path.join(image_dir, image_file)
            mask_filename = os.path.splitext(image_file)[0] + '.png'  # Assuming mask format is .png
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                print("Failed to find mask for", image_file)
                continue
            
            image = cv2.imread(image_path)
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_mask = true_mask > 0  # Convert to binary mask
            
            input_shape = input_details[0]['shape']
            input_image = preprocess_image(image, input_shape)
            
            interpreter.set_tensor(input_details[0]['index'], input_image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            pred_mask = postprocess_output(output_data, true_mask.shape, threshold=3)  # Adjust threshold if needed

            pred_mask_1 = pred_mask[:,:,2]
            pred_mask_2 = pred_mask[:,:,3]
            pred_mask = pred_mask_1 + pred_mask_2

            # Apply HSV skin color thresholding
            skin_color_mask = apply_skin_color_threshold(image)
            thresholded_pred_mask = cv2.bitwise_and(pred_mask, pred_mask, mask=skin_color_mask)

            # Save the thresholded mask
            thresholded_output_path = os.path.join(output_mask_dir, mask_filename)
    

            cv2.imwrite(thresholded_output_path, thresholded_pred_mask * 255)

            # Calculate metrics
            iou, precision, recall, f1, dice = calculate_metrics(thresholded_pred_mask, true_mask)
            
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            dice_scores.append(dice)
            
            # print(f"Image: {image_file}, IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Dice Score: {dice:.4f}")

    mean_iou = np.mean(iou_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_dice = np.mean(dice_scores)
    
    print(f"Mean IoU over dataset: {mean_iou:.4f}")
    print(f"Mean Precision over dataset: {mean_precision:.4f}")
    print(f"Mean Recall over dataset: {mean_recall:.4f}")
    print(f"Mean F1 Score over dataset: {mean_f1:.4f}")
    print(f"Mean Dice Score over dataset: {mean_dice:.4f}")

image_dir = "images_address"  
mask_dir = "masks_address"
output_mask_dir = "out_put_saving_address" 
evaluate_model_on_dataset(image_dir, mask_dir, output_mask_dir)