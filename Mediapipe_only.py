import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

model_path = "/media/carla/DIP/Project/model/selfie_multiclass_256x256.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
BG_COLOR = (192, 192, 192) 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image, input_shape):
    input_height, input_width = input_shape[1:3]
    image_resized = cv2.resize(image, (input_width, input_height))
    image_normalized = image_resized.astype(np.float32) / 255.0
    return np.expand_dims(image_normalized, axis=0)

def postprocess_output(output, target_shape, threshold=3):
    output = output.squeeze()  # Remove batch dimension
    output_aggregated = np.sum(output, axis=-1)  # Sum across the channel dimension
    mask_resized = cv2.resize(output_aggregated, (target_shape[1], target_shape[0]))
    mask = mask_resized > threshold
    return mask  # Return single-channel mask

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
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(image_dir, image_file)
            mask_filename = os.path.splitext(image_file)[0] + '.png'  # Assuming mask format is .png
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                continue
            
            image = cv2.imread(image_path)
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_mask = true_mask > 0 
            
            input_shape = input_details[0]['shape']
            input_image = preprocess_image(image, input_shape)
            
            interpreter.set_tensor(input_details[0]['index'], input_image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            pred_mask = postprocess_output(output_data, true_mask.shape, threshold=0.5)  # Adjust threshold if needed
            
            # Debug prints to verify shapes
            # print(f"Image shape: {image.shape}")
            # print(f"True mask shape: {true_mask.shape}")
            # print(f"Predicted mask shape: {pred_mask.shape}")
                        # Save the predicted mask
            pred_mask_1 = pred_mask[:,:,2]
            pred_mask_2 = pred_mask[:,:,3]
            pred_mask = pred_mask_1 + pred_mask_2

            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            pred_mask_33 = np.repeat(pred_mask[:, :, np.newaxis], 3, axis=2)   # Expand mask to have 3 channels
            output_image = np.where(pred_mask_33, image, bg_image)

            output_mask_path = os.path.join(output_mask_dir, mask_filename)
            cv2.imwrite(output_mask_path, pred_mask * 255)  # Save as binary mask image (0 or 255)            

            # Calculate metrics
            iou, precision, recall, f1, dice = calculate_metrics(pred_mask, true_mask)
            
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            dice_scores.append(dice)
            
            print(f"Image: {image_file}, IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Dice Score: {dice:.4f}")

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
