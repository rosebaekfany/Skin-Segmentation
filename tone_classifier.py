import cv2
import numpy as np
import os

# Define monk colors as RGB
monk_colors = [
    np.array([246, 237, 228]),  # Monk 01
    np.array([243, 231, 219]),  # Monk 02
    np.array([247, 234, 208]),  # Monk 03
    np.array([234, 218, 186]),  # Monk 04
    np.array([215, 189, 150]),  # Monk 05
    np.array([160, 126, 86]),   # Monk 06
    np.array([130, 92, 67]),    # Monk 07
    np.array([96, 65, 52]),     # Monk 08
    np.array([61, 49, 42]),     # Monk 09
    np.array([41, 36, 32])      # Monk 10
]

def calculate_mse(color1, color2):
    """Calculate the mean squared error between two RGB colors."""
    return np.mean((color1 - color2) ** 2)

def classify_skin_tone(average_color, monk_colors):
    """Classify skin tone by finding the monk color with the minimum MSE."""
    mse_values = [calculate_mse(average_color, monk_color) for monk_color in monk_colors]
    min_mse_index = np.argmin(mse_values)
    return min_mse_index, mse_values[min_mse_index]

def main(mask_dir):
    """Process all masks in the directory to calculate average color and classify based on monk skin tones."""
    for filename in os.listdir(mask_dir):
        if filename.endswith(('.png', '.jpg')):
            mask_path = os.path.join(mask_dir, filename)
            mask = cv2.imread(mask_path)
            if mask is None:
                print(f"Failed to load mask for {filename}")
                continue

            
            skin_pixels = mask[mask.sum(axis=2) > 0]  
            if skin_pixels.size == 0:
                print(f"No skin pixels found in {filename}")
                continue

            average_color = np.mean(skin_pixels, axis=0)
            monk_index, _ = classify_skin_tone(average_color, monk_colors)
            print(f"{filename}: Classified as Monk Skin Tone {monk_index + 1} with Average Color: {average_color}")

if __name__ == "__main__":
    mask_dir = 'path_to_your_mask' 
    main(mask_dir)
