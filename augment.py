import os
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import argparse

def load_config(config_file):
    config = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                operations = line.split(';')
                config.append(operations)
    return config

def custom_brightness(image, factor):
    result = image.copy()
    result = result.astype(np.float32)
    result = np.clip(result * factor, 0, 255)
    return result.astype(np.uint8)

def color_channel_shift(image, channel, value):
    result = image.copy()
    result[:, :, channel] = np.clip(result[:, :, channel].astype(np.int32) + value, 0, 255)
    return result

def gaussian_noise(image, mean, std):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def gaussian_blur(image, kernel_size, sigma):
    k = kernel_size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    result = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[2]):
        for i in range(k, image.shape[0] - k):
            for j in range(k, image.shape[1] - k):
                region = image[i-k:i+k+1, j-k:j+k+1, channel]
                result[i, j, channel] = np.sum(region * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)

def custom_scale(image, scale_x, scale_y):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_y), int(width * scale_x)
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x / scale_x)
            src_y = int(y / scale_y)
            if 0 <= src_x < width and 0 <= src_y < height:
                result[y, x] = image[src_y, src_x]
    return result

def flip(image, flip_code):
    height, width = image.shape[:2]
    result = np.zeros_like(image)

    if flip_code > 0: # Horizontal
        for y in range(height):
            for x in range(width):
                result[y, x] = image[y, width - 1 - x]

    elif flip_code == 0: # Vertical
        for y in range(height):
            for x in range(width):
                result[y, x] = image[height - 1 - y, x]

    else: # Both
        for y in range(height):
            for x in range(width):
                result[y, x] = image[height - 1 - y, width - 1 - x]

    return result

def shear(image, shear_factor):
    height, width = image.shape[:2]
    matrix = np.array([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (width, height))

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def apply_augmentation(image, op_name, params):
    if op_name == 'brightness':
        factor = float(params[0])
        return custom_brightness(image, factor)
    elif op_name == 'channel_shift':
        channel = int(params[0])
        value = int(params[1])
        return color_channel_shift(image, channel, value)
    elif op_name == 'noise':
        mean = float(params[0])
        std = float(params[1])
        return gaussian_noise(image, mean, std)
    elif op_name == 'blur':
        kernel_size = int(params[0])
        sigma = float(params[1])
        return gaussian_blur(image, kernel_size, sigma)
    elif op_name == 'scale':
        scale_x = float(params[0])
        scale_y = float(params[1])
        return custom_scale(image, scale_x, scale_y)
    elif op_name == 'shear':
        shear_factor = float(params[0])
        return shear(image, shear_factor)
    elif op_name == 'flip':
        flip_code = int(params[0])
        return flip(image, flip_code)
    elif op_name == 'rotation':
        angle = int(params[0])
        return rotate_image(image, angle)
    elif op_name == 'contrast':
        alpha = float(params[0])
        return adjust_contrast(image, alpha)
    else:
        print(f"Invalid operation: {op_name}")
        return image

def process_image(image_path, output_dir, config):
    image = cv2.imread(image_path)
    for i, operations in enumerate(config):
        augmented_image = image.copy()
        op_names = []
        for operation in operations:
            op_parts = operation.split(',')
            op_name = op_parts[0]
            op_params = op_parts[1:] if len(op_parts) > 1 else []
            augmented_image = apply_augmentation(augmented_image, op_name, op_params)
            op_names.append(op_name)

        op_suffix = '_'.join(op_names)
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{op_suffix}_{i+1}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, augmented_image)

def process_directory(input_dir, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir, config)

def main():
    parser = argparse.ArgumentParser(description="Image augmentation program")
    parser.add_argument("config_file", help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(args.config_file)

    root = tk.Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="Select input directory")

    if not input_dir:
        print("No directory selected. Exiting.")
        return

    output_dir = input_dir + "_aug"
    process_directory(input_dir, output_dir, config)
    print(f"Augmentation complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()