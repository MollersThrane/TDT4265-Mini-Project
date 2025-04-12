import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from glob import glob

def visualize_and_save_labels(output_dir, num_images=10):
    """
    Visualize and save the label bounding boxes on training images.
    
    Args:
        output_dir (str): Directory to save the visualized images.
        num_images (int): Number of images to visualize and save.
    """
    # Get a list of training images
    image_dir = "./data/lidar/train/images/"
    image_paths = glob(os.path.join(image_dir, "*.png"))
    # image_paths.sort()  # Sort the images to ensure consistent ordering
    image_paths = image_paths[:num_images]  # Limit to the specified number of images
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Get list of label files
    label_dir = "./data/lidar/train/labels/"
    label_paths = glob(os.path.join(label_dir, "*.txt"))
    # label_paths.sort()  # Sort the labels to ensure consistent ordering
    label_paths = label_paths[:num_images]  # Limit to the specified number of labels

    for image_path, label_path in zip(image_paths, label_paths):
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the label file
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Draw bounding boxes on the image
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.strip().split())
            x_center *= image.shape[1]
            y_center *= image.shape[0]
            width *= image.shape[1]
            height *= image.shape[0]

            # Convert YOLO format to bounding box coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Save the visualized image
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    output_dir = "./visualized_labels/"
    visualize_and_save_labels(output_dir, num_images=25)