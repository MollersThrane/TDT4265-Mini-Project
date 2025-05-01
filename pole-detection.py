import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from glob import glob
import time  # Import the time module

def train_model():
    MODEL_NAME = "yolo11n.pt"
    model = YOLO(MODEL_NAME)
    data_path = "data.yaml"

    print("Starting training...")
    start_time = time.time()  # Start timing
    model.train(
        data=data_path,
        epochs=200,  # Increase epochs for longer training
        batch=32,  # Larger batch size for better convergence
        lr0=0.005,  # Higher initial learning rate
        lrf=0.001,  # Reduce learning rate at the end of training
        dropout=0.1,  # Dropout for regularization
        momentum=0.9,  # Higher momentum for smoother training
        weight_decay=0.0005,  # Regularization to prevent overfitting
        warmup_epochs=5,  # Warm-up for a smoother start
        imgsz=1024,  # Image size, you can try 1024 for better detail
        close_mosaic=20,  # Close mosaic augmentation
        iou=0.7,  # IoU threshold for detection
        conf=0.4,  # Confidence threshold for detection
        flipud=0.0,  # Disable vertical flip
        fliplr=0.5,  # Horizontal flip probability
        hsv_h=0.015,  # Random hue jitter
        hsv_s=0.7,  # Random saturation jitter
        hsv_v=0.4,  # Random value jitter
        scale=0.7,  # Scale augmentation
        translate=0.1, # Translation augmentation
        perspective=0.001,  # Perspective augmentation
        erasing=0.2,  # Erasing augmentation
        device="0",  # Specify GPU (0 for first GPU)
    )
    end_time = time.time()  # End timing
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

def load_latest_model():
    # Load the latest model
    run_dirs = sorted(glob("/work/tobiart/TDT4265-Mini-Project/runs/detect/train*/weights/best.pt"), key=os.path.getmtime)
    latest_model_path = run_dirs[-1]  # most recent
    model = YOLO(latest_model_path)
    return model

def model_predict(model):
    print("Starting predictions...")
    start_time = time.time()  # Start timing
    model.predict(
        source="data/lidar/test/images/",
        project="test_predictions/",
        name="test1",
        save_txt=True,
        save_conf=True  # <--- This adds the probability of each predicted box
    )
    end_time = time.time()  # End timing
    print(f"Predictions completed in {end_time - start_time:.2f} seconds.")

def print_metrics(model):
    print("Validating the model...")
    start_time = time.time()  # Start timing
    metrics = model.val()
    end_time = time.time()  # End timing

    print("Metrics:")
    print(f"Precision: {metrics.box.mp:.4f}")  # Precision
    print(f"Recall: {metrics.box.mr:.4f}")    # Recall
    print(f"F1 Score: {metrics.box.f1}")      # F1 Score
    print(f"mAP@0.5: {metrics.box.map50:.4f}")    # mAP at IoU=0.5
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}") # mAP at IoU=0.5:0.95
    print(f"Validation completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Train the model
    train_model()

    # Load the best model
    model = load_latest_model()

    # Print the metrics of the model
    print_metrics(model)

    # Make predictions on the test set
    #model_predict(model)

