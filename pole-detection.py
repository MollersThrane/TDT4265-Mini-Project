import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from glob import glob

MODEL_NAME = "yolo11n.pt"

model = YOLO(MODEL_NAME)

data_path = "data.yaml"

# results = model.train(
#     data=data_path,
#     epochs=5,
#     device="0",
# )
results = model.train(
    data=data_path,
    epochs=100,  # Increase epochs for longer training
    batch=32,  # Larger batch size for better convergence
    lr0=0.01,  # Higher initial learning rate
    lrf=0.005,  # Reduce learning rate at the end of training
    momentum=0.9,  # Higher momentum for smoother training
    weight_decay=0.0005,  # Regularization to prevent overfitting
    warmup_epochs=3,  # Warm-up for a smoother start
    imgsz=1024,  # Image size, you can try 1024 for better detail
    iou=0.7,  # IoU threshold for detection
    conf=0.4,  # Confidence threshold for detection
    flipud=0.0,  # Disable vertical flip
    fliplr=0.5,  # Horizontal flip probability
    # hsv_h=0.015,  # Random hue jitter
    # hsv_s=0.7,  # Random saturation jitter
    # hsv_v=0.4,  # Random value jitter
    device="0",  # Specify GPU (0 for first GPU)
)

# Load the best model
run_dirs = sorted(glob("/work/tobiart/TDT4265-Mini-Project/runs/detect/train*/weights/best.pt"), key=os.path.getmtime)
best_model_path = run_dirs[-1]  # most recent
model = YOLO(best_model_path)

# Test the model on a sample image
# image_dir = "./data/lidar/test/images/"
# image_paths = glob(os.path.join(image_dir, "*.png"))

# for image_path in image_paths:
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     test_results = model.predict(
#         source=image_path,
#         conf=0.2,
#         save=True
#     )

    # Save the annotated image
    # output_path = "output/" + os.path.basename(image_path)
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # print(f"Annotated image saved to {output_path}")

