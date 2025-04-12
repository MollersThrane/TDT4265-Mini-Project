import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"

model = YOLO(MODEL_NAME)

data_path = "data/images/"

