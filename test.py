import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import os
from PIL import Image
from ultralytics import YOLO

model1_path = YOLO(r'D:\data\yolov8\runs\classify\train\weights\best.pt')
# model2_path = YOLO(r'D:\data\yolov8\runs\classify\train\weights\best.pt')
image_path = r'D:\data\yolov8\3.jpg'


pil_image = Image.open(image_path)
# Load a model
result1 = model1_path(pil_image, save = True)  # load a custom model

# Load the image as a PIL Image
# image = np.array(pil_image)
# result2 = model2_path(image, save = True)
