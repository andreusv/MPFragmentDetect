import torch
import cv2
import numpy as np
import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def overlay_mask(image, mask, color=(0,255,0), alpha=0.4):
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask>0.5] = color,
    return cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)

def draw_bbox(image, box, label, color=(255,0,0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
