import cv2
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

class BilateralFilter(object):
    def __init__(self, d=9, sigmaColor=75, sigmaSpace=75):
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img)
        # Apply bilateral filter
        img_np = cv2.bilateralFilter(img_np, self.d, self.sigmaColor, self.sigmaSpace)
        # Convert numpy array back to PIL image
        img = Image.fromarray(img_np)
        return img
