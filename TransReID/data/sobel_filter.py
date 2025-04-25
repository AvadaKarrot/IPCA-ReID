from PIL import Image, ImageFile
import numpy as np
import cv2
import torch
from torchvision import transforms as T
# 自定义Sobel滤波transform
class SobelFilter(object):
    def __init__(self, ksize =  3):
        self.ksize = ksize
    def __call__(self, img):
        # 转换为 numpy 数组
        img = np.array(img)
        
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 应用 Sobel 滤波器
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.ksize)
        sobel = cv2.magnitude(sobelx, sobely)
        
        # 将结果归一化到0-255范围
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
        sobel = sobel.astype(np.uint8)
        
        # 将单通道图像转换为3通道图像
        sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
        
        # 转换回 PIL 图像
        sobel_img = Image.fromarray(sobel)
        
        return sobel_img
    
# 自定义Sobel滤波transform
class SobelFilter_sep_Add(object):
    def __init__(self, ksize=3, alpha=0.8):
        self.ksize = ksize
        self.alpha = alpha

    def __call__(self, img):
        # 转换为 numpy 数组
        img = np.array(img)
        img_orig = img.astype(np.float32)
        
        # 对每个通道应用 Sobel 滤波器
        sobel_channels = []
        for i in range(3):  # 对 R, G, B 每个通道分别进行处理
            channel = img[:, :, i]
            sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=self.ksize)
            sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=self.ksize)
            sobel = cv2.magnitude(sobelx, sobely)
            
            # 将结果归一化到0-255范围
            sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
            sobel_channels.append(sobel)
        
        # 合并通道
        sobel = np.stack(sobel_channels, axis=-1).astype(np.uint8)
        
        # # 对 Sobel 滤波后的图像进行亮度调整
        sobel = sobel * (np.mean(img_orig) / np.mean(sobel))
        sobel = np.clip(sobel, 0, 255).astype(np.float32)
        
        # 混合原始图像和 Sobel 滤波图像
        combined_img = cv2.addWeighted(img_orig, self.alpha, sobel, 1 - self.alpha, 0)
        combined_img = np.clip(combined_img, 0, 255).astype(np.uint8)
        
        # 转换回 PIL 图像
        sobel_img = Image.fromarray(combined_img)
        
        return sobel_img