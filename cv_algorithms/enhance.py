import cv2
import numpy as np

def apply_histogram_equalization(img):
    """1. 直方圖等化 (支援灰階與彩色)"""
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        # 轉換為 YCrCb，只對 Y (亮度) 通道進行等化
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)

def apply_clahe(img, clip_limit=2.0, grid_size=8):
    """2. CLAHE (支援灰階與彩色)"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    if len(img.shape) == 2:
        return clahe.apply(img)
    elif len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)

def apply_negative(img):
    """3. 負片轉換 (Image Negative)"""
    return cv2.bitwise_not(img)