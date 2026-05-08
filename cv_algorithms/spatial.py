import cv2
import numpy as np

def apply_mean_filter(img_gray, ksize=3):
    """均值濾波 (Mean Filter)：最基本的模糊，容易讓邊緣變糊"""
    return cv2.blur(img_gray, (ksize, ksize))

def apply_gaussian_filter(img_gray, ksize=3, sigma=0):
    """高斯濾波 (Gaussian Filter)：權重模糊，保留較多邊緣細節"""
    return cv2.GaussianBlur(img_gray, (ksize, ksize), sigmaX=sigma)

def apply_median_filter(img_gray, ksize=3):
    """中值濾波 (Median Filter)：去除胡椒鹽雜訊 (點狀雜訊) 的超級神器"""
    return cv2.medianBlur(img_gray, ksize)