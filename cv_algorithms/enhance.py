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

def apply_binary_threshold(img, threshold_value=127):
    """
    4. 手動二值化 (Manual Binarization)
    根據指定的 threshold_value 切割前景與背景
    """
    # 步驟 1：確保圖像是單通道灰階
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
        
    # 步驟 2：進行二值化 (大於 threshold_value 變 255，否則變 0)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 步驟 3：轉回 3 通道 (維持與 App 流水線相容的 RGB 格式)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def apply_otsu_threshold(img):
    """
    5. Otsu 自動二值化 (Otsu's Thresholding)
    讓演算法自動尋找最佳的分割門檻值
    """
    # 步驟 1：確保圖像是單通道灰階
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
        
    # 步驟 2：加上 THRESH_OTSU 旗標。
    # 這裡門檻值參數隨便填(例如0)，因為 Otsu 會自動計算並覆蓋它
    best_thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # (可選) 你可以在終端機偷偷印出來，看看 Otsu 這次幫你找的及格線是幾分
    print(f"[演算法日誌] Otsu 計算出的最佳門檻值為: {best_thresh}")
    
    # 步驟 3：轉回 3 通道
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def apply_adaptive_threshold(img, block_size=11, C_value=2):
    """
    6. 自適應二值化 (Adaptive Thresholding)
    解決光照不均的問題，將圖片切小塊，各自尋找最佳門檻。
    """
    # 步驟 1：確保圖像是單通道灰階
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
        
    # 步驟 2：防呆機制，確保 block_size 是 >= 3 的奇數
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1  # 如果是偶數，自動加 1 變成奇數
    if block_size < 3:
        block_size = 3
        
    C_value = int(C_value)

    # 步驟 3：執行自適應二值化 (使用高斯加權平均)
    binary = cv2.adaptiveThreshold(
        gray, 
        255,                                # 滿足條件時賦予的顏色 (純白)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,     # 計算鄰居平均分數的方法 (高斯權重)
        cv2.THRESH_BINARY,                  # 黑白切割法
        block_size,                         # 鄰居區塊大小
        C_value                             # 微調常數 (從平均分扣掉的數字)
    )
    
    # 步驟 4：轉回 3 通道維持系統相容性
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)