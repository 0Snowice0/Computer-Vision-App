import cv2
import numpy as np

def _ensure_binary(img):
    """智慧防呆：確保輸入給物件分析的影像是二值化影像 (黑白)"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 如果影像不是純黑白 (唯一值超過2個)，就自動幫它做 Otsu 二值化
    unique_vals = np.unique(img)
    if len(unique_vals) > 2:
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return bin_img
    return img

def apply_connected_components(img, connectivity=8):
    """
    連通區域標記 (Connected Components)
    將獨立的區塊塗上不同的顏色，方便視覺化與計數。
    """
    bin_img = _ensure_binary(img)
    
    # num_labels 包含背景，所以實際物件數量是 num_labels - 1
    num_labels, labels = cv2.connectedComponents(bin_img, connectivity=connectivity)
    
    # 建立彩色映射 (HSV轉BGR) 來讓不同標籤有不同顏色
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    
    # 把背景 (標籤 0) 設回純黑色
    labeled_img[label_hue == 0] = 0
    
    return labeled_img

def apply_find_contours(img, mode_str="EXTERNAL", method_str="SIMPLE"):
    """
    尋找輪廓 (Find Contours)
    找出物體邊界並畫上螢光綠色外框。
    """
    bin_img = _ensure_binary(img)
    
    mode_dict = {
        "EXTERNAL": cv2.RETR_EXTERNAL,
        "LIST": cv2.RETR_LIST,
        "TREE": cv2.RETR_TREE
    }
    method_dict = {
        "SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
        "NONE": cv2.CHAIN_APPROX_NONE
    }
    
    mode = mode_dict.get(mode_str, cv2.RETR_EXTERNAL)
    method = method_dict.get(method_str, cv2.CHAIN_APPROX_SIMPLE)
    
    contours, hierarchy = cv2.findContours(bin_img, mode, method)
    
    # 為了讓綠色外框明顯，我們把原圖轉成較暗的彩色圖當底底
    if len(img.shape) == 2:
        out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out_img = img.copy()
        
    # 將圖片稍微調暗，讓綠色輪廓更凸顯
    out_img = (out_img * 0.5).astype(np.uint8)
    
    # 畫上輪廓 (螢光綠, 線條粗細2)
    cv2.drawContours(out_img, contours, -1, (0, 255, 0), 2)
    
    return out_img