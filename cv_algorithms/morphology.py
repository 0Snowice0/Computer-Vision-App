import cv2
import numpy as np

def morphological_hole_filling(img, kernel):
    """條件式膨脹：填補空洞"""
    mask = cv2.bitwise_not(img)
    X = np.zeros_like(img)
    X[0, :] = 255
    X[-1, :] = 255
    X[:, 0] = 255
    X[:, -1] = 255
    X = cv2.bitwise_and(X, mask)
    
    while True:
        X_next = cv2.dilate(X, kernel, iterations=1)
        X_next = cv2.bitwise_and(X_next, mask)
        if np.array_equal(X, X_next):
            break
        X = X_next
    return cv2.bitwise_not(X)

def alternating_sequential_filter(img):
    """
    交替連續濾波 (ASF)
    原理：依序使用從小到大 (3x3, 5x5, 7x7) 的圓形結構元素進行「開運算 -> 閉運算」
    特點：極度溫和的深度去噪，不破壞大物體邊界結構。
    """
    result = img.copy()
    # 依序使用 3, 5, 7 的尺寸 (OpenCV 建議 kernel size 為奇數)
    for k in [3, 5, 7]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result

def apply_advanced_morphology(img, operation, se_matrix, iterations=1):
    """執行 11 大形態學演算法"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k_pos = np.where(se_matrix == 1, 1, 0).astype(np.uint8)
    k_hm = se_matrix.astype(np.int8)
    iterations = int(iterations)

    if operation == "膨脹":
        return cv2.dilate(img, k_pos, iterations=iterations)
    elif operation == "侵蝕":
        return cv2.erode(img, k_pos, iterations=iterations)
    elif operation == "Opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, k_pos, iterations=iterations)
    elif operation == "Closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_pos, iterations=iterations)
    elif operation == "邊界抽取":
        # 原圖 - 侵蝕圖 (精細內部邊緣)
        eroded = cv2.erode(img, k_pos, iterations=iterations)
        return cv2.subtract(img, eroded)
    elif operation == "形態學梯度":
        # 膨脹圖 - 侵蝕圖 (OpenCV 的 MORPH_GRADIENT 預設就是這個，粗邊緣)
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k_pos, iterations=iterations)
    elif operation == "Top-Hat":
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k_pos, iterations=iterations)
    elif operation == "Black-Hat":
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k_pos, iterations=iterations)
    elif operation == "Hit-or-Miss":
        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(bin_img, cv2.MORPH_HITMISS, k_hm)
    elif operation == "空洞填補":
        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return morphological_hole_filling(bin_img, k_pos)
    elif operation == "交替連續濾波 (ASF)":
        return alternating_sequential_filter(img)
        
    return img