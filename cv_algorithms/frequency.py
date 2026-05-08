import cv2
import numpy as np

import cv2
import numpy as np

def apply_frequency_filter(img_gray, filter_shape="ideal", pass_type="low_pass", d0=30, n=2):
    """
    終極頻率域濾波器
    filter_shape: "ideal" (理想), "gaussian" (高斯), "butterworth" (巴特沃斯)
    pass_type: "low_pass" (低通/模糊), "high_pass" (高通/銳化)
    d0: 截止半徑
    n: 巴特沃斯專用階數 (Order)
    """
    img_float = np.float32(img_gray) / 255.0
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=[0, 1])

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # 建立距離矩陣 D(u,v)
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    
    # 防止 D0 為 0 導致分母爆炸
    d0 = max(d0, 0.1) 
    H = np.zeros((rows, cols), np.float32)
    
    # --- 1. 理想濾波器 (Ideal) ---
    if filter_shape == "ideal":
        if pass_type == "low_pass":
            H[D <= d0] = 1
        else:
            H[D > d0] = 1
            
    # --- 2. 高斯濾波器 (Gaussian) ---
    elif filter_shape == "gaussian":
        if pass_type == "low_pass":
            H = np.exp(-(D**2) / (2 * (d0**2)))
        else:
            H = 1 - np.exp(-(D**2) / (2 * (d0**2)))
            
    # --- 3. 巴特沃斯濾波器 (Butterworth) ---
    elif filter_shape == "butterworth":
        if pass_type == "low_pass":
            H = 1 / (1 + (D / d0)**(2 * n))
        else:
            # 為了高通避免中心點 (D=0) 分母為 0
            D_safe = np.where(D == 0, 0.0001, D)
            H = 1 / (1 + (d0 / D_safe)**(2 * n))
            H[crow, ccol] = 0 # 高通的中心直流分量必須是 0

    # 將單通道遮罩轉為雙通道 (實部與虛部)
    mask = np.zeros((rows, cols, 2), np.float32)
    mask[:,:,0] = H
    mask[:,:,1] = H
    
    # 套用遮罩並反轉換
    fshift_filtered = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered, axes=[0, 1])
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return np.uint8(img_back)

def apply_notch_filter(img_gray, u_k=7, v_k=0, d0=3, harmonics=5):
    """2.5 陷波濾波器"""
    img_float = np.float32(img_gray) / 255.0
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=[0, 1])

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2
    
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    mask = np.ones((rows, cols), np.float32)
    d0_sq = max(d0**2, 0.1)

    for i in range(1, harmonics + 1):
        u_center = ccol + i * u_k
        v_center = crow - i * v_k
        u_center_sym = ccol - i * u_k
        v_center_sym = crow + i * v_k

        D1_sq = (X - u_center)**2 + (Y - v_center)**2
        D2_sq = (X - u_center_sym)**2 + (Y - v_center_sym)**2

        notch1 = 1.0 - np.exp(-D1_sq / (2 * d0_sq))
        notch2 = 1.0 - np.exp(-D2_sq / (2 * d0_sq))

        mask *= (notch1 * notch2)

    mask_2ch = np.dstack((mask, mask))
    fshift_filtered = dft_shift * mask_2ch
    f_ishift = np.fft.ifftshift(fshift_filtered, axes=[0, 1])
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(img_back)