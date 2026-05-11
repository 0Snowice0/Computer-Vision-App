import cv2
import numpy as np

def apply_sobel(img_gray, direction="both", ksize=3):
    """3. Sobel 邊緣檢測 (進階版)"""
    if direction == "x_only":
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        return cv2.convertScaleAbs(sobel)
    elif direction == "y_only":
        sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
        return cv2.convertScaleAbs(sobel)
    else: # both
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
        return cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)

def apply_laplacian_edge(img_gray, ksize=3):
    """純拉普拉斯邊緣檢測 (若要抗噪，請在 UI 前方勾選高斯濾波)"""
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(laplacian)

def apply_canny(img_gray, t1=100, t2=200, aperture_size=3, l2_gradient=False):
    """4. Canny 邊緣檢測 (進階版)"""
    return cv2.Canny(img_gray, t1, t2, apertureSize=aperture_size, L2gradient=l2_gradient)

def apply_hough_transform(img_input, do_lines, line_t, line_len, line_gap, do_circles, param2, min_r, max_r, min_dist):
    """
    修正版霍氏轉換：支援在現有彩色圖上累加繪製
    """
    # 1. 準備運算用的灰階圖 (修正 BGR 權重問題)
    if len(img_input.shape) == 3:
        img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        canvas = img_input.copy() # 直接在輸入的圖上畫，不重置畫布
    else:
        img_gray = img_input
        # 如果是灰階圖，要轉成 BGR 格式才能畫出有顏色的線
        canvas = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)

    # 2. 偵測直線
    if do_lines:
        # Canny 的門檻值可以稍微調降，增加靈敏度
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=line_t, minLineLength=line_len, maxLineGap=line_gap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 使用 BGR 顏色：綠色 (0, 255, 0)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 3. 偵測圓形
    if do_circles:
        # 圓形偵測前做一點模糊，效果會好很多
        blurred = cv2.medianBlur(img_gray, 5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
            param1=50, param2=param2, minRadius=min_r, maxRadius=max_r
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # 畫圓周：藍色 (255, 0, 0)
                cv2.circle(canvas, (i[0], i[1]), i[2], (255, 0, 0), 2)
                # 畫圓心：紅色 (0, 0, 255)
                cv2.circle(canvas, (i[0], i[1]), 2, (0, 0, 255), 3)

    return canvas