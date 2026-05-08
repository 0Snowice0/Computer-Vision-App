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

def apply_hough_transform(img_gray, do_lines, line_t, line_len, line_gap, do_circles, param2, min_r, max_r, min_dist):
    """5. 霍氏轉換 (進階版)"""
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
        
    canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    if do_lines:
        edges = cv2.Canny(img_gray, 50, 150)
        # 加入了 line_gap 參數
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=line_t, minLineLength=line_len, maxLineGap=line_gap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if do_circles:
        blurred = cv2.medianBlur(img_gray, 5)
        # 加入了 min_dist 參數
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
                                   param1=50, param2=param2, minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(canvas, (i[0], i[1]), i[2], (255, 0, 0), 2)
                cv2.circle(canvas, (i[0], i[1]), 2, (0, 0, 255), 3)

    return canvas