import cv2
import numpy as np
import tkinter.filedialog as fd

def interactive_perspective_transform(img_rgb):
    """
    開啟獨立視窗讓使用者點擊 4 個點，完成後回傳轉正的影像。
    """
    # 你的 App 是存 RGB，但 OpenCV 顯示需要 BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    clone = img_bgr.copy()
    src_pts = []

    def click_event(event, x, y, flags, param):
        # 點擊左鍵，且還沒選滿 4 個點時
        if event == cv2.EVENT_LBUTTONDOWN and len(src_pts) < 4:
            src_pts.append([x, y])
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 Points", clone)

    cv2.imshow("Select 4 Points", clone)
    cv2.setMouseCallback("Select 4 Points", click_event)
    print("📌 [系統提示] 請在彈出的 OpenCV 視窗中，依序點擊 左上、右上、右下、左下 4 個角。")
    print("⚠️ [警告] 點完 4 個點之前，請勿點擊原本的深色主視窗，以免程式失去回應！")

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 🚀 新增這兩行：檢查視窗是否還活著！
        # 如果使用者按了右上角的 X 關閉視窗，WND_PROP_VISIBLE 的值會變成 0 或 -1
        if cv2.getWindowProperty("Select 4 Points", cv2.WND_PROP_VISIBLE) < 1:
            break
            
        # 按下 ESC (27) 或點滿 4 個點時跳出
        if len(src_pts) == 4 or key == 27: 
            break

    # 迴圈結束後，確保視窗被完全銷毀
    cv2.destroyAllWindows()

    if len(src_pts) == 4:
        pts1 = np.float32(src_pts)
        width, height = 500, 500
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img_bgr, M, (width, height))
        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    else:
        print("⚠️ [系統提示] 未選滿 4 個點，取消轉換。")
        return img_rgb # 取消則回傳原圖

def align_images_sift(src_img_rgb):
    """
    跳出檔案選擇器讓使用者載入目標圖片，執行 SIFT 對齊後回傳影像。
    """
    # 跳出視窗選擇要對齊的目標影像 (IMG_S.jpg)
    path = fd.askopenfilename(title="請選擇要對齊的『目標影像 (Target)』", 
                              filetypes=[("Image", "*.jpg *.png *.jpeg")])
    if not path:
        return src_img_rgb

    # 讀取目標影像並處理色彩空間
    target_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    src_bgr = cv2.cvtColor(src_img_rgb, cv2.COLOR_RGB2BGR)

    gray_src = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_src, des_src = sift.detectAndCompute(gray_src, None)
    kp_tgt, des_tgt = sift.detectAndCompute(gray_tgt, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des_src, des_tgt, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) > 10:
        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        tgt_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC, 5.0)
        
        h_tgt, w_tgt = target_bgr.shape[:2]
        
        # 產生扭曲後的城堡
        warped_src = cv2.warpPerspective(src_bgr, H, (w_tgt, h_tgt))
        
        print("✅ [系統提示] SIFT 對齊成功！回傳兩張獨立影像。")
        
        # 🚀 關鍵修改：不要 hstack 了！分別轉換成 RGB 後，用逗號隔開一起回傳
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        warped_rgb = cv2.cvtColor(warped_src, cv2.COLOR_BGR2RGB)
        
        return target_rgb, warped_rgb
        
    else:
        print("⚠️ [系統提示] 匹配點太少，無法對齊！")
        # 如果失敗，回傳 None 和原本的圖防呆
        return None, src_img_rgb