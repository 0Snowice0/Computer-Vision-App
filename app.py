import streamlit as st
import cv2
import numpy as np
import pandas as pd

def get_hist_data(img):
    """將 OpenCV 計算的直方圖轉換為 Streamlit 可以畫圖的 Pandas 格式"""
    if len(img.shape) == 2:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        return pd.DataFrame({"Gray": hist})
    else:
        # 讀取 R, G, B 三個通道
        data = {}
        channels = ['R', 'G', 'B']
        for i, col in enumerate(channels):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
            data[col] = hist
        return pd.DataFrame(data)

# 🌟 從我們自己寫的模組資料夾中，把函數 import 進來
from cv_algorithms.enhance import apply_histogram_equalization, apply_clahe, apply_negative
from cv_algorithms.frequency import apply_frequency_filter, apply_notch_filter
from cv_algorithms.spatial import apply_mean_filter, apply_gaussian_filter, apply_median_filter
from cv_algorithms.feature import apply_sobel, apply_laplacian_edge, apply_canny, apply_hough_transform
# --- App 介面設定 ---
st.set_page_config(layout="wide")
st.title("影像處理 App 🪄")

# ==========================================
# 🏷️ 統一名稱定義
# ==========================================
algo_negative = "負片轉換 (反轉)"
algo_eq = "直方圖等化"
algo_clahe = "CLAHE"
algo_mean = "均值濾波 (平滑)"
algo_gaussian = "高斯濾波 (平滑)"
algo_median = "中值濾波 (去雜訊)"
algo_laplacian = "拉普拉斯 (邊緣強化)"
algo_filter = "頻率域濾波 (進階)"
algo_notch = "陷波濾波 (去週期波)"
algo_sobel = "Sobel 邊緣"
algo_canny = "Canny 邊緣"
algo_hough = "霍氏轉換 (特徵)"

all_algos = [algo_negative, algo_eq, algo_clahe, algo_mean, algo_gaussian, algo_median, algo_laplacian, 
             algo_filter, algo_notch, algo_sobel, algo_canny, algo_hough]

# ==========================================
# 🧠 Session State 追蹤
# ==========================================
if 'process_order' not in st.session_state:
    st.session_state.process_order = []

def update_order(algo_name):
    if st.session_state[algo_name]: 
        if algo_name not in st.session_state.process_order:
            st.session_state.process_order.append(algo_name)
    else:
        if algo_name in st.session_state.process_order:
            st.session_state.process_order.remove(algo_name)

def clear_all_states():
    st.session_state.process_order = []
    for key in all_algos:
        if key in st.session_state:
            st.session_state[key] = False

# ==========================================
# 🎛️ 左側控制台
# ==========================================
with st.sidebar:
    st.header("⚙️ 排程控制台")
    st.subheader("📋 目前執行順序")
    if len(st.session_state.process_order) > 0:
        flow_text = "\n\n⬇️\n\n".join([f"**{item}**" for item in st.session_state.process_order])
        st.info(flow_text)
    else:
        st.warning("請在下方勾選演算法")
    st.divider()

    # 📁 1. 亮度與對比
    with st.expander("🌟 亮度與對比", expanded=False):
        use_negative = st.checkbox(algo_negative, key=algo_negative, on_change=update_order, args=(algo_negative,))
        use_eq = st.checkbox(algo_eq, key=algo_eq, on_change=update_order, args=(algo_eq,))
        use_clahe = st.checkbox(algo_clahe, key=algo_clahe, on_change=update_order, args=(algo_clahe,))
        if use_clahe:
            c_limit = st.slider("對比度限制", 1.0, 10.0, 2.0, 0.1)
            g_size = st.slider("網格大小", 4, 32, 8, 4)

    # 📁 2. 空間域濾波 (新加入的家族！)
    with st.expander("🖼️ 空間域平滑與銳化", expanded=True):
        st.caption("💡 提示：遮罩大小必須為奇數 (3, 5, 7...)")
        
        use_mean = st.checkbox(algo_mean, key=algo_mean, on_change=update_order, args=(algo_mean,))
        if use_mean:
            mean_k = st.slider("均值遮罩大小", 3, 31, 3, step=2, key="mean_k")
            
        use_gaussian = st.checkbox(algo_gaussian, key=algo_gaussian, on_change=update_order, args=(algo_gaussian,))
        if use_gaussian:
            gauss_k = st.slider("高斯遮罩大小", 3, 31, 5, step=2, key="gauss_k")
            gauss_s = st.slider("高斯標準差 (Sigma)", 0.0, 10.0, 0.0, step=0.5, help="設為0則由遮罩大小自動計算")
            
        use_median = st.checkbox(algo_median, key=algo_median, on_change=update_order, args=(algo_median,))
        if use_median:
            median_k = st.slider("中值遮罩大小", 3, 31, 5, step=2, key="median_k")

    # 📁 3. 頻譜與濾波 (大幅升級！)
    with st.expander("🌊 頻譜與濾波 (進階)", expanded=False):
        use_filter = st.checkbox(algo_filter, key=algo_filter, on_change=update_order, args=(algo_filter,))
        if use_filter:
            # 種類選擇
            f_shape_raw = st.selectbox("濾波器形狀", ["ideal (理想-會有波紋)", "gaussian (高斯-平滑過渡)", "butterworth (巴特沃斯-可控)"], index=2)
            f_shape = f_shape_raw.split(" ")[0] # 取出英文代號
            
            f_pass_raw = st.selectbox("通關類型", ["low_pass (低通/平滑)", "high_pass (高通/銳化)"])
            f_pass = f_pass_raw.split(" ")[0]
            
            f_d0 = st.slider("截止頻率 (D0半徑)", 5, 300, 50)
            
            if f_shape == "butterworth":
                f_n = st.slider("巴特沃斯階數 (n)", 1, 10, 2, help="越高越接近理想濾波器，越低越接近高斯")
            else:
                f_n = 2 # 預設防呆

        use_notch = st.checkbox(algo_notch, key=algo_notch, on_change=update_order, args=(algo_notch,))
        if use_notch:
            notch_u = st.slider("X 軸偏移 (U)", -150, 150, 7)
            notch_v = st.slider("Y 軸偏移 (V)", -150, 150, 0)
            notch_d0 = st.slider("消除半徑 (D0)", 1, 50, 3)
            notch_n = st.slider("倍頻數量", 1, 10, 5)

# 📁 4. 邊緣與特徵 (大師級參數版)
    with st.expander("📐 邊緣與特徵", expanded=True):
        
        # --- Sobel ---
        use_sobel = st.checkbox(algo_sobel, key=algo_sobel, on_change=update_order, args=(algo_sobel,))
        if use_sobel:
            sobel_dir = st.selectbox("偵測方向", ["both (雙向)", "x_only (只抓垂直線)", "y_only (只抓水平線)"])
            sobel_dir = sobel_dir.split(" ")[0]
            sobel_k = st.slider("Sobel 遮罩大小", 1, 7, 3, step=2, key="sobel_k")
            
        # --- Laplacian ---
        use_laplacian = st.checkbox(algo_laplacian, key=algo_laplacian, on_change=update_order, args=(algo_laplacian,))
        if use_laplacian:
            lap_k = st.slider("Laplacian 遮罩大小", 1, 7, 3, step=2, key="lap_k2")

        # --- Canny ---
        use_canny = st.checkbox(algo_canny, key=algo_canny, on_change=update_order, args=(algo_canny,))
        if use_canny:
            canny_t1 = st.slider("最低門檻 (T1)", 0, 255, 100)
            canny_t2 = st.slider("最高門檻 (T2)", 0, 255, 200)
            canny_ap = st.slider("內部 Sobel 遮罩", 3, 7, 3, step=2)
            canny_l2 = st.checkbox("L2 精準梯度計算", value=False)

        # --- Hough ---
        use_hough = st.checkbox(algo_hough, key=algo_hough, on_change=update_order, args=(algo_hough,))
        if use_hough:
            do_lines = st.checkbox("📏 偵測直線", value=True)
            if do_lines:
                hough_line_t = st.slider("直線靈敏度 (越低越敏感)", 10, 200, 80)
                hough_line_len = st.slider("最短線段長度", 10, 200, 50)
                hough_line_gap = st.slider("最大斷點間隙", 1, 50, 10, help="允許虛線連成直線的最大缺口")
            else:
                hough_line_t, hough_line_len, hough_line_gap = 80, 50, 10
                
            do_circles = st.checkbox("⭕ 偵測圓形", value=False)
            if do_circles:
                hough_circle_p2 = st.slider("圓心靈敏度 (越低假圓越多)", 10, 100, 30)
                hough_min_r = st.slider("最小半徑", 0, 200, 10)
                hough_max_r = st.slider("最大半徑", 10, 300, 50)
                hough_min_dist = st.slider("最小圓心距離", 1, 100, 20, help="避免同一個地方畫出好幾個重疊的圓")
            else:
                hough_circle_p2, hough_min_r, hough_max_r, hough_min_dist = 30, 10, 50, 20
                
    st.button("🗑️ 清空排程", on_click=clear_all_states)

process_order = st.session_state.process_order

# ==========================================
# 🏭 主處理邏輯與輸送帶
# ==========================================
uploaded_file = st.file_uploader("請上傳測試影像", type=["jpg", "png", "jpeg", "tif", "tiff"])

# ✨ 全局設定：把決定權交給使用者！
force_gray = st.checkbox("🖼️ 強制以單色灰階模式處理 (若是黑白老照片請打勾)", value=False)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # 統一先以 3 通道讀取
    img_origin_bgr = cv2.imdecode(file_bytes, 1)
    
    # 💡 拿掉所有複雜的數學判斷，完全聽打勾方塊的指令！
    if force_gray:
        img_origin = cv2.cvtColor(img_origin_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_origin = cv2.cvtColor(img_origin_bgr, cv2.COLOR_BGR2RGB)
        
    img_processed = img_origin.copy()

    # 📸 準備相片本，存入第一張原圖
    step_history = [("原始影像", img_origin.copy())]
    
    hist_before, hist_after = None, None

    # ==========================================
    # ⚙️ 核心主迴圈 (輸送帶運作區) - 你剛剛不小心刪掉的地方！
    # ==========================================
    for step in process_order:
        
        # 🛡️ 灰階防呆：如果演算法不支援彩色，就強制轉灰階
        color_supported_algos = [algo_eq, algo_clahe, algo_negative, algo_hough]
        if step not in color_supported_algos and len(img_processed.shape) == 3:
             img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2GRAY)

        # 🌟 亮度與對比
        if step == algo_negative:
            img_processed = apply_negative(img_processed)
        elif step in [algo_eq, algo_clahe]:
            hist_before = get_hist_data(img_processed) # 執行前拍直方圖
            if step == algo_eq:
                img_processed = apply_histogram_equalization(img_processed)
            else:
                img_processed = apply_clahe(img_processed, c_limit, g_size)
            hist_after = get_hist_data(img_processed)  # 執行後拍直方圖

        # 🖼️ 空間域平滑與銳化
        elif step == algo_mean:
            img_processed = apply_mean_filter(img_processed, mean_k)
        elif step == algo_gaussian:
            img_processed = apply_gaussian_filter(img_processed, gauss_k, gauss_s)
        elif step == algo_median:
            img_processed = apply_median_filter(img_processed, median_k)

        # 🌊 頻譜與濾波
        elif step == algo_filter:
            img_processed = apply_frequency_filter(img_processed, f_shape, f_pass, f_d0, f_n)
        elif step == algo_notch:
            img_processed = apply_notch_filter(img_processed, notch_u, notch_v, notch_d0, notch_n)

        # 📐 邊緣與特徵
        elif step == algo_sobel:
            img_processed = apply_sobel(img_processed, sobel_dir, sobel_k)
        elif step == algo_laplacian:
            img_processed = apply_laplacian_edge(img_processed, lap_k)
        elif step == algo_canny:
            img_processed = apply_canny(img_processed, canny_t1, canny_t2, canny_ap, canny_l2)
        elif step == algo_hough:
            img_processed = apply_hough_transform(
                img_processed, do_lines, hough_line_t, hough_line_len, hough_line_gap,
                do_circles, hough_circle_p2, hough_min_r, hough_max_r, hough_min_dist
            )

        # 📸 拍照存證：把每一步做完的結果放進相片本
        step_history.append((step, img_processed.copy()))


    # ==========================================
    # 🖼️ 顯示區塊 (原圖 vs 最終圖)
    # ==========================================
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始影像")
        st.image(img_origin, use_container_width=True)
    with col2:
        st.subheader("最終處理結果")
        st.image(img_processed, use_container_width=True)


    # ==========================================
    # 📊 直方圖對照區塊
    # ==========================================
    if (algo_eq in process_order or algo_clahe in process_order) and hist_before is not None:
        st.divider()
        
        current_algo = "CLAHE" if algo_clahe in process_order else "直方圖等化"
        
        title_col, toggle_col = st.columns([3, 1])
        with title_col:
            st.subheader(f"📊 {current_algo} 變化對照")
        with toggle_col:
            enable_clip = st.checkbox("✂️ 自動修剪極端值 (優化圖表比例)", value=True)
            
        hist_col1, hist_col2 = st.columns(2)
        color_map = ["#FF0000", "#00FF00", "#0000FF"] if "R" in hist_before.columns else ["#AAAAAA"]
        
        if enable_clip:
            y_ceiling = hist_before.max().max() * 1.5 
            plot_before = hist_before.clip(upper=y_ceiling)
            plot_after = hist_after.clip(upper=y_ceiling)
        else:
            plot_before = hist_before
            plot_after = hist_after

        with hist_col1:
            st.markdown(f"**執行 {current_algo} 前**")
            st.line_chart(plot_before, color=color_map)
        with hist_col2:
            st.markdown(f"**執行 {current_algo} 後**")
            st.line_chart(plot_after, color=color_map)


    # ==========================================
    # 🎬 執行過程全紀錄 (Step-by-Step)！
    # ==========================================
    if len(step_history) > 1:
        st.divider()
        
        show_history = st.checkbox("🔍 展開執行過程全紀錄 (Step-by-Step)", value=True)
        
        if show_history:
            cols_per_row = 3
            # 用迴圈動態產生排版
            for i in range(0, len(step_history), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(step_history):
                        step_name, img_snap = step_history[i + j]
                        with cols[j]:
                            st.image(img_snap, caption=f"[{i + j}] {step_name}", use_container_width=True)