import customtkinter as ctk
import tkinter.filedialog as fd
from PIL import Image
import cv2
import numpy as np
import os

# 引入你的演算法模組 (請確保裡面是你原本習慣吃 RGB 的版本！)
from cv_algorithms.enhance import apply_histogram_equalization, apply_clahe, apply_negative
from cv_algorithms.frequency import apply_frequency_filter, apply_notch_filter
from cv_algorithms.spatial import apply_mean_filter, apply_gaussian_filter, apply_median_filter
from cv_algorithms.feature import apply_sobel, apply_laplacian_edge, apply_canny, apply_hough_transform

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# ==========================================
# 📊 直方圖繪製工具 (配合 RGB 輸入修正顏色)
# ==========================================
def draw_rgb_histogram(img):
    h, w = 300, 400
    canvas = np.ones((h + 40, w + 40, 3), dtype=np.uint8) * 30 
    
    cv2.line(canvas, (40, 10), (40, h), (200, 200, 200), 2) 
    cv2.line(canvas, (40, h), (w + 20, h), (200, 200, 200), 2) 
    cv2.putText(canvas, "Count", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, "0", (30, h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, "255", (w + 10, h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, "Intensity", (w // 2, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if len(img.shape) == 2:
        channels, colors = [0], [(255, 255, 255)]
    else:
        # 🌟 既然全面改用 RGB，這裡的顏色對應也要改成 RGB (紅、綠、藍)
        channels, colors = [0, 1, 2], [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
    
    for i, col in zip(channels, colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, h - 20, cv2.NORM_MINMAX)
        for x in range(1, 256):
            pt1 = (int(40 + (x-1) * (w/256)), h - int(hist[x-1][0]))
            pt2 = (int(40 + x * (w/256)), h - int(hist[x][0]))
            cv2.line(canvas, pt1, pt2, col, 2, cv2.LINE_AA)
    return canvas

class PipelineApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("影像處理平台")
        self.geometry("1650x950")
        self.configure(fg_color="#1A1A1A")

        self.cv_img_rgb = None # 🌟 變數名稱改為 RGB，時時刻刻提醒我們它是 RGB
        self.process_order = []
        self.ui_vars = {}        
        self.param_sections = {}
        self.step_history = [] 

        self.grid_columnconfigure(0, weight=0) 
        self.grid_columnconfigure(1, weight=0) 
        self.grid_columnconfigure(2, weight=1) 
        self.grid_rowconfigure(0, weight=1)

        # --- 左中右面板設定 (維持不變) ---
        self.toggle_sidebar = ctk.CTkScrollableFrame(self, width=240, fg_color="#242424", label_text="⚙️ 演算法開關")
        self.toggle_sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkButton(self.toggle_sidebar, text="📂 載入圖片", command=self.open_image).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(self.toggle_sidebar, text="🗑️ 清空全部", command=self.clear_all, fg_color="#A93226").pack(fill="x", padx=10, pady=5)
        self.force_gray_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.toggle_sidebar, text="強制灰階模式", variable=self.force_gray_var, command=self.run_pipeline).pack(pady=10, padx=10, anchor="w")

        self.param_sidebar = ctk.CTkScrollableFrame(self, width=350, fg_color="#2D2D2D", label_text="🎛️ 參數與排程")
        self.param_sidebar.grid(row=0, column=1, sticky="nsew")
        self.pipeline_info = ctk.CTkFrame(self.param_sidebar, fg_color="#3D3D3D", corner_radius=10)
        self.pipeline_info.pack(fill="x", padx=10, pady=10)
        self.pipeline_text_label = ctk.CTkLabel(self.pipeline_info, text="( 空白 )", wraplength=280, text_color="#00FFCC")
        self.pipeline_text_label.pack(pady=10, padx=10)

        self.main_area = ctk.CTkScrollableFrame(self, fg_color="#1A1A1A")
        self.main_area.grid(row=0, column=2, sticky="nsew")
        self.compare_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.compare_frame.pack(fill="x", pady=10)
        self.compare_frame.grid_columnconfigure((0, 1), weight=1)
        self.lbl_before = ctk.CTkLabel(self.compare_frame, text="原始影像")
        self.lbl_before.grid(row=0, column=0)
        self.lbl_after = ctk.CTkLabel(self.compare_frame, text="最終處理結果")
        self.lbl_after.grid(row=0, column=1)

        self.hist_container = ctk.CTkFrame(self.main_area, fg_color="#222222", corner_radius=15)
        ctk.CTkLabel(self.hist_container, text="📊 R G B 直方圖對照", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.hist_plot_frame = ctk.CTkFrame(self.hist_container, fg_color="transparent")
        self.hist_plot_frame.pack(fill="x", padx=10, pady=10)
        self.lbl_hist_before = ctk.CTkLabel(self.hist_plot_frame, text="處理前", compound="top")
        self.lbl_hist_before.pack(side="left", expand=True)
        self.lbl_hist_after = ctk.CTkLabel(self.hist_plot_frame, text="處理後", compound="top")
        self.lbl_hist_after.pack(side="right", expand=True)

        self.gallery_container = ctk.CTkScrollableFrame(self.main_area, height=280, orientation="horizontal", label_text="🎬 過程紀錄 (點擊可放大)")
        self.gallery_container.pack(fill="x", padx=20, pady=20)

        self.build_comprehensive_ui()

    # --- UI 建立函數群 (維持不變) ---
    def build_comprehensive_ui(self):
        config = [
            ("亮度對比", ["負片轉換 (反轉)", "直方圖等化", ("CLAHE", [("C_limit", "限制", 1.0, 10.0, 2.0, 0.1), ("C_grid", "網格", 4, 32, 8, 4)])]),
            ("平滑濾波", [("均值濾波 (平滑)", [("M_k", "大小", 3, 31, 3, 2)]), ("高斯濾波 (平滑)", [("G_k", "大小", 3, 31, 5, 2), ("G_s", "Sigma", 0, 10, 0, 0.5)]), ("中值濾波 (去雜訊)", [("Med_k", "大小", 3, 31, 5, 2)])]),
            ("頻率域 (分開)", [("理想濾波器 (Ideal)", [("F_i_p", "類型", ["low_pass", "high_pass"]), ("F_i_d", "半徑", 5, 300, 50, 1)]), ("高斯濾波器 (Gaussian)", [("F_g_p", "類型", ["low_pass", "high_pass"]), ("F_g_d", "半徑", 5, 300, 50, 1)]), ("巴特沃斯 (Butterworth)", [("F_b_p", "類型", ["low_pass", "high_pass"]), ("F_b_d", "半徑", 5, 300, 50, 1), ("F_b_n", "階數", 1, 10, 2, 1)]), ("陷波濾波 (去週期波)", [("N_u", "X偏移", -150, 150, 7, 1), ("N_v", "Y偏移", -150, 150, 0, 1), ("N_d0", "消除半徑", 1, 50, 3, 1), ("N_har", "倍頻數", 1, 10, 5, 1)])]),
            ("邊緣特徵", [("Sobel 邊緣", [("S_dir", "方向", ["both", "x_only", "y_only"]), ("S_k", "大小", 1, 7, 3, 2)]), ("拉普拉斯 (邊緣強化)", [("L_k", "大小", 1, 7, 3, 2)]), ("Canny 邊緣偵測", [("Can_t1", "T1", 0, 255, 100, 1), ("Can_t2", "T2", 0, 255, 200, 1)]), ("霍氏直線偵測", [("H_l_t", "靈敏度", 10, 200, 80, 1), ("H_l_len", "最短長度", 10, 200, 50, 1), ("H_l_gap", "最大間隙", 1, 50, 10, 1)]), ("霍氏圓形偵測", [("H_c_p2", "圓心靈敏", 10, 100, 30, 1), ("H_minR", "最小半徑", 0, 200, 10, 1), ("H_maxR", "最大半徑", 10, 300, 50, 1), ("H_minD", "最小圓心距離", 1, 100, 20, 1)])])
        ]
        for cat, items in config:
            ctk.CTkLabel(self.toggle_sidebar, text=cat, text_color="#5DADE2", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5), padx=20, anchor="w")
            for item in items:
                if isinstance(item, str): self.add_checkbox(item)
                else: self.add_checkbox(item[0], item[1])

    def add_checkbox(self, name, params=None):
        var = ctk.BooleanVar(value=False)
        self.ui_vars[name] = var
        ctk.CTkCheckBox(self.toggle_sidebar, text=name, variable=var, command=lambda: self.toggle_section(name)).pack(anchor="w", padx=30, pady=3)
        if params:
            frame = ctk.CTkFrame(self.param_sidebar, fg_color="transparent")
            self.param_sections[name] = frame
            ctk.CTkLabel(frame, text=f"■ {name}", text_color="#F1C40F", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5), anchor="w")
            for p in params:
                if len(p) == 3: self.add_option(frame, p[0], p[2], p[1])
                else: self.add_slider(frame, p[0], p[1], p[2], p[3], p[4], p[5])

    def add_slider(self, parent, key, text, min_v, max_v, def_v, step):
        self.ui_vars[key] = ctk.DoubleVar(value=def_v)
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=5)
        header = ctk.CTkFrame(f, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(header, text=text, font=ctk.CTkFont(size=12)).pack(side="left")
        v_lbl = ctk.CTkLabel(header, text=f"{def_v:.1f}", text_color="#FF5555")
        v_lbl.pack(side="left", padx=10)
        def update_val(val):
            v_lbl.configure(text=f"{float(val):.1f}")
            self.run_pipeline()
        ctk.CTkSlider(f, from_=min_v, to=max_v, number_of_steps=int((max_v-min_v)/step), variable=self.ui_vars[key], command=update_val).pack(fill="x")

    def add_option(self, parent, key, options, lbl_text):
        self.ui_vars[key] = ctk.StringVar(value=options[0])
        ctk.CTkLabel(parent, text=lbl_text, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10)
        ctk.CTkOptionMenu(parent, values=options, variable=self.ui_vars[key], command=lambda _: self.run_pipeline()).pack(fill="x", padx=10, pady=5)

    def toggle_section(self, name):
        if self.ui_vars[name].get():
            if name in self.param_sections: self.param_sections[name].pack(fill="x", pady=5)
            if name not in self.process_order: self.process_order.append(name)
        else:
            if name in self.param_sections: self.param_sections[name].pack_forget()
            if name in self.process_order: self.process_order.remove(name)
        self.pipeline_text_label.configure(text=" ➡️ ".join(self.process_order) if self.process_order else "( 空白 )")
        self.run_pipeline()

    # ==========================================
    # 核心管線 (完全相容 RGB 流水線)
    # ==========================================
    def run_pipeline(self):
        if self.cv_img_rgb is None: return
        
        # 🌟 此時的圖片已經是 RGB 了！
        img = cv2.cvtColor(self.cv_img_rgb, cv2.COLOR_RGB2GRAY) if self.force_gray_var.get() else self.cv_img_rgb.copy()
        
        self.render(self.lbl_before, img)
        self.step_history = [("原始影像", img.copy())]
        v = {k: (var.get() if hasattr(var, 'get') else var) for k, var in self.ui_vars.items()}
        show_hist = False
        hb, ha = None, None

        for step in self.process_order:
            # ⚠️ 注意這裡：轉灰階時是用 COLOR_RGB2GRAY
            grayscale_algos = ["Sobel 邊緣", "拉普拉斯 (邊緣強化)", "Canny 邊緣偵測", "理想濾波器 (Ideal)", "高斯濾波器 (Gaussian)", "巴特沃斯 (Butterworth)", "陷波濾波 (去週期波)"]
            if step in grayscale_algos and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            try:
                if step in ["直方圖等化", "CLAHE", "負片轉換 (反轉)"]:
                    show_hist = True
                    hb = draw_rgb_histogram(img)
                    if step == "直方圖等化": img = apply_histogram_equalization(img)
                    elif step == "CLAHE": img = apply_clahe(img, v['C_limit'], int(v['C_grid']))
                    else: img = apply_negative(img)
                    ha = draw_rgb_histogram(img)
                elif step == "均值濾波 (平滑)": img = apply_mean_filter(img, int(v['M_k']))
                elif step == "高斯濾波 (平滑)": img = apply_gaussian_filter(img, int(v['G_k']), v['G_s'])
                elif step == "中值濾波 (去雜訊)": img = apply_median_filter(img, int(v['Med_k']))
                elif step == "拉普拉斯 (邊緣強化)": img = apply_laplacian_edge(img, int(v['L_k']))
                elif step == "理想濾波器 (Ideal)": img = apply_frequency_filter(img, "ideal", v['F_i_p'].split(" ")[0], v['F_i_d'])
                elif step == "高斯濾波器 (Gaussian)": img = apply_frequency_filter(img, "gaussian", v['F_g_p'].split(" ")[0], v['F_g_d'])
                elif step == "巴特沃斯 (Butterworth)": img = apply_frequency_filter(img, "butterworth", v['F_b_p'].split(" ")[0], v['F_b_d'], int(v['F_b_n']))
                elif step == "陷波濾波 (去週期波)": img = apply_notch_filter(img, int(v['N_u']), int(v['N_v']), v['N_d0'], int(v['N_har']))
                elif step == "Sobel 邊緣": img = apply_sobel(img, v['S_dir'].split(" ")[0], int(v['S_k']))
                elif step == "Canny 邊緣偵測": img = apply_canny(img, int(v['Can_t1']), int(v['Can_t2']))
                # 🌟 霍氏轉換呼叫 (你原汁原味的 Streamlit 版演算法)
                # 🌟 修正後：用 int() 把拉桿數值包起來
                elif step == "霍氏直線偵測": 
                    img = apply_hough_transform(img, True, int(v['H_l_t']), int(v['H_l_len']), int(v['H_l_gap']), False, 30, 0, 0, 20)
                elif step == "霍氏圓形偵測": 
                    img = apply_hough_transform(img, False, 80, 50, 10, True, int(v['H_c_p2']), int(v['H_minR']), int(v['H_maxR']), int(v['H_minD']))
            except Exception as e: print(f"Error: {e}")
            self.step_history.append((step, img.copy()))

        self.render(self.lbl_after, img)
        if show_hist and hb is not None:
            self.hist_container.pack(fill="x", padx=20, pady=10)
            self.render(self.lbl_hist_before, hb, size=(380, 280))
            self.render(self.lbl_hist_after, ha, size=(380, 280))
        else: self.hist_container.pack_forget()
        self.update_gallery()

    def update_gallery(self):
        for w in self.gallery_container.winfo_children(): w.destroy()
        for i, (name, snap) in enumerate(self.step_history):
            f = ctk.CTkFrame(self.gallery_container, fg_color="#222222", corner_radius=8)
            f.pack(side="left", padx=8, pady=5)
            lbl = ctk.CTkLabel(f, text="", cursor="hand2")
            lbl.pack(padx=5, pady=5)
            self.render(lbl, snap, size=(240, 180))
            lbl.bind("<Button-1>", lambda e, s=snap, n=name: self.open_magnifier(s, n))
            ctk.CTkLabel(f, text=f"Step {i}: {name}", font=ctk.CTkFont(size=11), text_color="#888888").pack()

    def open_magnifier(self, img_matrix, title):
        top = ctk.CTkToplevel(self)
        top.title(f"預覽: {title}")
        top.geometry("900x700")
        top.after(100, lambda: top.focus())
        big_lbl = ctk.CTkLabel(top, text="")
        big_lbl.pack(expand=True, fill="both", padx=20, pady=20)
        self.render(big_lbl, img_matrix, size=(850, 650))

    def render(self, label, img, size=(500, 380)):
        # 🌟 因為底層已經是 RGB 了，如果送過來的是 3 通道，就不需要再轉換！
        disp = img.copy() if len(img.shape)==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(disp)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=size)
        label.configure(image=ctk_img, text="")
        label.image = ctk_img

    def open_image(self):
        path = fd.askopenfilename(filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            bgr_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 🌟 關鍵核心：在源頭就轉成 RGB！
            self.cv_img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            self.run_pipeline()

    def clear_all(self):
        for f in self.param_sections.values(): f.pack_forget()
        for v in self.ui_vars.values():
            if isinstance(v, ctk.BooleanVar): v.set(False)
        self.process_order.clear()
        self.pipeline_text_label.configure(text="( 空白 )")
        self.run_pipeline()

if __name__ == "__main__":
    PipelineApp().mainloop()