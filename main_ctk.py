import customtkinter as ctk
import tkinter as tk
import tkinter.filedialog as fd
from PIL import Image
import cv2
import numpy as np
import os

from cv_algorithms import enhance, frequency, spatial, feature, morphology, object_analysis, geometry

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# ==========================================
# 🌟 演算法精華字典 (游標懸浮提示內容)
# ==========================================
ALGO_TIPS = {
    "負片轉換 (反轉)": "將影像黑白/色彩顛倒，適用於凸顯暗部特徵。",
    "直方圖等化": "自動拉伸整體對比度，讓過暗或過亮的影像變清晰。",
    "CLAHE": "限制對比度的局部等化，去霧、處理光線不均神器。",
    "手動二值化": "依據自訂門檻值，將影像強制轉換為非黑即白。",
    "Otsu 自動二值化": "自動計算出最佳門檻值，完美切分前景與背景。",
    "Adaptive 自適應二值化": "依據局部區域動態計算門檻，克服強烈光影不均。",
    
    "均值濾波 (平滑)": "用平均值模糊影像，最基礎的平滑化。",
    "高斯濾波 (平滑)": "符合常態分配的模糊，保留較多邊緣細節。",
    "中值濾波 (去雜訊)": "取中位數，去除胡椒鹽雜訊 (點狀雜點) 的超級剋星。",
    
    "理想濾波器 (Ideal)": "頻率域濾波：像一堵牆直接切斷高低頻，會產生明顯漣漪效應(Ringing)。",
    "高斯濾波器 (Gaussian)": "頻率域濾波：平滑過渡，沒有漣漪效應，最常使用。",
    "巴特沃斯 (Butterworth)": "頻率域濾波：介於理想與高斯之間，可透過階數(n)調整銳利度。",
    "陷波濾波 (去週期波)": "專門消除影像中網格狀、條紋狀的週期性干擾雜訊。",
    
    "Sobel 邊緣": "利用一階微分尋找影像的水平或垂直邊界。",
    "拉普拉斯 (邊緣強化)": "利用二階微分，對細節極度敏感，常用於銳利化。",
    "Canny 邊緣偵測": "業界最推的邊緣檢測！具備自動去噪與雙門檻機制，邊緣僅1像素寬。",
    "霍氏直線偵測": "在混亂邊緣中精準找出「直線」方程式，適用於車道線、幾何測量。",
    "霍氏圓形偵測": "利用梯度方向精準定位「圓心與半徑」，適用於硬幣、細胞、藥丸檢測。",
    
    "膨脹": "找最大值。讓亮部變胖，填補物體內部微小破洞或加粗細線。",
    "侵蝕": "找最小值。讓亮部變瘦，消除微小亮點雜訊或分離沾黏物體。",
    "Opening": "【先侵蝕後膨脹】斷開沾黏物體、消除背景細小亮點 (平滑化)。",
    "Closing": "【先膨脹後侵蝕】填補物體內部小洞、消除物體上的暗色雜點 (平滑化)。",
    "邊界抽取": "【原圖 - 侵蝕】從物體內部削掉一層皮，精準抽出 1 像素寬的極細邊界。",
    "形態學梯度": "【膨脹 - 侵蝕】抽出較粗的雙軌邊緣，完美凸顯灰階漸層的輪廓。",
    "空洞填補": "【條件式膨脹】像倒油漆一樣，精準填滿封閉牆壁內的巨大破洞。",
    "Top-Hat": "【原圖 - Opening】抓出比周圍「亮」的微小細節，無懼打光不均！",
    "Black-Hat": "【Closing - 原圖】抓出比周圍「暗」的微小細節 (如金屬表面黑斑)。",
    "Hit-or-Miss": "【嚴格形狀雷達】依據自訂 SE 的 1 與 -1，精準尋找特定幾何圖案。",
    "交替連續濾波 (ASF)": "從小到大依序執行開閉運算。無損深度去噪，完美保留直角與輪廓！",

    # 🌟 新增的物件分析提示
    "連通區域標記 (Connected Components)": "將相連的亮色區塊視為同一個物件，並塗上隨機顏色。用於區分與計算畫面中有幾個獨立物體。",
    "尋找輪廓 (Find Contours)": "精準描繪出獨立物件的邊界框線。EXTERNAL 抓最外圍，TREE 可抓出包含內部破洞的完整父子階層關係。"
}

# ==========================================
# 懸浮提示框 (ToolTip) 類別
# ==========================================
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tw = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True) 
        self.tw.wm_geometry(f"+{x}+{y}")
        self.tw.attributes('-topmost', True) 
        
        label = tk.Label(self.tw, text=self.text, justify="left",
                         background="#2980B9", foreground="white", 
                         relief="solid", borderwidth=1,
                         font=("微軟正黑體", 11, "bold"), padx=10, pady=6)
        label.pack()

    def leave(self, event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None

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
        channels, colors = [0, 1, 2], [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
    
    for i, col in zip(channels, colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, h - 20, cv2.NORM_MINMAX)
        for x in range(1, 256):
            pt1 = (int(40 + (x-1) * (w/256)), h - int(hist[x-1][0]))
            pt2 = (int(40 + x * (w/256)), h - int(hist[x][0]))
            cv2.line(canvas, pt1, pt2, col, 2, cv2.LINE_AA)
    return canvas

# ==========================================
# 自訂 SE 設計面板 (獨立彈出視窗)
# ==========================================
class SEDesignerWindow(ctk.CTkToplevel):
    def __init__(self, master, current_matrix, callback):
        super().__init__(master)
        self.title("⚙️ 自訂結構元素 (SE) 設計")
        self.geometry("380x480")
        self.attributes('-topmost', True) 
        
        self.callback = callback
        self.se_size = current_matrix.shape[0]
        self.se_matrix = np.copy(current_matrix)
        self.buttons = []
        self.build_ui()

    def build_ui(self):
        size_frame = ctk.CTkFrame(self)
        size_frame.pack(pady=15, padx=10, fill="x")
        ctk.CTkLabel(size_frame, text="設定 SE 尺寸:").pack(side="left", padx=10)
        self.size_var = ctk.StringVar(value=f"{self.se_size}x{self.se_size}")
        ctk.CTkOptionMenu(size_frame, variable=self.size_var, values=["3x3", "5x5"], 
                          command=self.change_size).pack(side="left", padx=5)

        self.grid_frame = ctk.CTkFrame(self)
        self.grid_frame.pack(pady=10)
        self.draw_grid()

        ctk.CTkLabel(self, text="點擊網格切換狀態:\n🟩 1 (前景) | 🟥 -1 (背景) | ⬜ 0 (忽略)", 
                     text_color="#F1C40F", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=10)
        ctk.CTkButton(btn_frame, text="重設十字", width=90, command=self.set_cross).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="重設實心", width=90, command=self.set_solid).pack(side="left", padx=5)

    def draw_grid(self):
        for widget in self.grid_frame.winfo_children(): widget.destroy()
        self.buttons.clear()
        for r in range(self.se_size):
            for c in range(self.se_size):
                btn = ctk.CTkButton(self.grid_frame, width=45, height=45, text="", font=ctk.CTkFont(weight="bold", size=16))
                btn.grid(row=r, column=c, padx=3, pady=3)
                btn.configure(command=lambda row=r, col=c: self.toggle_cell(row, col))
                self.buttons.append((r, c, btn))
        self.update_buttons()

    def toggle_cell(self, r, c):
        val = self.se_matrix[r, c]
        self.se_matrix[r, c] = -1 if val == 1 else (0 if val == -1 else 1)
        self.update_buttons()
        self.callback(self.se_matrix) 

    def update_buttons(self):
        for r, c, btn in self.buttons:
            val = self.se_matrix[r, c]
            if val == 1: btn.configure(text="1", fg_color="#2ECC71", hover_color="#27AE60", text_color="black")
            elif val == -1: btn.configure(text="-1", fg_color="#E74C3C", hover_color="#C0392B", text_color="white")
            else: btn.configure(text="0", fg_color="#95A5A6", hover_color="#7F8C8D", text_color="black")

    def change_size(self, new_size):
        self.se_size = int(new_size.split("x")[0])
        self.se_matrix = np.ones((self.se_size, self.se_size), dtype=int)
        self.draw_grid()
        self.callback(self.se_matrix)

    def set_cross(self):
        self.se_matrix.fill(0)
        mid = self.se_size // 2
        self.se_matrix[mid, :] = 1
        self.se_matrix[:, mid] = 1
        self.update_buttons()
        self.callback(self.se_matrix)

    def set_solid(self):
        self.se_matrix.fill(1)
        self.update_buttons()
        self.callback(self.se_matrix)


class PipelineApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("影像處理平台")
        self.geometry("1650x950")
        self.configure(fg_color="#1A1A1A")

        self.cv_img_rgb = None 
        self.process_order = []
        self.ui_vars = {}        
        self.param_sections = {}
        self.step_history = [] 
        
        self.current_se_matrix = np.ones((3, 3), dtype=int) 
        self.se_window = None

        self.current_before_rgb = None
        self.current_after_rgb = None
        self.swipe_ratio = 0.5 

        self.grid_columnconfigure(0, weight=0, minsize=240) 
        self.grid_columnconfigure(1, weight=0, minsize=350) 
        self.grid_columnconfigure(2, weight=1)              
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.toggle_sidebar = ctk.CTkScrollableFrame(self, width=240, fg_color="#242424", label_text="⚙️ 演算法開關", label_font=ctk.CTkFont(size=18, weight="bold"))
        self.toggle_sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew") 

        ctk.CTkButton(self.toggle_sidebar, text="📂 載入測試影像", command=self.open_image, height=35).pack(fill="x", padx=10, pady=(5, 5))
        ctk.CTkButton(self.toggle_sidebar, text="🗑️ 清空全部", command=self.clear_all, fg_color="#A93226", height=35).pack(fill="x", padx=10, pady=5)
        self.force_gray_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.toggle_sidebar, text="強制灰階模式", variable=self.force_gray_var, command=self.run_pipeline).pack(pady=10, padx=10, anchor="w")

        tool_frame = ctk.CTkFrame(self.toggle_sidebar, fg_color="transparent")
        tool_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(tool_frame, text="📌 框選四點轉正", command=self.tool_perspective, fg_color="#8E44AD", hover_color="#9B59B6", height=30).pack(fill="x", pady=(0, 5))
        ctk.CTkButton(tool_frame, text="🔗 SIFT 影像對齊", command=self.tool_sift_align, fg_color="#2980B9", hover_color="#3498DB", height=30).pack(fill="x")

        self.param_sidebar = ctk.CTkScrollableFrame(self, width=350, fg_color="#2D2D2D", label_text="🎛️ 參數調整", label_font=ctk.CTkFont(size=18, weight="bold"), label_text_color="#5DADE2")
        self.param_sidebar.grid(row=0, column=1, rowspan=2, sticky="nsew") 
        
        self.pipeline_info = ctk.CTkFrame(self.param_sidebar, fg_color="#3D3D3D", corner_radius=10)
        self.pipeline_info.pack(fill="x", padx=10, pady=10)
        self.pipeline_text_label = ctk.CTkLabel(self.pipeline_info, text="( 空白 )", wraplength=330, text_color="#00FFCC")
        self.pipeline_text_label.pack(pady=10, padx=10)

        self.top_right_container = ctk.CTkFrame(self, fg_color="#1A1A1A")
        self.top_right_container.grid(row=0, column=2, sticky="nsew", padx=20, pady=(20, 10))

        self.view_mode_var = ctk.StringVar(value="並排顯示")
        self.view_mode_selector = ctk.CTkSegmentedButton(
            self.top_right_container, values=["並排顯示", "滑動對比 (Swipe)"],
            variable=self.view_mode_var, command=self.switch_view_mode, width=300
        )
        self.view_mode_selector.pack(pady=(0, 10))

        self.frame_sbs = ctk.CTkFrame(self.top_right_container, fg_color="transparent")
        self.frame_sbs.pack(fill="both", expand=True)
        self.frame_sbs.grid_columnconfigure((0, 1), weight=1)
        self.frame_sbs.grid_rowconfigure(0, weight=1)
        self.lbl_before = ctk.CTkLabel(self.frame_sbs, text="原始影像", text_color="#AAAAAA")
        self.lbl_before.grid(row=0, column=0, padx=10)
        self.lbl_after = ctk.CTkLabel(self.frame_sbs, text="最終處理結果", text_color="#AAAAAA")
        self.lbl_after.grid(row=0, column=1, padx=10)

        self.frame_swipe = ctk.CTkFrame(self.top_right_container, fg_color="transparent")
        self.lbl_swipe = ctk.CTkLabel(self.frame_swipe, text="載入圖片以啟動滑動對比", cursor="sb_h_double_arrow")
        self.lbl_swipe.pack(pady=10)
        self.lbl_swipe.bind("<B1-Motion>", self.on_swipe_drag)
        self.lbl_swipe.bind("<Button-1>", self.on_swipe_drag)

        self.frame_bottom_area = ctk.CTkScrollableFrame(self, corner_radius=0, fg_color="#1A1A1A")
        self.frame_bottom_area.grid(row=1, column=2, sticky="nsew", padx=20, pady=(10, 20))

        self.hist_container = ctk.CTkFrame(self.frame_bottom_area, fg_color="#222222", corner_radius=15)
        ctk.CTkLabel(self.hist_container, text="📊 R G B 直方圖對照", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.hist_plot_frame = ctk.CTkFrame(self.hist_container, fg_color="transparent")
        self.hist_plot_frame.pack(fill="x", padx=10, pady=10)
        self.lbl_hist_before = ctk.CTkLabel(self.hist_plot_frame, text="處理前", compound="top")
        self.lbl_hist_before.pack(side="left", expand=True)
        self.lbl_hist_after = ctk.CTkLabel(self.hist_plot_frame, text="處理後", compound="top")
        self.lbl_hist_after.pack(side="right", expand=True)

        self.gallery_container = ctk.CTkScrollableFrame(self.frame_bottom_area, height=280, orientation="horizontal", label_text="🎬 執行過程全紀錄 (點擊可放大預覽)")
        self.gallery_container.pack(fill="x", pady=20)

        self.build_comprehensive_ui()

    def build_comprehensive_ui(self):
        config = [
            ("亮度對比", [
                "負片轉換 (反轉)", 
                "直方圖等化", 
                ("CLAHE", [("C_limit", "限制", 1.0, 10.0, 2.0, 0.1), ("C_grid", "網格", 2, 16, 8, 1)]),
                ("手動二值化", [("thresh_val", "門檻值", 0, 255, 127, 1)]),
                "Otsu 自動二值化",
                ("Adaptive 自適應二值化", [("adp_block", "區塊大小", 3, 99, 11, 2), ("adp_c", "常數 C", -20, 20, 2, 1)])
            ]),
            ("平滑濾波", [("均值濾波 (平滑)", [("M_k", "大小", 3, 31, 3, 2)]), ("高斯濾波 (平滑)", [("G_k", "大小", 3, 31, 5, 2), ("G_s", "Sigma", 0, 10, 0, 0.5)]), ("中值濾波 (去雜訊)", [("Med_k", "大小", 3, 31, 5, 2)])]),
            ("頻率域 (分開)", [("理想濾波器 (Ideal)", [("F_i_p", "類型", ["low_pass", "high_pass"]), ("F_i_d", "半徑", 5, 300, 50, 1)]), ("高斯濾波器 (Gaussian)", [("F_g_p", "類型", ["low_pass", "high_pass"]), ("F_g_d", "半徑", 5, 300, 50, 1)]), ("巴特沃斯 (Butterworth)", [("F_b_p", "類型", ["low_pass", "high_pass"]), ("F_b_d", "半徑", 5, 300, 50, 1), ("F_b_n", "階數", 1, 10, 2, 1)]), ("陷波濾波 (去週期波)", [("N_u", "X偏移", -150, 150, 7, 1), ("N_v", "Y偏移", -150, 150, 0, 1), ("N_d0", "消除半徑", 1, 50, 3, 1), ("N_har", "倍頻數", 1, 10, 5, 1)])]),
            ("邊緣特徵", [("Sobel 邊緣", [("S_dir", "方向", ["both", "x_only", "y_only"]), ("S_k", "大小", 1, 7, 3, 2)]), ("拉普拉斯 (邊緣強化)", [("L_k", "大小", 1, 7, 3, 2)]), ("Canny 邊緣偵測", [("Can_t1", "T1", 0, 255, 100, 1), ("Can_t2", "T2", 0, 255, 200, 1)]), ("霍氏直線偵測", [("H_l_t", "靈敏度", 10, 200, 80, 1), ("H_l_len", "最短長度", 10, 200, 50, 1), ("H_l_gap", "最大間隙", 1, 50, 10, 1)]), ("霍氏圓形偵測", [("H_c_p2", "圓心靈敏", 10, 100, 30, 1), ("H_minR", "最小半徑", 0, 200, 10, 1), ("H_maxR", "最大半徑", 10, 300, 50, 1), ("H_minD", "最小圓心距離", 1, 100, 20, 1)])]),
            ("形態學處理", [
                ("膨脹", [("Iter_Dil", "迭代次數", 1, 10, 1, 1)]),
                ("侵蝕", [("Iter_Ero", "迭代次數", 1, 10, 1, 1)]),
                ("Opening", [("Iter_Opn", "迭代次數", 1, 10, 1, 1)]),
                ("Closing", [("Iter_Cls", "迭代次數", 1, 10, 1, 1)]),
                ("邊界抽取", [("Iter_Bnd", "迭代次數", 1, 10, 1, 1)]),
                ("形態學梯度", [("Iter_Grad", "迭代次數", 1, 10, 1, 1)]),
                "空洞填補",       
                ("Top-Hat", [("Iter_TH", "迭代次數", 1, 10, 1, 1), ("SE_Size_TH", "結構元素大小", 3, 99, 45, 2)]),
                ("Black-Hat", [("Iter_BH", "迭代次數", 1, 10, 1, 1), ("SE_Size_BH", "結構元素大小", 3, 99, 45, 2)]),
                "Hit-or-Miss",  
                "交替連續濾波 (ASF)" 
            ]),
            # 🌟 新增：輪廓與物件分析
            ("輪廓與物件分析", [
                ("連通區域標記 (Connected Components)", [("CC_conn", "連通性", ["8-way", "4-way"])]),
                ("尋找輪廓 (Find Contours)", [
                    ("Cont_mode", "模式", ["EXTERNAL", "LIST", "TREE"]),
                    ("Cont_meth", "近似法", ["SIMPLE", "NONE"])
                ])
            ])
        ]
        
        for cat, items in config:
            ctk.CTkLabel(self.toggle_sidebar, text=cat, text_color="#5DADE2", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5), padx=20, anchor="w")
            for item in items:
                if isinstance(item, str): self.add_checkbox(item)
                else: self.add_checkbox(item[0], item[1])

        se_btn = ctk.CTkButton(self.toggle_sidebar, text="⚙️ 開啟自訂 SE 設計面板", 
                               fg_color="#F39C12", hover_color="#D68910", text_color="black", 
                               font=ctk.CTkFont(weight="bold"), command=self.open_se_designer)
        se_btn.pack(pady=20, padx=20, fill="x")

    def open_se_designer(self):
        if self.se_window is None or not self.se_window.winfo_exists():
            self.se_window = SEDesignerWindow(self, self.current_se_matrix, self.update_se_matrix)
        else:
            self.se_window.focus()

    def update_se_matrix(self, new_matrix):
        self.current_se_matrix = new_matrix
        self.run_pipeline()

    def add_checkbox(self, name, params=None):
        var = ctk.BooleanVar(value=False)
        self.ui_vars[name] = var
        cb = ctk.CTkCheckBox(self.toggle_sidebar, text=name, variable=var, command=lambda: self.toggle_section(name))
        cb.pack(anchor="w", padx=30, pady=3)
        
        if name in ALGO_TIPS:
            ToolTip(cb, ALGO_TIPS[name])
            
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

    def switch_view_mode(self, mode):
        if mode == "並排顯示":
            self.frame_swipe.pack_forget()
            self.frame_sbs.pack(fill="both", expand=True)
        else:
            self.frame_sbs.pack_forget()
            self.frame_swipe.pack(fill="both", expand=True)
            self.update_swipe_view()

    def on_swipe_drag(self, event):
        if self.current_before_rgb is None or self.current_after_rgb is None: return
        widget_width = self.lbl_swipe.winfo_width()
        if widget_width == 0: return
        ratio = event.x / widget_width
        self.swipe_ratio = max(0.0, min(1.0, ratio))
        self.update_swipe_view()

    def update_swipe_view(self):
        if self.current_before_rgb is None or self.current_after_rgb is None: return
        disp_w, disp_h = 800, 500
        img_b = cv2.resize(self.current_before_rgb, (disp_w, disp_h))
        img_a = cv2.resize(self.current_after_rgb, (disp_w, disp_h))

        if len(img_b.shape) == 2: img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)
        if len(img_a.shape) == 2: img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)

        split_px = int(disp_w * self.swipe_ratio)
        composite = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        composite[:, :split_px] = img_b[:, :split_px]
        composite[:, split_px:] = img_a[:, split_px:]

        cv2.line(composite, (split_px, 0), (split_px, disp_h), (255, 255, 255), 3)
        cv2.circle(composite, (split_px, disp_h // 2), 15, (255, 255, 255), -1)
        cv2.circle(composite, (split_px, disp_h // 2), 10, (100, 100, 100), -1)

        pil = Image.fromarray(composite)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(disp_w, disp_h))
        self.lbl_swipe.configure(image=ctk_img, text="")
        self.lbl_swipe.image = ctk_img

    def run_pipeline(self):
        if self.cv_img_rgb is None: return
        
        img = cv2.cvtColor(self.cv_img_rgb, cv2.COLOR_RGB2GRAY) if self.force_gray_var.get() else self.cv_img_rgb.copy()
        
        self.current_before_rgb = img.copy()
        self.render(self.lbl_before, img)
        self.step_history = [("原始影像", img.copy())]
        
        v = {k: (var.get() if hasattr(var, 'get') else var) for k, var in self.ui_vars.items()}
        show_hist = False
        hb, ha = None, None

        for step in self.process_order:
            grayscale_algos = ["Sobel 邊緣", "拉普拉斯 (邊緣強化)", "Canny 邊緣偵測", "理想濾波器 (Ideal)", "高斯濾波器 (Gaussian)", "巴特沃斯 (Butterworth)", "陷波濾波 (去週期波)", 
                               "膨脹", "侵蝕", "Opening", "Closing", "Hit-or-Miss", "邊界抽取", "形態學梯度", "空洞填補", "Top-Hat", "Black-Hat", "交替連續濾波 (ASF)"]
            if step in grayscale_algos and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            try:
                if step in ["直方圖等化", "CLAHE", "負片轉換 (反轉)"]:
                    show_hist = True
                    hb = draw_rgb_histogram(img)
                    if step == "直方圖等化": img = enhance.apply_histogram_equalization(img)
                    elif step == "CLAHE": img = enhance.apply_clahe(img, v['C_limit'], int(v['C_grid']))
                    else: img = enhance.apply_negative(img)
                    ha = draw_rgb_histogram(img)
                elif step == "均值濾波 (平滑)": img = spatial.apply_mean_filter(img, int(v['M_k']))
                elif step == "高斯濾波 (平滑)": img = spatial.apply_gaussian_filter(img, int(v['G_k']), v['G_s'])
                elif step == "中值濾波 (去雜訊)": img = spatial.apply_median_filter(img, int(v['Med_k']))
                elif step == "拉普拉斯 (邊緣強化)": img = feature.apply_laplacian_edge(img, int(v['L_k']))
                elif step == "理想濾波器 (Ideal)": img = frequency.apply_frequency_filter(img, "ideal", v['F_i_p'].split(" ")[0], v['F_i_d'])
                elif step == "高斯濾波器 (Gaussian)": img = frequency.apply_frequency_filter(img, "gaussian", v['F_g_p'].split(" ")[0], v['F_g_d'])
                elif step == "巴特沃斯 (Butterworth)": img = frequency.apply_frequency_filter(img, "butterworth", v['F_b_p'].split(" ")[0], v['F_b_d'], int(v['F_b_n']))
                elif step == "手動二值化": img = enhance.apply_binary_threshold(img, threshold_value=int(v['thresh_val']))
                elif step == "Otsu 自動二值化": img = enhance.apply_otsu_threshold(img)
                elif step == "Adaptive 自適應二值化":
                        b_size = int(v['adp_block'])
                        c_val = int(v['adp_c'])
                        img = enhance.apply_adaptive_threshold(img, block_size=b_size, C_value=c_val)
                elif step == "陷波濾波 (去週期波)": img = frequency.apply_notch_filter(img, int(v['N_u']), int(v['N_v']), v['N_d0'], int(v['N_har']))
                elif step == "Sobel 邊緣": img = feature.apply_sobel(img, v['S_dir'].split(" ")[0], int(v['S_k']))
                elif step == "Canny 邊緣偵測": img = feature.apply_canny(img, int(v['Can_t1']), int(v['Can_t2']))
                elif step == "霍氏直線偵測": img = feature.apply_hough_transform(img, True, int(v['H_l_t']), int(v['H_l_len']), int(v['H_l_gap']), False, 30, 0, 0, 20)
                elif step == "霍氏圓形偵測": img = feature.apply_hough_transform(img, False, 80, 50, 10, True, int(v['H_c_p2']), int(v['H_minR']), int(v['H_maxR']), int(v['H_minD']))
                # 1. 專屬於 Top-Hat 與 Black-Hat (使用動態巨大的圓形 SE)
                elif step in ["Top-Hat", "Black-Hat"]:
                    iters = int(v[f"Iter_{'TH' if step=='Top-Hat' else 'BH'}"])
                    se_size = int(v[f"SE_Size_{'TH' if step=='Top-Hat' else 'BH'}"])
                    
                    # 動態生成巨大的圓形結構元素 (對應投影片的 Disk)
                    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
                    
                    img = morphology.apply_advanced_morphology(img, step, big_kernel, iters)

                # 2. 其他一般的形態學 (繼續使用原本的自訂小 SE 面板)
                elif step in ["膨脹", "侵蝕", "Opening", "Closing", "Hit-or-Miss", "邊界抽取", "形態學梯度", "空洞填補", "交替連續濾波 (ASF)"]: 
                    iter_dict = {
                        "膨脹": "Iter_Dil", "侵蝕": "Iter_Ero", 
                        "Opening": "Iter_Opn", "Closing": "Iter_Cls",
                        "邊界抽取": "Iter_Bnd", "形態學梯度": "Iter_Grad"
                    }
                    iters = int(v[iter_dict[step]]) if step in iter_dict else 1
                    img = morphology.apply_advanced_morphology(img, step, self.current_se_matrix, iters)
                
                # 🌟 新增：輪廓與物件分析派發
                elif step == "連通區域標記 (Connected Components)":
                    conn = 8 if v['CC_conn'] == "8-way" else 4
                    img = object_analysis.apply_connected_components(img, conn)
                elif step == "尋找輪廓 (Find Contours)":
                    img = object_analysis.apply_find_contours(img, v['Cont_mode'], v['Cont_meth'])

            except Exception as e: print(f"Error at {step}: {e}")
            self.step_history.append((step, img.copy()))

        self.current_after_rgb = img.copy()
        
        self.render(self.lbl_after, img)
        if self.view_mode_var.get() == "滑動對比 (Swipe)":
            self.update_swipe_view()

        if show_hist and hb is not None:
            self.hist_container.pack(fill="x", padx=10, pady=10)
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
        disp = img.copy() if len(img.shape)==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(disp)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=size)
        label.configure(image=ctk_img, text="")
        label.image = ctk_img

    def open_image(self):
        path = fd.askopenfilename(filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            bgr_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.cv_img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            self.run_pipeline()

    def clear_all(self):
        for f in self.param_sections.values(): f.pack_forget()
        for v in self.ui_vars.values():
            if isinstance(v, ctk.BooleanVar): v.set(False)
        self.process_order.clear()
        self.pipeline_text_label.configure(text="( 空白 )")
        self.run_pipeline()

    def tool_perspective(self):
        """執行四點透視轉正工具"""
        if self.cv_img_rgb is None:
            return
        
        # 呼叫 geometry，並將回傳的轉正影像「覆蓋」掉原始影像
        warped_img = geometry.interactive_perspective_transform(self.cv_img_rgb)
        
        # 確保有正常回傳圖片 (沒有按 X 取消)
        if warped_img is not None: 
            self.cv_img_rgb = warped_img
            self.run_pipeline() # 重新跑一次流水線，主畫面就會立刻更新

    def tool_sift_align(self):
        """執行 SIFT 影像對齊，並無縫支援並排與滑動雙模式"""
        if self.cv_img_rgb is None: 
            return
            
        # 1. 取得標準目標圖 (左) 與 扭曲後的待測圖 (右)
        target_img, warped_img = geometry.align_images_sift(self.cv_img_rgb)
        
        if target_img is not None:
            from PIL import Image
            
            # --- 【UI 顯示更新：並排模式】 ---
            pil_tgt = Image.fromarray(target_img)
            pil_wrp = Image.fromarray(warped_img)
            ctk_tgt = ctk.CTkImage(light_image=pil_tgt, size=(400, 400))
            ctk_wrp = ctk.CTkImage(light_image=pil_wrp, size=(400, 400))

            self.lbl_before.configure(image=ctk_tgt)
            self.lbl_after.configure(image=ctk_wrp)
            
            # --- 🚀 【底層資料更新：打通滑動對比功能】 🚀 ---
            # 把這兩張圖存進底層變數，餵給滑動功能！
            self.current_before_rgb = target_img.copy()
            self.current_after_rgb = warped_img.copy()
            
            # 如果目前 UI 正好停在「滑動對比」分頁，就立刻觸發畫面更新
            if self.view_mode_var.get() == "滑動對比 (Swipe)":
                self.update_swipe_view()
            # -----------------------------------------------
            
            # 更新系統底圖
            self.cv_img_rgb = warped_img
            
            print("✅ [系統提示] SIFT 對齊已成功！(完美支援滑動無鬼影對比)")
            
        else:
            print("⚠️ 對齊失敗或已取消，UI 不做更動")
if __name__ == "__main__":
    PipelineApp().mainloop()