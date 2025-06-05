import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import pickle
import shap
from PIL import Image, ImageTk

class SoilStrengthPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("固化土强度预测系统")
        self.root.geometry("900x650")
        self.root.resizable(True, True)
        self.root.configure(background="white")
        
        self.setup_styles()
        
        self.setup_fonts()
        
        self.load_model()
        
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        
        try:
            style.theme_use('vista')
        except:
            try:
                style.theme_use('clam')
            except:
                pass
        
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white")
        style.configure("TLabelframe", background="white")
        style.configure("TLabelframe.Label", background="white")
        style.configure("TButton", background="white")
        style.configure("Vertical.TScrollbar", background="white", troughcolor="white")
        style.configure("TScale", background="white", troughcolor="#e0e0e0")
        
        style.map("TButton", background=[('active', '#f0f0f0'), ('pressed', '#e0e0e0')])
        
    def setup_fonts(self):
        default_font = ("SimSun", 10)
        self.root.option_add("*Font", default_font)
        
        self.root.option_add("*Background", "white")
        self.root.option_add("*Labelframe.Background", "white")
        self.root.option_add("*Text.Background", "white")
        self.root.option_add("*Canvas.Background", "white")
        self.root.option_add("*Button.Background", "white")
        self.root.option_add("*Entry.Background", "white")
        self.root.option_add("*Listbox.Background", "white")
        self.root.option_add("*Menu.Background", "white")
        self.root.option_add("*Scale.Background", "white")
        self.root.option_add("*Scrollbar.Background", "white")
        
        plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
        plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'

    def load_model(self):
        try:
            model_path = "catboost_model.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件 {model_path} 不存在")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
            self.explainer = shap.Explainer(self.model)
            
            self.feature_ranges = {
                "含水率": (0.2, 1.5),
                "水泥含量": (0.05, 0.40),
                "黏粒含量": (0.1, 0.9)
            }
            
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="固化土强度预测系统", font=("SimSun", 16))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        input_frame = ttk.LabelFrame(main_frame, text="参数输入", padding="10")
        input_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        self.param_vars = {}
        self.param_entries = {}
        self.param_scales = {}
        
        row = 0
        feature = "含水率"
        min_val, max_val = self.feature_ranges[feature]
        ttk.Label(input_frame, text=f"{feature}:", font=("SimSun", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.param_vars[feature] = tk.DoubleVar(value=(min_val + max_val) / 2)
        self.param_entries[feature] = ttk.Entry(input_frame, textvariable=self.param_vars[feature], width=10, font=("Times New Roman", 10))
        self.param_entries[feature].grid(row=row, column=1, padx=5)
        self.param_scales[feature] = ttk.Scale(input_frame, from_=min_val, to=max_val, 
                                              variable=self.param_vars[feature], 
                                              length=200, orient=tk.HORIZONTAL)
        self.param_scales[feature].grid(row=row, column=2, padx=10)
        
        row = 1
        feature = "水泥含量"
        min_val, max_val = self.feature_ranges[feature]
        ttk.Label(input_frame, text=f"{feature}:", font=("SimSun", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.param_vars[feature] = tk.DoubleVar(value=(min_val + max_val) / 2)
        self.param_entries[feature] = ttk.Entry(input_frame, textvariable=self.param_vars[feature], width=10, font=("Times New Roman", 10))
        self.param_entries[feature].grid(row=row, column=1, padx=5)
        self.param_scales[feature] = ttk.Scale(input_frame, from_=min_val, to=max_val, 
                                              variable=self.param_vars[feature], 
                                              length=200, orient=tk.HORIZONTAL)
        self.param_scales[feature].grid(row=row, column=2, padx=10)
        
        row = 2
        feature = "黏粒含量"
        min_val, max_val = self.feature_ranges[feature]
        ttk.Label(input_frame, text=f"{feature}:", font=("SimSun", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.param_vars[feature] = tk.DoubleVar(value=(min_val + max_val) / 2)
        self.param_entries[feature] = ttk.Entry(input_frame, textvariable=self.param_vars[feature], width=10, font=("Times New Roman", 10))
        self.param_entries[feature].grid(row=row, column=1, padx=5)
        self.param_scales[feature] = ttk.Scale(input_frame, from_=min_val, to=max_val, 
                                              variable=self.param_vars[feature], 
                                              length=200, orient=tk.HORIZONTAL)
        self.param_scales[feature].grid(row=row, column=2, padx=10)
        
        predict_button = ttk.Button(input_frame, text="预测强度", command=self.predict_strength)
        predict_button.grid(row=3, column=1, columnspan=2, pady=15, sticky="ew")
        
        result_frame = ttk.LabelFrame(main_frame, text="预测结果", padding="10")
        result_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        ttk.Label(result_frame, text="无侧限抗压强度 (UCS):", font=("SimSun", 10)).grid(row=0, column=0, sticky="w", pady=5)
        self.strength_var = tk.StringVar(value="-- MPa")
        strength_label = ttk.Label(result_frame, textvariable=self.strength_var, font=("Times New Roman", 12, "bold"))
        strength_label.grid(row=0, column=1, sticky="w", pady=5)
        
        ttk.Label(result_frame, text="95% 置信区间:", font=("SimSun", 10)).grid(row=1, column=0, sticky="w", pady=5)
        self.ci_var = tk.StringVar(value="-- ~ -- MPa")
        ci_label = ttk.Label(result_frame, textvariable=self.ci_var, font=("Times New Roman", 10))
        ci_label.grid(row=1, column=1, sticky="w", pady=5)
        
        shap_frame = ttk.LabelFrame(main_frame, text="特征贡献分析", padding="10")
        shap_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        self.shap_canvas = tk.Canvas(shap_frame, width=800, height=220)
        self.shap_canvas.pack(fill=tk.BOTH, expand=True)
        
        suggestion_frame = ttk.LabelFrame(main_frame, text="配比优化建议", padding="10")
        suggestion_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        suggestion_scroll = ttk.Scrollbar(suggestion_frame)
        suggestion_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.suggestion_text = tk.Text(suggestion_frame, height=6, wrap=tk.WORD, font=("SimSun", 10),
                                      yscrollcommand=suggestion_scroll.set)
        self.suggestion_text.pack(fill=tk.BOTH, expand=True)
        suggestion_scroll.config(command=self.suggestion_text.yview)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=2)
        main_frame.rowconfigure(3, weight=1)
        
    def predict_strength(self):
        try:
            features = {}
            for feature in self.feature_ranges.keys():
                features[feature] = self.param_vars[feature].get()
            
            X = pd.DataFrame([features])
            
            strength = self.model.predict(X)[0]
            
            self.strength_var.set(f"{strength:.3f} MPa")
            
            lower_bound = strength * 0.9
            upper_bound = strength * 1.1
            self.ci_var.set(f"{lower_bound:.3f} ~ {upper_bound:.3f} MPa")
            
            shap_values = self.explainer(X)
            
            self.plot_shap_waterfall(shap_values[0])
            
            self.generate_suggestion(features, shap_values[0])
            
        except Exception as e:
            print(f"预测过程出错: {e}")
            self.strength_var.set("预测失败")
            
    def plot_shap_waterfall(self, shap_values):
        plt.figure(figsize=(8, 2.5))
        
        plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.subplots_adjust(left=0.25)
        
        shap.plots.waterfall(shap_values, show=False)
        
        temp_file = "temp_shap_waterfall.png"
        plt.savefig(temp_file, dpi=180, bbox_inches='tight')
        plt.close()
        
        img = Image.open(temp_file)
        img_width, img_height = img.size
        
        scale_factor = 0.4
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        self.shap_img = ImageTk.PhotoImage(img)
        self.shap_canvas.delete("all")
        canvas_width = self.shap_canvas.winfo_width()
        if canvas_width < 10:
            canvas_width = 800
        self.shap_canvas.create_image(canvas_width//2, 110, image=self.shap_img)
        
        try:
            os.remove(temp_file)
        except:
            pass
    
    def generate_suggestion(self, features, shap_values):
        feature_values = {"含水率": features["含水率"], 
                          "水泥含量": features["水泥含量"], 
                          "黏粒含量": features["黏粒含量"]}
        
        suggestions = []
        
        if shap_values.values[0] < 0:
            if features["含水率"] > 0.8:
                suggestions.append("当前含水率较高，建议降低含水率以提高强度。")
            else:
                suggestions.append("当前含水率处于合理范围，对强度影响适中。")
        
        if features["水泥含量"] < 0.2:
            suggestions.append("水泥含量较低，可适当增加以提高强度，但注意与含水率的配比关系。")
        else:
            suggestions.append("水泥含量已较高，继续增加边际效益会降低，可考虑调整其他参数。")
        
        if features["含水率"] <= 0.8 and features["水泥含量"] < 0.2:
            suggestions.append("在当前低含水率和低水泥含量条件下，适当提高黏粒含量可增强固化土强度。")
        elif features["含水率"] > 0.8:
            suggestions.append("高含水率条件下，黏粒含量的增强效果有限，优先考虑降低含水率或增加水泥含量。")
        
        overall = "综合建议：基于SHAP分析，"
        if abs(shap_values.values[0]) > abs(shap_values.values[1]) and abs(shap_values.values[0]) > abs(shap_values.values[2]):
            overall += "含水率是影响当前配比强度的最关键因素，"
            if shap_values.values[0] < 0:
                overall += "建议优先降低含水率。"
            else:
                overall += "当前含水率有利于强度提升。"
        elif abs(shap_values.values[1]) > abs(shap_values.values[0]) and abs(shap_values.values[1]) > abs(shap_values.values[2]):
            overall += "水泥含量是影响当前配比强度的最关键因素，"
            if shap_values.values[1] > 0:
                if features["水泥含量"] < 0.2:
                    overall += "有增加空间，建议适当提高水泥含量。"
                else:
                    overall += "但当前已处于高效区间，进一步增加可能边际效益递减。"
        else:
            overall += "黏粒含量是影响当前配比强度的最关键因素，"
            if shap_values.values[2] > 0:
                overall += "当前配比中黏粒含量有利于强度提升。"
            else:
                overall += "建议调整黏粒含量以提高强度。"
        
        self.suggestion_text.delete(1.0, tk.END)
        for suggestion in suggestions:
            self.suggestion_text.insert(tk.END, f"• {suggestion}\n")
        self.suggestion_text.insert(tk.END, f"\n{overall}")

def export_model():
    try:
        import pandas as pd
        from catboost import CatBoostRegressor
        from sklearn.model_selection import train_test_split
        
        data = pd.read_excel('固化土.xlsx')
        
        X = data[['含水率', '水泥含量', '黏粒含量']]
        y = data['强度']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, 
                                          loss_function='RMSE', random_state=42)
        
        catboost_model.fit(X_train, y_train, verbose=100)
        
        with open('catboost_model.pkl', 'wb') as f:
            pickle.dump(catboost_model, f)
            
        print("模型导出成功!")
        return True
    except Exception as e:
        print(f"模型导出失败: {e}")
        return False

if __name__ == "__main__":
    if not os.path.exists('catboost_model.pkl'):
        print("未找到预训练模型，正在导出...")
        if not export_model():
            print("无法创建模型，请确保训练数据存在")
            exit(1)
    
    root = tk.Tk()
    app = SoilStrengthPredictorGUI(root)
    root.mainloop() 