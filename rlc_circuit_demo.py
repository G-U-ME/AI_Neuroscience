import tkinter as tk
from tkinter import ttk, font
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.font_manager as fm
import matplotlib as mpl
import os

# 检测操作系统并设置合适的中文字体
def setup_chinese_font():
    # 获取系统中文字体路径
    if os.name == 'nt':  # Windows系统
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]
    else:  # Linux和macOS系统
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/System/Library/Fonts/PingFang.ttc'
        ]
    
    # 尝试加载字体
    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break
    
    if font_path:
        # 添加字体文件
        font_manager = fm.FontManager()
        font_manager.addfont(font_path)
        
        # 设置matplotlib的字体
        mpl.rcParams['font.family'] = ['sans-serif']
        if 'Microsoft YaHei' in fm.get_font_names():
            mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        elif 'SimHei' in fm.get_font_names():
            mpl.rcParams['font.sans-serif'] = ['SimHei']
        else:
            # 使用刚刚加载的字体
            custom_font = fm.FontProperties(fname=font_path)
            mpl.rcParams['font.sans-serif'] = [custom_font.get_name()]
        
        mpl.rcParams['axes.unicode_minus'] = False
        return True
    return False

# 初始化字体设置
if not setup_chinese_font():
    print("警告：无法找到合适的中文字体，显示可能会有问题")

class RLCCircuitDemo(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        
        # 设置窗口标题
        self.master.title("RLC电路仿真")
        
        # 初始化标志和动画状态
        self.initialized = False
        self.is_animating = False
        self.animation_index = 0
        self.voltage_scale = 1.5  # 电压显示范围的缩放因子
        
        # 创建变量
        self.R_var = tk.DoubleVar(value=100.0)  # 电阻（欧姆）
        self.L_var = tk.DoubleVar(value=0.1)    # 电感（亨利）
        self.C_var = tk.DoubleVar(value=1.0)    # 电容（微法）
        self.V0_var = tk.DoubleVar(value=10.0)  # 电压源（伏特）
        self.f_var = tk.DoubleVar(value=50.0)   # 频率（赫兹）
        
        # 创建界面
        self.create_widgets()
        self.pack(fill=tk.BOTH, expand=True)
        
        # 初始化matplotlib
        if self.init_matplotlib():
            self.initialized = True
            # 延迟100ms后进行初始绘图
            self.master.after(100, self.initial_plot)
        
        # 绑定窗口大小改变事件
        self.master.bind('<Configure>', self.on_resize)
        
    def initial_plot(self):
        """初始绘图"""
        if self.initialized and hasattr(self, 'ax1') and hasattr(self, 'ax2'):
            self.update_plot()
        else:
            # 如果还没准备好，再等500ms
            self.master.after(500, self.initial_plot)
    
    def on_resize(self, event):
        """处理窗口大小改变事件"""
        if hasattr(self, 'ax1') and hasattr(self, 'ax2'):
            # 重新设置子图位置
            self.ax1.set_position([0.1, 0.1, 0.35, 0.8])
            self.ax2.set_position([0.55, 0.1, 0.35, 0.8])
            
            # 如果存在colorbar，重新设置其位置
            if len(self.fig.axes) > 2:  # 有colorbar
                self.fig.axes[2].set_position([0.92, 0.1, 0.02, 0.3])
            
            self.canvas.draw()
    
    def init_matplotlib(self):
        """初始化matplotlib图形"""
        try:
            # 创建画布并禁用自动调整
            self.fig = Figure(figsize=(12, 6), constrained_layout=False)
            
            # 创建子图并固定位置和大小
            self.ax1 = self.fig.add_subplot(121)  # 波形图
            self.ax2 = self.fig.add_subplot(122)  # 电路图
            
            # 固定子图位置
            self.ax1.set_position([0.1, 0.1, 0.35, 0.8])
            self.ax2.set_position([0.55, 0.1, 0.35, 0.8])
            
            # 固定电路图的显示范围
            self.ax2.set_xlim(0, 13)
            self.ax2.set_ylim(1, 7)
            self.ax2.set_aspect('equal')
            self.ax2.axis('off')
            
            # 创建matplotlib画布
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 绑定变量更新事件
            for var in [self.R_var, self.L_var, self.C_var, self.V0_var, self.f_var]:
                var.trace_add('write', self.schedule_update)
            
            return True
            
        except Exception as e:
            print(f"Error in init_matplotlib: {e}")
            return False
    
    def schedule_update(self, *args):
        """调度更新，避免过于频繁的更新"""
        if hasattr(self, '_update_job'):
            self.master.after_cancel(self._update_job)
        self._update_job = self.master.after(100, self.update_plot)
    
    def update_plot(self, *args):
        """更新图表显示"""
        if not self.initialized:
            print("Warning: Not fully initialized yet")
            return
            
        if not hasattr(self, 'ax1') or not hasattr(self, 'ax2'):
            print("Warning: Plot not initialized yet")
            return
            
        try:
            # 保存当前位置
            ax1_pos = self.ax1.get_position()
            ax2_pos = self.ax2.get_position()
            
            # 清除当前图形
            self.ax1.clear()
            self.ax2.clear()
            
            # 获取当前参数值
            V0 = self.V0_var.get()
            R = self.R_var.get()
            L = self.L_var.get()
            C = self.C_var.get() * 1e-6  # 转换为法拉
            f = self.f_var.get()
            
            # 计算RLC电路响应
            t, v, i = self.calculate_response()
            
            # 绘制电压和电流曲线
            self.ax1.plot(t, v, 'b-', label='电压 (V)', linewidth=2)
            self.ax1.plot(t, i*R, 'r-', label='电流 × R (V)', linewidth=2)
            
            # 设置波形图属性
            self.ax1.set_xlabel('时间 (s)', fontsize=10)
            self.ax1.set_ylabel('电压 (V)', fontsize=10)
            self.ax1.set_title('RLC电路响应', fontsize=12)
            self.ax1.grid(True)
            
            # 设置固定的刻度字体大小
            self.ax1.tick_params(axis='both', labelsize=10)
            
            # 调整图例位置和样式
            legend = self.ax1.legend(
                loc='upper right',
                bbox_to_anchor=(0.95, 0.95),
                fontsize=10,
                frameon=True,
                framealpha=0.8,
                edgecolor='gray',
                fancybox=True,
                shadow=True
            )
            
            # 重置电路图
            self.ax2.set_xlim(0, 13)
            self.ax2.set_ylim(1, 7)
            self.ax2.set_aspect('equal')
            self.ax2.axis('off')
            
            # 绘制电路图
            self.draw_circuit(v[0])
            
            # 恢复子图位置
            self.ax1.set_position(ax1_pos)
            self.ax2.set_position(ax2_pos)
            
            # 计算电路特性并更新信息标签
            self.update_info_label(R, L, C, f)
            
            # 更新画布
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error in update_plot: {e}")
    
    def update_info_label(self, R, L, C, f):
        """更新信息标签"""
        try:
            # 计算电路特性
            Z = np.sqrt(R**2 + (2*np.pi*f*L - 1/(2*np.pi*f*C))**2)  # 阻抗
            Z_L = 2*np.pi*f*L  # 电感阻抗
            Z_C = 1/(2*np.pi*f*C)  # 电容阻抗
            resonant_f = 1/(2*np.pi*np.sqrt(L*C))  # 谐振频率
            damping = R/(2*np.sqrt(L/C))  # 阻尼系数
            
            # 确定振荡类型
            if abs(f - resonant_f) < 0.1:
                oscillation_type = "谐振"
            elif abs(damping - 1.0) < 0.01:
                oscillation_type = "临界阻尼"
            elif damping > 1:
                oscillation_type = "过阻尼"
            else:
                oscillation_type = "欠阻尼"
            
            # 更新信息标签
            info_text = (
                f"阻抗: {Z:.2f}Ω\n"
                f"电感阻抗: {Z_L:.2f}Ω\n"
                f"电容阻抗: {Z_C:.2f}Ω\n"
                f"谐振频率: {resonant_f:.2f}Hz\n"
                f"阻尼系数: {damping:.2f}\n"
                f"振荡类型: {oscillation_type}"
            )
            if hasattr(self, 'info_label'):
                self.info_label.config(text=info_text)
        except Exception as e:
            print(f"Error updating info label: {e}")
    
    def create_variables(self):
        """创建变量"""
        self.R_var = tk.DoubleVar(value=100.0)  # 电阻（欧姆）
        self.L_var = tk.DoubleVar(value=0.1)    # 电感（亨利）
        self.C_var = tk.DoubleVar(value=1.0)    # 电容（微法）
        self.V0_var = tk.DoubleVar(value=10.0)  # 电压源（伏特）
        self.f_var = tk.DoubleVar(value=50.0)   # 频率（赫兹）
    
    def create_widgets(self):
        """创建GUI组件"""
        # 创建主框架
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧控制面板
        self.control_frame = ttk.LabelFrame(self.main_frame, text="参数控制")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 添加滑动条
        self.create_slider(self.control_frame, "电阻 (Ω)", self.R_var, 1, 100, 10)
        self.create_slider(self.control_frame, "电感 (H)", self.L_var, 0.001, 1, 0.1)
        self.create_slider(self.control_frame, "电容 (μF)", self.C_var, 0.1, 100, 1, scale=1e-6)
        self.create_slider(self.control_frame, "电压 (V)", self.V0_var, 1, 1000, 10)
        self.create_slider(self.control_frame, "频率 (Hz)", self.f_var, 1, 10000, 1000)
        
        # 创建动画控制按钮
        self.animate_button = ttk.Button(self.control_frame, text="开始动画", command=self.toggle_animation)
        self.animate_button.pack(pady=10)
        
        # 创建信息标签框架
        self.info_frame = ttk.LabelFrame(self.control_frame, text="电路参数")
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建信息标签
        self.info_label = ttk.Label(self.info_frame, text="计算中...", justify=tk.LEFT)
        self.info_label.pack(padx=5, pady=5)
        
        # 创建右侧图形面板
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_slider(self, parent, label, variable, min_value, max_value, default_value, scale=1):
        """创建滑动条"""
        slider_frame = ttk.Frame(parent)
        slider_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(slider_frame, text=label).pack(side=tk.LEFT)
        
        slider = ttk.Scale(
            slider_frame,
            from_=min_value,
            to=max_value,
            variable=variable,
            orient="horizontal",
            command=self.schedule_update
        )
        slider.set(default_value)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def calculate_response(self):
        """计算RLC电路响应"""
        R = self.R_var.get()
        L = self.L_var.get()
        C = self.C_var.get() * 1e-6  # 转换为法拉
        V0 = self.V0_var.get()
        f = self.f_var.get()
        
        # 计算特征参数
        omega0 = 1 / np.sqrt(L * C)  # 固有角频率
        alpha = R / (2 * L)  # 衰减系数
        
        # 判断振荡类型
        if alpha < omega0:  # 欠阻尼
            omega = np.sqrt(omega0**2 - alpha**2)  # 实际振荡角频率
            # 计算至少显示3个完整周期所需的时间
            T = 2 * np.pi / omega  # 一个周期的时间
            t_max = 5 * T  # 显示5个周期
            
            # 根据频率自适应调整时间步长
            num_points = 1000  # 基础点数
            if omega > 1000:
                num_points = int(omega * 2)  # 高频时增加采样点
            
            t = np.linspace(0, t_max, num_points)
            v = V0 * np.exp(-alpha * t) * np.cos(omega * t)
            i = C * V0 * np.exp(-alpha * t) * (omega * np.cos(omega * t) + alpha * np.sin(omega * t))
            
        elif alpha == omega0:  # 临界阻尼
            t_max = 10 / alpha  # 显示到电压衰减到接近0
            t = np.linspace(0, t_max, 1000)
            v = V0 * (1 + alpha * t) * np.exp(-alpha * t)
            i = C * V0 * alpha**2 * t * np.exp(-alpha * t)
            
        else:  # 过阻尼
            omega1 = -alpha + np.sqrt(alpha**2 - omega0**2)
            omega2 = -alpha - np.sqrt(alpha**2 - omega0**2)
            t_max = 10 / min(abs(omega1), abs(omega2))
            t = np.linspace(0, t_max, 1000)
            v = V0 * (np.exp(omega1 * t) - np.exp(omega2 * t)) / (omega1 - omega2)
            i = C * V0 * (omega1 * np.exp(omega1 * t) - omega2 * np.exp(omega2 * t)) / (omega1 - omega2)
        
        return t, v, i

    def draw_circuit(self, voltage):
        """绘制详细的电路图"""
        try:
            # 清除电路图但保持固定范围和属性
            self.ax2.clear()
            self.ax2.set_xlim(0, 13)
            self.ax2.set_ylim(1, 7)
            self.ax2.set_aspect('equal')
            self.ax2.axis('off')
            
            # 获取当前电路参数
            R = self.R_var.get()
            L = self.L_var.get()
            C = self.C_var.get() * 1e-6  # 转换为法拉
            V0 = self.V0_var.get()
            
            # 计算电流和各元件电压
            i = voltage / R  # 电流
            v_R = i * R  # 电阻上的电压
            v_L = L * np.array(voltage / R)  # 电感上的电压
            v_C = voltage - v_R - v_L  # 电容上的电压
            
            # 创建颜色映射
            cmap = plt.cm.RdYlBu
            max_voltage = V0 * self.voltage_scale  # 使用缩放后的电压范围
            norm = plt.Normalize(vmin=-max_voltage, vmax=max_voltage)
            
            # 定义导线宽度
            line_width = 4  # 导线总宽度
            edge_width = 1  # 边框宽度
            inner_width = line_width - 2 * edge_width  # 内部填充宽度
            
            # 定义电路元件的位置
            circuit_points = {
                'source_center': (2, 4),
                'source_right': (3, 4),
                'R_start': (4, 4),
                'R_end': (6, 4),
                'L_start': (7, 4),
                'L_end': (9, 4),
                'C_start': (10, 4),
                'C_end': (11, 4),
                'wire_right': (12, 4),
                'wire_bottom_right': (12, 2),
                'wire_bottom_left': (2, 2),
                'wire_bottom_up': (2, 3.5)
            }
            
            # 绘制导线函数
            def draw_wire_path(points, voltage_start, voltage_end):
                path_points = self.create_rounded_path(points)
                if len(path_points) < 2:
                    return
                
                segments = np.concatenate([path_points[:-1, None], path_points[1:, None]], axis=1)
                voltages = np.linspace(voltage_start, voltage_end, len(segments))
                
                # 三层绘制：黑边框、白底、彩色
                lc_edge = LineCollection(segments, colors='black', linewidth=line_width,
                                      capstyle='round', joinstyle='round')
                self.ax2.add_collection(lc_edge)
                
                lc_bg = LineCollection(segments, colors='white', linewidth=line_width-edge_width,
                                    capstyle='round', joinstyle='round')
                self.ax2.add_collection(lc_bg)
                
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=inner_width,
                                 capstyle='round', joinstyle='round')
                lc.set_array(voltages)
                self.ax2.add_collection(lc)
            
            # 绘制水平导线
            horizontal_wire_segments = [
                ([circuit_points['source_right'], circuit_points['R_start']], voltage, voltage),
                ([circuit_points['R_end'], circuit_points['L_start']], voltage-v_R, voltage-v_R),
                ([circuit_points['L_end'], circuit_points['C_start']], voltage-v_R-v_L, voltage-v_R-v_L),
                ([circuit_points['C_end'], circuit_points['wire_right']], 0, 0)
            ]
            for points, v_start, v_end in horizontal_wire_segments:
                draw_wire_path(np.array(points), v_start, v_end)
            
            # 绘制回路导线
            loop_points = np.array([
                circuit_points['C_end'],
                circuit_points['wire_right'],
                circuit_points['wire_bottom_right'],
                circuit_points['wire_bottom_left'],
                circuit_points['wire_bottom_up'],
                circuit_points['source_center']
            ])
            draw_wire_path(loop_points, 0, 0)
            
            # 绘制电阻
            R_x = circuit_points['R_start'][0]
            R_y = circuit_points['R_start'][1]
            R_length = circuit_points['R_end'][0] - circuit_points['R_start'][0]
            R_segments = 6
            segment_width = R_length / R_segments
            R_height = 0.3
            
            x_points = []
            y_points = []
            for i in range(R_segments + 1):
                x_points.append(R_x + i * segment_width)
                y_points.append(R_y + (R_height if i % 2 == 0 else -R_height))
            
            points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            voltages = np.linspace(voltage, voltage-v_R, len(segments))
            
            # 电阻三层绘制
            lc_edge = LineCollection(segments, colors='black', linewidth=2)
            self.ax2.add_collection(lc_edge)
            
            lc_bg = LineCollection(segments, colors='white', linewidth=1.5)
            self.ax2.add_collection(lc_bg)
            
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
            lc.set_array(voltages)
            self.ax2.add_collection(lc)
            
            # 绘制电感
            L_x = circuit_points['L_start'][0]
            L_y = circuit_points['L_start'][1]
            L_length = circuit_points['L_end'][0] - circuit_points['L_start'][0]
            turns = 5
            points_per_turn = 20
            t = np.linspace(0, turns*2*np.pi, turns*points_per_turn)
            x = L_x + np.linspace(0, L_length, len(t))
            y = L_y + 0.2*np.sin(t)
            
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            voltages = np.linspace(voltage-v_R, voltage-v_R-v_L, len(segments))
            
            # 电感三层绘制
            lc_edge = LineCollection(segments, colors='black', linewidth=2)
            self.ax2.add_collection(lc_edge)
            
            lc_bg = LineCollection(segments, colors='white', linewidth=1.5)
            self.ax2.add_collection(lc_bg)
            
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
            lc.set_array(voltages)
            self.ax2.add_collection(lc)
            
            # 绘制电容
            C_x1 = circuit_points['C_start'][0]
            C_x2 = circuit_points['C_end'][0]
            C_y = circuit_points['C_start'][1]
            plate_height = 0.3
            
            # 左板
            self.ax2.plot([C_x1, C_x1], [C_y-plate_height, C_y+plate_height], 'k-', linewidth=2)
            self.ax2.plot([C_x1, C_x1], [C_y-plate_height, C_y+plate_height], 'w-', linewidth=1.5)
            self.ax2.plot([C_x1, C_x1], [C_y-plate_height, C_y+plate_height], 
                         color=cmap(norm(voltage-v_R-v_L)), linewidth=1)
            
            # 右板
            self.ax2.plot([C_x2, C_x2], [C_y-plate_height, C_y+plate_height], 'k-', linewidth=2)
            self.ax2.plot([C_x2, C_x2], [C_y-plate_height, C_y+plate_height], 'w-', linewidth=1.5)
            self.ax2.plot([C_x2, C_x2], [C_y-plate_height, C_y+plate_height], 
                         color=cmap(norm(0)), linewidth=1)
            
            # 绘制电压源
            source_radius = 0.3
            circle = plt.Circle(circuit_points['source_center'], source_radius, fill=False, color='k', linewidth=1)
            self.ax2.add_artist(circle)
            
            # 添加正负极标识
            source_x, source_y = circuit_points['source_center']
            self.ax2.plot([source_x - 0.15, source_x + 0.15], [source_y - 0.1, source_y - 0.1], 'k-', linewidth=1)
            self.ax2.plot([source_x - 0.15, source_x + 0.15], [source_y + 0.1, source_y + 0.1], 'k-', linewidth=1)
            self.ax2.plot([source_x, source_x], [source_y, source_y + 0.2], 'k-', linewidth=1)
            
            # 添加元件标签
            font_props = fm.FontProperties(family='Microsoft YaHei' if 'Microsoft YaHei' in fm.get_font_names() else 'SimHei')
            
            # 电源电压标签
            self.ax2.text(1.5, 4, f'V₀={V0:.4g}V', 
                         ha='right', va='center', 
                         fontproperties=font_props, fontsize=9)
            
            # 电阻标签
            R_center_x = (circuit_points['R_start'][0] + circuit_points['R_end'][0]) / 2
            self.ax2.text(R_center_x, 4.5, f'R={R:.4g}Ω', 
                         ha='center', va='bottom', 
                         fontproperties=font_props, fontsize=9)
            
            # 电感标签
            L_center_x = (circuit_points['L_start'][0] + circuit_points['L_end'][0]) / 2
            self.ax2.text(L_center_x, 4.5, f'L={L:.4g}H', 
                         ha='center', va='bottom', 
                         fontproperties=font_props, fontsize=9)
            
            # 电容标签
            C_center_x = (circuit_points['C_start'][0] + circuit_points['C_end'][0]) / 2
            self.ax2.text(C_center_x, 4.5, f'C={C*1e6:.4g}μF', 
                         ha='center', va='bottom', 
                         fontproperties=font_props, fontsize=9)
            
            # 频率标签
            self.ax2.text(1.5, 3.5, f'f={self.f_var.get():.4g}Hz', 
                         ha='right', va='center', 
                         fontproperties=font_props, fontsize=9)
            
            # 添加固定位置的颜色条
            if len(self.fig.axes) > 2:  # 如果已经存在colorbar，先移除
                self.fig.delaxes(self.fig.axes[2])
            
            # 创建固定位置的颜色条
            cax = self.fig.add_axes([0.92, 0.1, 0.02, 0.3])
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
                orientation='vertical'
            )
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label('电压 (V)', fontproperties=font_props, fontsize=10)
            
        except Exception as e:
            print(f"Error in draw_circuit: {e}")
            
    def toggle_animation(self):
        """切换动画状态"""
        if not self.initialized:
            return
            
        self.is_animating = not self.is_animating
        self.animate_button.config(text="停止动画" if self.is_animating else "开始动画")
        
        if self.is_animating:
            self.animation_index = 0  # 重置动画索引
            self.update_animation()  # 开始动画循环
    
    def update_animation(self):
        """更新动画"""
        if self.is_animating:
            try:
                # 计算RLC电路响应
                t, v, i = self.calculate_response()
                
                # 更新电压和电流曲线
                self.ax1.clear()
                
                # 绘制完整的电压和电流曲线（虚线）
                self.ax1.plot(t, v, 'b--', alpha=0.3, label='电压 (V)')
                self.ax1.plot(t, i*self.R_var.get(), 'r--', alpha=0.3, label='电流 × R (V)')
                
                # 确保动画索引在有效范围内
                if self.animation_index >= len(t):
                    self.animation_index = 0
                
                # 绘制到当前时刻的实线
                if self.animation_index > 0:
                    self.ax1.plot(t[:self.animation_index], v[:self.animation_index], 'b-', linewidth=2)
                    self.ax1.plot(t[:self.animation_index], i[:self.animation_index]*self.R_var.get(), 'r-', linewidth=2)
                    
                    # 添加当前点的标记
                    current_t = t[self.animation_index-1]
                    current_v = v[self.animation_index-1]
                    current_i = i[self.animation_index-1]
                    self.ax1.plot(current_t, current_v, 'bo', markersize=8)
                    self.ax1.plot(current_t, current_i*self.R_var.get(), 'ro', markersize=8)
                
                # 设置波形图属性
                self.ax1.set_xlabel('时间 (s)', fontsize=10)
                self.ax1.set_ylabel('电压 (V)', fontsize=10)
                self.ax1.set_title('RLC电路响应', fontsize=12)
                self.ax1.grid(True)
                self.ax1.legend(loc='upper right')
                
                # 更新电路图
                self.ax2.clear()
                self.ax2.set_xlim(0, 13)
                self.ax2.set_ylim(1, 7)
                self.ax2.set_aspect('equal')
                self.ax2.axis('off')
                
                # 使用当前时刻的电压值更新电路图
                current_v = v[self.animation_index-1] if self.animation_index > 0 else v[0]
                self.draw_circuit(current_v)
                
                # 更新画布
                self.canvas.draw()
                
                # 更新动画索引
                self.animation_index += 5  # 增加步长使动画更快
                
                # 继续动画
                if self.is_animating:
                    self.master.after(50, self.update_animation)  # 降低帧率使动画更流畅
                
            except Exception as e:
                print(f"Error in update_animation: {e}")
                self.is_animating = False
                self.animate_button.config(text="开始动画")
    
    def create_rounded_path(self, points, radius=0.3):
        """创建带有圆角的路径"""
        if len(points) < 2:
            return np.array([])
        
        def get_bezier_points(p0, p1, p2, num_points=20):
            """创建贝塞尔曲线点"""
            t = np.linspace(0, 1, num_points)
            x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
            y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
            return np.column_stack([x, y])
        
        path_points = []
        for i in range(len(points)-1):
            if i == 0:
                # 第一段，从起点到第一个圆角
                start = points[i]
                if len(points) > 2:
                    v1 = points[i+1] - points[i]
                    v1_norm = v1 / np.linalg.norm(v1)
                    end = points[i+1] - v1_norm * radius
                else:
                    end = points[i+1]
                path_points.extend(np.linspace(start, end, 20))
            
            if i < len(points)-2:
                # 中间段，创建圆角
                p0 = points[i+1] - (points[i+1] - points[i]) / np.linalg.norm(points[i+1] - points[i]) * radius
                p2 = points[i+1] - (points[i+1] - points[i+2]) / np.linalg.norm(points[i+1] - points[i+2]) * radius
                curve_points = get_bezier_points(p0, points[i+1], p2)
                path_points.extend(curve_points)
            
            if i == len(points)-2:
                # 最后一段，从最后一个圆角到终点
                if len(points) > 2:
                    v2 = points[-1] - points[-2]
                    v2_norm = v2 / np.linalg.norm(v2)
                    start = points[-2] + v2_norm * radius
                    end = points[-1]
                    path_points.extend(np.linspace(start, end, 20))
        
        return np.array(path_points)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = RLCCircuitDemo(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print("发生错误：")
        print(traceback.format_exc())
        input("按回车键退出...")  # 保持窗口打开以查看错误信息
