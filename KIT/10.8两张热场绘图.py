import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch import nn
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义 Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 定义模型类
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            Swish(),
            nn.Linear(64, 64),
            Swish(),
            nn.Linear(64, 64),
            Swish(),
            nn.Linear(64, 64),
            Swish(),
            nn.Linear(64, 64),
            Swish(),
            nn.Linear(64, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# 储存热源位置
positions = torch.Tensor([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75],
                          [0.75, -0.75], [-0.5, 0.75]])
units = torch.Tensor([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25], [0.6, 0.6], [0.25, 0.25]])
x_loc, y_loc = [], []
for i in range(len(positions)):
    x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
    y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])

# 判定每个点是否在 chip 位置范围内
def is_chip(x, y, x_locs, y_locs, margin=0.001):
    is_chip = torch.zeros_like(x, dtype=torch.bool)
    for i in range(len(x_locs)):
        x_min, x_max = x_locs[i]
        y_min, y_max = y_locs[i]
        mask = (x >= (x_min - margin)) & (x <= (x_max + margin)) & (y >= (y_min - margin)) & (y <= (y_max + margin))
        is_chip = is_chip | mask
    return is_chip

# 设置字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

# 创建绘图
fig1, axes1 = plt.subplots(4, 6, figsize=(30, 20))
fig2, axes2 = plt.subplots(4, 6, figsize=(30, 20))

# 调整子图间距
fig1.subplots_adjust(hspace=0.3, wspace=0.3)
fig2.subplots_adjust(hspace=0.3, wspace=0.3)

# 遍历每个case和数据量
cases = ['case1', 'case2', 'case3']
data_points = [64, 94, 124, 154]

# 定义每个 case 对应的测试集路径
case_to_path = {
    'case1': 'D:\\桌面\\新选点\\一万个点测试集.csv',
    'case2': 'D:\\桌面\\新选点\\绝热测试集.csv',
    'case3': 'D:\\桌面\\新选点\\复杂边界测试集.csv'
}

# 初始化存储所有绝对误差的列表
all_absolute_errors1 = []
all_absolute_errors2 = []

for i, case in enumerate(cases):
    # 读取实际温度数据
    actual_data_path = case_to_path[case]
    actual_data = pd.read_csv(actual_data_path)

    # 处理重复数据
    actual_data = actual_data.groupby(['x', 'y'], as_index=False).mean()

    # 使用模型预测实际值位置的温度
    test_x_actual = torch.tensor(actual_data['x'].values, dtype=torch.float32).reshape(-1, 1).to(device)
    test_y_actual = torch.tensor(actual_data['y'].values, dtype=torch.float32).reshape(-1, 1).to(device)

    for j, points in enumerate(data_points):
        # 创建模型并加载权重
        PINN = MLP().to(device)
        PINN1 = MLP().to(device)
        model_path_PINN = f'D:\\桌面\\{case}权重\\pinn{points}.pth'
        model_path_PINN1 = f'D:\\桌面\\{case}权重\\1pinn{points}.pth'
        model_path_PINN2 = f'D:\\桌面\\{case}权重\\nn{points}.pth'

        # 加载模型权重
        PINN.load_state_dict(torch.load(model_path_PINN))
        PINN1.load_state_dict(torch.load(model_path_PINN1))
        PINN2 = MLP().to(device)
        PINN2.load_state_dict(torch.load(model_path_PINN2))

        # 使用模型预测实际值位置的温度
        with torch.no_grad():
            is_chip_data_actual = is_chip(test_x_actual, test_y_actual, x_loc, y_loc)
            T_pred_chip_actual = PINN(torch.cat([test_x_actual, test_y_actual], dim=1))
            T_pred_non_chip_actual = PINN1(torch.cat([test_x_actual, test_y_actual], dim=1))
            T_pred_actual1 = torch.where(is_chip_data_actual, T_pred_chip_actual, T_pred_non_chip_actual)
            T_pred_actual2 = PINN2(torch.cat([test_x_actual, test_y_actual], dim=1))

        # 计算绝对误差
        absolute_error1 = np.abs(T_pred_actual1.cpu().numpy().flatten() - actual_data['T'].values)
        absolute_error2 = np.abs(T_pred_actual2.cpu().numpy().flatten() - actual_data['T'].values)

        # 将所有绝对误差存储到列表中
        all_absolute_errors1.extend(absolute_error1)
        all_absolute_errors2.extend(absolute_error2)

# 获取所有绝对误差的最小值和最大值
min_error = min(min(all_absolute_errors1), min(all_absolute_errors2))
max_error = max(max(all_absolute_errors1), max(all_absolute_errors2))

# 创建统一的颜色映射
norm = Normalize(vmin=min_error, vmax=max_error)
colors = [(1, 1, 1), (0.8, 0.8, 1), (0.6, 0.6, 1), (0.4, 0.4, 1), (0.2, 0.2, 1), (0, 0, 1)]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

# 创建变淡的颜色映射
colors_lightened = [(min(c[0] + 0.03, 1), min(c[1] + 0.03, 1), min(c[2]+ 0.02 , 1)) for c in colors]
cmap_lightened = LinearSegmentedColormap.from_list("lightened_cmap", colors_lightened, N=100)

for i, case in enumerate(cases):
    # 读取实际温度数据
    actual_data_path = case_to_path[case]
    actual_data = pd.read_csv(actual_data_path)

    # 处理重复数据
    actual_data = actual_data.groupby(['x', 'y'], as_index=False).mean()

    # 使用模型预测实际值位置的温度
    test_x_actual = torch.tensor(actual_data['x'].values, dtype=torch.float32).reshape(-1, 1).to(device)
    test_y_actual = torch.tensor(actual_data['y'].values, dtype=torch.float32).reshape(-1, 1).to(device)

    for j, points in enumerate(data_points):
        # 创建模型并加载权重
        PINN = MLP().to(device)
        PINN1 = MLP().to(device)
        model_path_PINN = f'D:\\桌面\\{case}权重\\pinn{points}.pth'
        model_path_PINN1 = f'D:\\桌面\\{case}权重\\1pinn{points}.pth'
        model_path_PINN2 = f'D:\\桌面\\{case}权重\\nn{points}.pth'

        # 加载模型权重
        PINN.load_state_dict(torch.load(model_path_PINN))
        PINN1.load_state_dict(torch.load(model_path_PINN1))
        PINN2 = MLP().to(device)
        PINN2.load_state_dict(torch.load(model_path_PINN2))

        # 使用模型预测实际值位置的温度
        with torch.no_grad():
            is_chip_data_actual = is_chip(test_x_actual, test_y_actual, x_loc, y_loc)
            T_pred_chip_actual = PINN(torch.cat([test_x_actual, test_y_actual], dim=1))
            T_pred_non_chip_actual = PINN1(torch.cat([test_x_actual, test_y_actual], dim=1))
            T_pred_actual1 = torch.where(is_chip_data_actual, T_pred_chip_actual, T_pred_non_chip_actual)
            T_pred_actual2 = PINN2(torch.cat([test_x_actual, test_y_actual], dim=1))

        # 计算绝对误差
        absolute_error1 = np.abs(T_pred_actual1.cpu().numpy().flatten() - actual_data['T'].values)
        absolute_error2 = np.abs(T_pred_actual2.cpu().numpy().flatten() - actual_data['T'].values)

        # 生成密集的网格点用于预测图绘制
        x = np.linspace(-1.25, 1.25, 1000)
        y = np.linspace(-1.25, 1.25, 1000)
        x_grid, y_grid = np.meshgrid(x, y)
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        test_x = torch.tensor(x_flat, dtype=torch.float32).reshape(-1, 1).to(device)
        test_y = torch.tensor(y_flat, dtype=torch.float32).reshape(-1, 1).to(device)

        # 进行预测
        PINN.eval()
        PINN1.eval()
        PINN2.eval()
        with torch.no_grad():
            is_chip_data = is_chip(test_x, test_y, x_loc, y_loc)
            T_pred_chip = PINN(torch.cat([test_x, test_y], dim=1))
            T_pred_non_chip = PINN1(torch.cat([test_x, test_y], dim=1))
            T_pred1 = torch.where(is_chip_data, T_pred_chip, T_pred_non_chip)
            T_pred2 = PINN2(torch.cat([test_x, test_y], dim=1))

        # 转换预测结果为 numpy 数组
        T_pred1 = T_pred1.cpu().numpy().reshape(x_grid.shape)
        T_pred2 = T_pred2.cpu().numpy().reshape(x_grid.shape)

        # 绘制第一个程序的预测温度图
        ax_temp1 = axes1[j, i * 2]
        divider1 = make_axes_locatable(ax_temp1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.2)
        c1 = ax_temp1.contourf(x_grid, y_grid, T_pred1, levels=100, cmap='jet')
        cbar1 = fig1.colorbar(c1, cax=cax1)
        cbar1.ax.set_ylabel('')
        cbar1.set_ticks(cbar1.get_ticks()[::2])
        cbar1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_temp1.set_title(f'{case} {points}', fontsize=24)
        ax_temp1.set_xticks([])
        ax_temp1.set_yticks([])
        ax_temp1.set_aspect('equal')

        # 绘制第一个程序的绝对误差热场图（颜色变淡）
        ax_diff1 = axes1[j, i * 2 + 1]
        divider2 = make_axes_locatable(ax_diff1)
        cax2 = divider2.append_axes("right", size="5%", pad=0.2)
        c2 = ax_diff1.tricontourf(actual_data['x'], actual_data['y'], absolute_error1, levels=100, cmap=cmap_lightened, norm=norm)
        cbar2 = fig1.colorbar(c2, cax=cax2)
        cbar2.ax.set_ylabel('')
        cbar2.set_ticks(cbar2.get_ticks()[::2])
        cbar2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_diff1.set_title(f'{case} Abs Error {points}', fontsize=22)
        ax_diff1.set_xticks([])
        ax_diff1.set_yticks([])
        ax_diff1.set_aspect('equal')

        # 绘制第二个程序的预测温度图
        ax_temp2 = axes2[j, i * 2]
        divider3 = make_axes_locatable(ax_temp2)
        cax3 = divider3.append_axes("right", size="5%", pad=0.2)
        c3 = ax_temp2.contourf(x_grid, y_grid, T_pred2, levels=100, cmap='jet')
        cbar3 = fig2.colorbar(c3, cax=cax3)
        cbar3.ax.set_ylabel('')
        cbar3.set_ticks(cbar3.get_ticks()[::2])
        cbar3.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_temp2.set_title(f'{case} {points}', fontsize=22)
        ax_temp2.set_xticks([])
        ax_temp2.set_yticks([])
        ax_temp2.set_aspect('equal')

        # 绘制第二个程序的绝对误差热场图
        ax_diff2 = axes2[j, i * 2 + 1]
        divider4 = make_axes_locatable(ax_diff2)
        cax4 = divider4.append_axes("right", size="5%", pad=0.2)
        c4 = ax_diff2.tricontourf(actual_data['x'], actual_data['y'], absolute_error2, levels=100, cmap=cmap, norm=norm)
        cbar4 = fig2.colorbar(c4, cax=cax4)
        cbar4.ax.set_ylabel('')
        cbar4.set_ticks(cbar4.get_ticks()[::2])
        cbar4.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_diff2.set_title(f'{case} Abs Error {points}', fontsize=24)
        ax_diff2.set_xticks([])
        ax_diff2.set_yticks([])
        ax_diff2.set_aspect('equal')

# 确保保存路径存在
save_path_svg1 = 'D:\\桌面\\科研绘图pinn110111.svg'
save_path_png1 = 'D:\\桌面\\科研绘图pinn110111.png'
save_path_svg2 = 'D:\\桌面\\科研绘图nn110111.svg'
save_path_png2 = 'D:\\桌面\\科研绘图nn110111.png'
os.makedirs(os.path.dirname(save_path_svg1), exist_ok=True)
os.makedirs(os.path.dirname(save_path_svg2), exist_ok=True)

# 保存最终的大图
fig1.savefig(save_path_svg1, format='svg')
fig1.savefig(save_path_png1, format='png', dpi=300)
fig2.savefig(save_path_svg2, format='svg')
fig2.savefig(save_path_png2, format='png', dpi=300)
plt.show()
