import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch import nn
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
plt.rcParams['font.size'] = 18  # 修改字体大小为18号

# 创建绘图
fig, axes = plt.subplots(4, 6, figsize=(30, 20))

# 调整子图间距
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 减少子图之间的距离

# 遍历每个case和数据量
cases = ['case1', 'case2', 'case3']
data_points = [64, 94, 124, 154]

# 定义每个 case 对应的测试集路径
case_to_path = {
    'case1': 'D:\\桌面\\新选点\\一万个点测试集.csv',
    'case2': 'D:\\桌面\\新选点\\绝热测试集.csv',
    'case3': 'D:\\桌面\\新选点\\复杂边界测试集.csv'
}

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
        model_path_PINN = f'D:\\桌面\\{case}权重\\nn{points}.pth'

        # 加载模型权重
        PINN.load_state_dict(torch.load(model_path_PINN))

        # 使用模型预测实际值位置的温度
        with torch.no_grad():
            is_chip_data_actual = is_chip(test_x_actual, test_y_actual, x_loc, y_loc)
            T_pred_actual = PINN(torch.cat([test_x_actual, test_y_actual], dim=1))

        # 计算绝对误差
        absolute_error = np.abs(T_pred_actual.cpu().numpy().flatten() - actual_data['T'].values)

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
        with torch.no_grad():
            is_chip_data = is_chip(test_x, test_y, x_loc, y_loc)
            T_pred = PINN(torch.cat([test_x, test_y], dim=1))

        # 转换预测结果为 numpy 数组
        T_pred = T_pred.cpu().numpy().reshape(x_grid.shape)

        # 绘制预测温度图
        ax_temp = axes[j, i * 2]
        divider1 = make_axes_locatable(ax_temp)
        cax1 = divider1.append_axes("right", size="5%", pad=0.2)  # 调整pad值以增加间距
        c1 = ax_temp.contourf(x_grid, y_grid, T_pred, levels=100, cmap='jet')
        cbar1 = fig.colorbar(c1, cax=cax1)
        cbar1.ax.set_ylabel('')  # 移除颜色条标签
        cbar1.set_ticks(cbar1.get_ticks()[::2])  # 只显示每隔一个刻度的标签
        ax_temp.set_title(f'{case} {points}', fontsize=20)  # 简化标题
        ax_temp.set_xticks([])  # 移除x轴刻度
        ax_temp.set_yticks([])  # 移除y轴刻度
        ax_temp.set_aspect('equal')  # 确保热场图是正方形

        # 绘制绝对误差热场图
        ax_diff = axes[j, i * 2 + 1]
        divider2 = make_axes_locatable(ax_diff)
        cax2 = divider2.append_axes("right", size="5%", pad=0.2)  # 调整pad值以增加间距
        c2 = ax_diff.tricontourf(actual_data['x'], actual_data['y'], absolute_error, levels=100, cmap='jet')
        cbar2 = fig.colorbar(c2, cax=cax2)
        cbar2.ax.set_ylabel('')  # 移除颜色条标签
        cbar2.set_ticks(cbar2.get_ticks()[::2])  # 只显示每隔一个刻度的标签
        ax_diff.set_title(f'{case} Abs Error {points}', fontsize=20)  # 简化标题
        ax_diff.set_xticks([])  # 移除x轴刻度
        ax_diff.set_yticks([])  # 移除y轴刻度
        ax_diff.set_aspect('equal')  # 确保热场图是正方形

# 确保保存路径存在
save_path_svg = 'D:\\桌面\\科研绘图21nn.svg'
save_path_png = 'D:\\桌面\\科研绘图21nn.png'
os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)

# 保存最终的大图
plt.savefig(save_path_svg, format='svg')
plt.savefig(save_path_png, format='png', dpi=300)  # 保存为PNG格式，设置dpi为300以确保清晰度
plt.show()
