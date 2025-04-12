import torch
import pandas as pd
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

# 数据加载
test_data = pd.read_csv(r'test.csv')

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_data(data):
    x = torch.from_numpy(data['x'].values.astype(np.float32)).to(device).reshape(-1, 1)
    y = torch.from_numpy(data['y'].values.astype(np.float32)).to(device).reshape(-1, 1)
    T_actual = torch.from_numpy(data['T'].values.astype(np.float32)).to(device).reshape(-1, 1)
    return x, y, T_actual

# 数据预处理
test_x, test_y, test_T_actual = preprocess_data(test_data)

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

    def forward(self, x):
        return self.net(x)

# 创建模型并加载权重
PINN = MLP().to(device)
PINN1 = MLP().to(device)


model_path_PINN = r'D:.pth'
model_path_PINN1 = r'D:.pth'

# 加载模型权重
PINN.load_state_dict(torch.load(model_path_PINN))
PINN1.load_state_dict(torch.load(model_path_PINN1))

# 储存热源位置
positions = torch.Tensor([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75],
                          [0.75, -0.75], [-0.5, 0.75]])
units = torch.Tensor([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25], [0.6, 0.6], [0.25, 0.25]])
x_loc, y_loc = [], []
for i in range(len(positions)):
    x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
    y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])

# 判定每个点是否在 chip 位置范围内
def is_chip(x, y, x_locs, y_locs, margin=0.01):
    is_chip = torch.zeros_like(x, dtype=torch.bool)
    for i in range(len(x_locs)):
        x_min, x_max = x_locs[i]
        y_min, y_max = y_locs[i]
        mask = (x >= (x_min - margin)) & (x <= (x_max + margin)) & (y >= (y_min - margin)) & (y <= (y_max + margin))
        is_chip = is_chip | mask
    return is_chip

# 进行预测并计算性能指标
PINN.eval()
PINN1.eval()
with torch.no_grad():
    data_batch = {
        'x': test_x,
        'y': test_y,
        'T_actual': test_T_actual
    }

    # 判定每个点是否在 chip 位置范围内
    is_chip_data = is_chip(test_x, test_y, x_loc, y_loc)
    T_pred_chip = PINN(torch.cat([test_x, test_y], dim=1))
    T_pred_non_chip = PINN1(torch.cat([test_x, test_y], dim=1))
    T_pred = torch.where(is_chip_data, T_pred_chip, T_pred_non_chip)

    # 计算 MAE
    def calculate_mae(pred, actual):
        return torch.mean(torch.abs(pred - actual))

    # 计算 MRE
    def calculate_mre(pred, actual):
        return torch.mean(torch.abs(pred - actual) / torch.abs(actual))

    # 计算 R² 分数
    def calculate_r2(pred, actual):
        total_var = torch.sum((actual - torch.mean(actual)) ** 2)
        unexplained_var = torch.sum((actual - pred) ** 2)
        r2_value = 1 - (unexplained_var / total_var)
        return r2_value

    mae_loss_value = calculate_mae(T_pred, test_T_actual)
    mre_loss_value = calculate_mre(T_pred, test_T_actual)
    r2_loss_value = calculate_r2(T_pred, test_T_actual)

    print(f"Test MAE: {mae_loss_value.item():.8f}")
    print(f"Test MRE: {mre_loss_value.item():.8f}")
    print(f"Test R²: {r2_loss_value.item():.8f}")

    # 计算 AE 和 RE
    AE = torch.abs(T_pred - test_T_actual).cpu().numpy().flatten()
    RE = (torch.abs(T_pred - test_T_actual) / torch.abs(test_T_actual)).cpu().numpy().flatten()

    # 将 AE 和 RE 数据分别保存到 CSV 文件中
    ae_data = pd.DataFrame({'AE': AE})
    re_data = pd.DataFrame({'RE': RE})
    ae_data.to_csv(r'.csv', index=False)
    re_data.to_csv(r'.csv', index=False)

    # 使用 np.histogram_bin_edges 自动分割区间
    AE_bins = np.histogram_bin_edges(AE, bins=5)
    RE_bins = np.histogram_bin_edges(RE, bins=5)

    AE_hist, AE_edges = np.histogram(AE, bins=AE_bins)
    RE_hist, RE_edges = np.histogram(RE, bins=RE_bins)

# 绘制 AE 直方图
plt.figure(figsize=(10, 7))
plt.bar(range(len(AE_hist)), AE_hist, color='blue', alpha=0.7, width=0.6,
        tick_label=[f'{AE_edges[i]:.1f}≤AE<{AE_edges[i + 1]:.1f}' for i in range(len(AE_hist))])
for i in range(len(AE_hist)):
    plt.text(i, AE_hist[i] + 50, str(AE_hist[i]), ha='center')
plt.xlabel('Absolute Error of Temperature (K)')
plt.ylabel('Number of Points')
plt.title('Statistical chart of absolute error')
plt.grid(True)
plt.show()

# 绘制 RE 直方图
plt.figure(figsize=(10, 7))
plt.bar(range(len(RE_hist)), RE_hist, color='green', alpha=0.7, width=0.6,
        tick_label=[f'{RE_edges[i]:.4f}≤RE<{RE_edges[i + 1]:.4f}' for i in range(len(RE_hist))])
for i in range(len(RE_hist)):
    plt.text(i, RE_hist[i] + 50, str(RE_hist[i]), ha='center')
plt.xlabel('Relative Error of Temperature')
plt.ylabel('Number of Points')
plt.title('Statistical chart of relative error')
plt.grid(True)
plt.show()