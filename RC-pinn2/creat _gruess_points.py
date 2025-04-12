import torch
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import qmc  # 用于低序偏差采样
import os
import matplotlib.pyplot as plt

# 打印 torch 版本
print(f"PyTorch version: {torch.__version__}")

# 数据加载
train_data = pd.read_csv(r'.csv')
test_data = pd.read_csv(r'.csv')  # 替换为测试集文件的实际路径

# 数据预处理
def preprocess_data(data):
    x = data['x'].values.reshape(-1, 1)
    y = data['y'].values.reshape(-1, 1)
    T = data['T'].values.reshape(-1, 1)
    return np.hstack((x, y)), T

x_train, T_train_actual = preprocess_data(train_data)
x_test, T_test_actual = preprocess_data(test_data)

# 数据归一化
scaler_x = MinMaxScaler()
scaler_T = MinMaxScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
T_train_actual_scaled = scaler_T.fit_transform(T_train_actual)

x_test_scaled = scaler_x.transform(x_test)

# 增大核函数参数的上限
kernel = C(1.0, (1e-4, 1e6)) * RBF([1.0, 1.0], (1e-4, 1e3))

# 自定义优化器函数，用于设置 max_iter
def custom_optimizer(obj_func, initial_theta, bounds):
    opt_result = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=10000)  # 设置 maxiter
    return opt_result[0], opt_result[1]

# 使用自定义优化器
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=custom_optimizer)
gp.fit(x_train_scaled, T_train_actual_scaled.ravel())

# 测试集预测
T_test_pred_scaled, sigma_test = gp.predict(x_test_scaled, return_std=True)
T_test_pred = scaler_T.inverse_transform(T_test_pred_scaled.reshape(-1, 1)).ravel()

# 计算测试集上的评估指标
mre_test = np.mean(np.abs((T_test_pred - T_test_actual.ravel()) / T_test_actual.ravel()))
mae_test = mean_absolute_error(T_test_actual, T_test_pred)
r2_test = r2_score(T_test_actual, T_test_pred)

print(f"Test Set Metrics:")
print(f"  Mean Relative Error (MRE): {mre_test:.4f}")
print(f"  Mean Absolute Error (MAE): {mae_test:.4f}")
print(f"  R² Score: {r2_test:.4f}")

# 使用低序偏差采样生成点（Sobol序列）
num_new_points = 100  #
x_min, x_max = x_train[:, 0].min(), x_train[:, 0].max()
y_min, y_max = x_train[:, 1].min(), x_train[:, 1].max()

sampler = qmc.Sobol(d=2, scramble=True)  # 创建Sobol采样器
low_discrepancy_points = sampler.random(num_new_points)  # 生成低序偏差采样点
low_discrepancy_points = qmc.scale(low_discrepancy_points, [x_min, y_min], [x_max, y_max])  # 将点映射到数据范围

# 对新点进行归一化
new_points_scaled = scaler_x.transform(low_discrepancy_points)

# 用高斯过程模型预测新点的温度和不确定度
new_targets_scaled, sigma = gp.predict(new_points_scaled, return_std=True)
new_targets = scaler_T.inverse_transform(new_targets_scaled.reshape(-1, 1)).ravel()

# 筛选策略 1: 不确定度最低的30个点
lowest_uncertainty_idx = np.argsort(sigma)[:30]
selected_points_1 = low_discrepancy_points[lowest_uncertainty_idx]
selected_targets_1 = new_targets[lowest_uncertainty_idx]

# 筛选策略 2:不确定度最低的点 + 不确定度适中的点
medium_uncertainty_idx = np.argsort(sigma)[40:45]
selected_idx_2 = np.concatenate([lowest_uncertainty_idx[:25], medium_uncertainty_idx[:10]])
selected_points_2 = low_discrepancy_points[selected_idx_2]
selected_targets_2 = new_targets[selected_idx_2]

# 筛选策略 3: 不确定度最低的点 + 不确定度最高的点
highest_uncertainty_idx = np.argsort(sigma)[-10:]
selected_idx_3 = np.concatenate([lowest_uncertainty_idx[:20], highest_uncertainty_idx])
selected_points_3 = low_discrepancy_points[selected_idx_3]
selected_targets_3 = new_targets[selected_idx_3]

# 保存到目标文件夹
output_folder = r''
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  # 创建文件夹

output_paths = [
    os.path.join(output_folder, "p1.csv"),
    os.path.join(output_folder, "p2.csv"),
    os.path.join(output_folder, "p3.csv"),
    os.path.join(output_folder, "p4.csv"),  # 保存全部点
]

# 保存全部点
all_points_data = np.column_stack((low_discrepancy_points, new_targets, sigma))
np.savetxt(output_paths[3], all_points_data, delimiter=',', header='x,y,T,sigma', comments='', encoding='utf-8')

# 保存筛选策略的点
selected_data_list = [
    np.column_stack((selected_points_1, selected_targets_1)),
    np.column_stack((selected_points_2, selected_targets_2)),
    np.column_stack((selected_points_3, selected_targets_3)),
]

for path, data in zip(output_paths[:3], selected_data_list):
    np.savetxt(path, data, delimiter=',', header='x,y,T', comments='', encoding='utf-8')
    print(f"Data saved to {path}")

# 打印测试集指标到文件
metrics_path = os.path.join(output_folder, "测试集指标.txt")
with open(metrics_path, "w", encoding="utf-8") as f:  # 使用UTF-8编码
    f.write(f"Test Set Metrics:\n")
    f.write(f"  Mean Relative Error (MRE): {mre_test:.4f}\n")
    f.write(f"  Mean Absolute Error (MAE): {mae_test:.4f}\n")
    f.write(f"  R² Score: {r2_test:.4f}\n")
print(f"Test set metrics saved to {metrics_path}")

# 可视化所有策略
plt.figure(figsize=(10, 10))
plt.scatter(low_discrepancy_points[:, 0], low_discrepancy_points[:, 1], c='green', label='Generated Points', alpha=0.3)

# 策略1
plt.scatter(selected_points_1[:, 0], selected_points_1[:, 1], c='red', label='Lowest 30 Points', alpha=0.8)
# 策略2
plt.scatter(selected_points_2[:, 0], selected_points_2[:, 1], c='blue', label='Lowest 20 + Medium 10', alpha=0.8)
# 策略3
plt.scatter(selected_points_3[:, 0], selected_points_3[:, 1], c='orange', label='Lowest 20 + Highest 10', alpha=0.8)

# 设置图例、标题和坐标轴标签
plt.legend()
plt.title("Visualization of Selected Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# 显示图像
plt.show()
