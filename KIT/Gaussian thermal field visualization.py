import torch
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

# 数据加载
train_data = pd.read_csv(r'D:\桌面\新选点\四边等温64.csv')
test_data = pd.read_csv(r'D:\桌面\新选点\200个点.csv')  # 替换为测试集文件的实际路径

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

# 生成网格以便于绘制预测图
grid_size = 300  # 增加网格密度来提高图像精度
x1 = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), grid_size)
x2 = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), grid_size)
x1x2 = np.meshgrid(x1, x2)
x_grid = np.column_stack([x1x2[0].ravel(), x1x2[1].ravel()])

# 对网格点进行归一化
x_grid_scaled = scaler_x.transform(x_grid)

# 用高斯过程模型对网格点进行预测不确定度
T_pred_scaled, sigma_grid = gp.predict(x_grid_scaled, return_std=True)

# 将预测结果和不确定度转换成网格形式，以便于绘图
sigma_grid_reshaped = sigma_grid.reshape(grid_size, grid_size)

# 设置淡雅的配色方案
cmap_uncertainty = 'magma_r'

# 随机生成64个点
np.random.seed(42)  # 固定种子，保证可重复性
random_points = np.random.rand(64, 2)
random_points[:, 0] = random_points[:, 0] * (x_train[:, 0].max() - x_train[:, 0].min()) + x_train[:, 0].min()
random_points[:, 1] = random_points[:, 1] * (x_train[:, 1].max() - x_train[:, 1].min()) + x_train[:, 1].min()

# 对64个点进行不确定度预测
random_points_scaled = scaler_x.transform(random_points)
_, sigma_random_points = gp.predict(random_points_scaled, return_std=True)

# 将不确定度最低的20个点挑选出来
lowest_uncertainty_indices = np.argsort(sigma_random_points)[:20]
lowest_uncertainty_points = random_points[lowest_uncertainty_indices]

# 在已有的不确定度图上标出不确定度最低的20个点
plt.figure(figsize=(10, 8))
plt.contourf(x1x2[0], x1x2[1], sigma_grid_reshaped, levels=100, cmap=cmap_uncertainty)
plt.colorbar(label="Uncertainty (Standard Deviation)")
plt.scatter(lowest_uncertainty_points[:, 0], lowest_uncertainty_points[:, 1], c='green', label='Lowest 20 Points', alpha=0.9)
plt.title('Uncertainty with Lowest 20 Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(False)

# 保存为 SVG 格式（不确定性最低的20个点）
plt.savefig(r'D:\桌面\svg\uncertainty_field_with_lowest_20_points.svg', format='svg')

# 挑选不确定度适中的10个点
middle_range_indices = np.argsort(sigma_random_points)[len(sigma_random_points)//2 - 5: len(sigma_random_points)//2 + 5]
middle_uncertainty_points = random_points[middle_range_indices]

# 在已有的不确定度图上标出不确定度适中的10个点
plt.figure(figsize=(10, 8))
plt.contourf(x1x2[0], x1x2[1], sigma_grid_reshaped, levels=100, cmap=cmap_uncertainty)
plt.colorbar(label="Uncertainty (Standard Deviation)")
plt.scatter(middle_uncertainty_points[:, 0], middle_uncertainty_points[:, 1], c='blue', label='Moderate 10 Points', alpha=0.9)
plt.title('Uncertainty with Moderate 10 Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(False)

# 保存为 SVG 格式（不确定度适中的10个点）
plt.savefig(r'D:\桌面\svg\uncertainty_field_with_middle_10_points.svg', format='svg')

print("Uncertainty plots with selected points saved successfully.")
