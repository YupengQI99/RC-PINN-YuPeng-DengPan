import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# 加载数据
data = pd.read_csv(r'D:\桌面\newtest\混合94个点_排序11.csv')

# 生成网格以便于绘制热力图
x = np.linspace(data['x'].min(), data['x'].max(), 100)
y = np.linspace(data['y'].min(), data['y'].max(), 100)
x_grid, y_grid = np.meshgrid(x, y)

# 使用 scipy 的 griddata 进行插值，生成温度场
temperature_grid = griddata((data['x'], data['y']), data['Temperature'], (x_grid, y_grid), method='cubic')

# 绘制热力图
plt.figure(figsize=(10, 8))
plt.contourf(x_grid, y_grid, temperature_grid, cmap='coolwarm', levels=100)

# 标注数据点
plt.scatter(data['x'], data['y'], c='black', marker='x', label='Data Points')

# 添加颜色条，表示温度
plt.colorbar(label='Temperature')

# 添加标题和坐标轴标签
plt.title('Temperature Field with Data Points')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# 显示图例和网格
plt.legend()
plt.grid(True)

# 保存图像为SVG格式
plt.savefig(r'D:\桌面\newtest\temperature_heatmap_with_points.svg', format='svg')

# 显示图像
plt.show()
