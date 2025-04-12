import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv(r'D:\桌面\newtest\混合94个点_排序11.csv')

# 创建散点图，x 轴为点的索引，y 轴为不确定度，颜色表示温度
plt.figure(figsize=(12, 6))

# 使用散点图，颜色代表温度，点的大小代表不确定度
sc = plt.scatter(range(len(data)), data['Uncertainty'], c=data['Temperature'],
                 cmap='coolwarm', s=data['Uncertainty']*500, alpha=0.7)

# 添加颜色条，表示温度
cbar = plt.colorbar(sc)
cbar.set_label('Temperature')

# 添加标题和轴标签
plt.title('Uncertainty and Temperature of Selected Points')
plt.xlabel('Point Index (sorted by uncertainty)')
plt.ylabel('Uncertainty')

# 显示网格
plt.grid(True)

# 保存图像为SVG格式
plt.savefig(r'D:\桌面\newtest\uncertainty_temperature_scatter.svg', format='svg')

# 显示图像
plt.show()
