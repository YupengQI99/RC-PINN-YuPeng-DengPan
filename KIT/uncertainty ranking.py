import matplotlib.pyplot as plt
import pandas as pd

# 加载数据
data = pd.read_csv(r'D:\桌面\newtest\混合94个点_排序11.csv')

# 绘制散点图
plt.figure(figsize=(10, 6))

# x 轴为点的索引，y 轴为不确定度
plt.scatter(range(len(data)), data['Uncertainty'], c='blue', label='Uncertainty')

# 添加标题和轴标签
plt.title('Points Sorted by Uncertainty')
plt.xlabel('Point Index (sorted by uncertainty)')
plt.ylabel('Uncertainty')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图像
plt.show()
