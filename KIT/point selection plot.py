import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件，假设文件包含 'x' 和 'y' 列
file_path = r'D:\桌面\新选点\四边等温64.csv'
data = pd.read_csv(file_path)

# 提取 x 和 y 坐标
x = data['x']
y = data['y']

# 创建图形
plt.figure(figsize=(8, 8))  # 设置图形为正方形

# 绘制数据点
plt.scatter(x, y, c='red', label='Selected Points', alpha=0.8)

# 设置正方形区域
plt.xlim([min(x.min(), y.min()), max(x.max(), y.max())])
plt.ylim([min(x.min(), y.min()), max(x.max(), y.max())])

# 设置轴比例为1:1，确保正方形
plt.gca().set_aspect('equal', adjustable='box')

# 设置图例、标题和坐标轴标签
plt.legend()
plt.title("Locations of Selected Points in a Square Area")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

# 保存为SVG格式
plt.savefig(r'D:\桌面\svg\选点64.svg', format='svg')

# 显示图像
plt.grid(True)
plt.show()
