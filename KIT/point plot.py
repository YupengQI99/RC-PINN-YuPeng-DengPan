import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 储存热源位置和单位
positions = np.array([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75],
                      [0.75, -0.75], [-0.5, 0.75]])
units = np.array([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25],
                  [0.6, 0.6], [0.25, 0.25]])  # 确保所有位置都有对应的单位

x_loc, y_loc = [], []
for i in range(len(positions)):
    x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
    y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])


# 定义外边界图形
def plot_boundary():
    rect = plt.Rectangle((-1.25, -1.25), 2.5, 2.5, linewidth=2, edgecolor='cyan', facecolor='none')
    plt.gca().add_patch(rect)


# 绘制热源区域
def plot_heat_sources(x_loc, y_loc):
    for i in range(len(x_loc)):
        rect = plt.Rectangle((x_loc[i][0], y_loc[i][0]), x_loc[i][1] - x_loc[i][0], y_loc[i][1] - y_loc[i][0],
                             linewidth=2, edgecolor='cyan', facecolor='none')
        plt.gca().add_patch(rect)


# 绘制输入数据点集
def plot_input_points(data, color='orange', marker='o', label='Selected Points', size=10):
    plt.scatter(data['x'], data['y'], color=color, s=size, edgecolor='black', marker=marker, label=label)


# 设置绘图区域和样式
def setup_plot(title='Chip and Heat Source Layout'):
    plt.figure(figsize=(6, 6))
    plot_boundary()
    plot_heat_sources(x_loc, y_loc)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title(title, fontsize=16)
    plt.grid(False)
    plt.gca().set_aspect('equal', adjustable='box')

    # 去掉坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# 绘制图形
def draw_chip_and_points(data):
    setup_plot()
    plot_input_points(data, color='orange', marker='.', size=20)  # 调整颜色和点大小
    plt.show()


# 示例：绘制图形
data = pd.read_csv('D:\桌面\新选点\四边等温64.csv')
draw_chip_and_points(data)
