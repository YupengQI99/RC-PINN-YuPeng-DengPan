import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_neural_net(ax, layer_sizes):
    v_spacing = 1. / (max(layer_sizes) - 1)
    h_spacing = 1. / (len(layer_sizes) - 1)

    # 画线和节点
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = np.linspace(0.5 * (1 - (layer_size_a - 1) * v_spacing),
                                  0.5 * (1 + (layer_size_a - 1) * v_spacing), layer_size_a)
        layer_top_b = np.linspace(0.5 * (1 - (layer_size_b - 1) * v_spacing),
                                  0.5 * (1 + (layer_size_b - 1) * v_spacing), layer_size_b)
        for j, y_a in enumerate(layer_top_a):
            for k, y_b in enumerate(layer_top_b):
                line = plt.Line2D([i * h_spacing, (i + 1) * h_spacing], [y_a, y_b], c='gray')
                ax.add_artist(line)

    # 绘制节点
    for i, layer_size in enumerate(layer_sizes):
        layer_top = np.linspace(0.5 * (1 - (layer_size - 1) * v_spacing),
                                0.5 * (1 + (layer_size - 1) * v_spacing), layer_size)
        for j, y in enumerate(layer_top):
            circle = plt.Circle((i * h_spacing, y), v_spacing / 4, color='lightblue', ec='black', zorder=4)
            ax.add_artist(circle)


# 绘制整个图像
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

# 定义层大小
layer_sizes = [4, 5, 4, 3, 4]

# 绘制神经网络
draw_neural_net(ax, layer_sizes)

# 添加红色椭圆
ellipse = patches.Ellipse((0.35, 0.5), 0.2, 0.1, color='red', alpha=0.6)
ax.add_patch(ellipse)

# 在网络中间添加浅色方框
rect1 = patches.Rectangle((0.2, 0.35), 0.25, 0.3, linewidth=2, edgecolor='lightblue', facecolor='none', linestyle='--')
ax.add_patch(rect1)

# 在网络右边添加浅色方框
rect2 = patches.Rectangle((0.6, 0.35), 0.25, 0.3, linewidth=2, edgecolor='lightblue', facecolor='none', linestyle='--')
ax.add_patch(rect2)

# 保存为SVG文件
plt.savefig("neural_network_structure.svg", format="svg")
plt.show()
