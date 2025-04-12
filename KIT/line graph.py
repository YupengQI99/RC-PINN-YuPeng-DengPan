import matplotlib.pyplot as plt

# 数据
data_points = [34, 49, 64, 79, 94, 109, 124, 139, 154]

# 第一个例子
mae_pinn = [1.8578, 1.3424, 1.1413, 1.0236, 0.9172, 0.8092, 0.7522, 0.7772]
mre_pinn = [0.5821, 0.4147, 0.3512, 0.2992, 0.2843, 0.2508, 0.2322, 0.2405]
r2_pinn = [0.9444, 0.9797, 0.9829, 0.9876, 0.9904, 0.9920, 0.9929, 0.9923]

# 第二个例子
mae_gp = [1.7117, 1.3805, 1.2107, 0.9909, 1.0178, 0.8775, 0.8987, 0.8804]
mre_gp = [0.5036, 0.4055, 0.3543, 0.2896, 0.2972, 0.2568, 0.2637, 0.2589]
r2_gp = [0.9767, 0.9858, 0.9904, 0.9930, 0.9913, 0.9937, 0.9940, 0.9929]

# 第三个例子
mae_case3 = [4.116, 2.809, 1.3451, 1.6614, 1.1254, 1.1990, 1.1685, 0.8923]
mre_case3 = [1.225, 0.8366, 0.3934, 0.4877, 0.3293, 0.3498, 0.3421, 0.2570]
r2_case3 = [0.9287, 0.9496, 0.9940, 0.9909, 0.9953, 0.9950, 0.9959, 0.9974]

# 设置科研风格
plt.style.use('seaborn-white')

# 创建 1x3 子图
fig, axs = plt.subplots(1, 3, figsize=(22, 6))

# 绘制 MAE 图
axs[0].plot(data_points, mae_pinn, marker='o', label='Case 1', color='blue')
axs[0].plot(data_points, mae_gp, marker='s', label='Case 2', color='green')
axs[0].plot(data_points, mae_case3, marker='^', label='Case 3', color='red')
axs[0].set_title('MAE Comparison')
axs[0].set_xlabel('Data Points')
axs[0].set_ylabel('MAE')
axs[0].legend()

# 绘制 MRE 图
axs[1].plot(data_points, mre_pinn, marker='o', label='Case 1', color='blue')
axs[1].plot(data_points, mre_gp, marker='s', label='Case 2', color='green')
axs[1].plot(data_points, mre_case3, marker='^', label='Case 3', color='red')
axs[1].set_title('MRE Comparison')
axs[1].set_xlabel('Data Points')
axs[1].set_ylabel('MRE')
axs[1].legend()

# 绘制 R² 图
axs[2].plot(data_points, r2_pinn, marker='o', label='Case 1', color='blue')
axs[2].plot(data_points, r2_gp, marker='s', label='Case 2', color='green')
axs[2].plot(data_points, r2_case3, marker='^', label='Case 3', color='red')
axs[2].set_title('R² Score Comparison')
axs[2].set_xlabel('Data Points')
axs[2].set_ylabel('R² Score')
axs[2].legend()

# 调整布局，保持整齐
plt.tight_layout()

# 保存为 SVG 格式
plt.savefig('D:\桌面\科研绘图\折线图.png', format='svg')

plt.show()
