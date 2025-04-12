import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import sympy
from PINN.models import PDE, PDE1, PDE2, is_neumann_boundary_x, is_neumann_boundary_y, PDE_inverse, setup_seed
import torch
import numpy as np
from PINN.models import d

# 数据加载
train_data = pd.read_csv(r':csv')
test_data = pd.read_csv(r'.csv')
gaussian_data = pd.read_csv(r'.csv')  # 加载高斯生成的数据


# 自动权重多任务损失类
class AutomaticWeightedLoss(nn.Module):
    """自动权重多任务损失
    Params：
        num: int，损失的数量
        x: 多任务损失
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


# 引入自适应权重模块
awl = AutomaticWeightedLoss(num=4)  # 你有4个不同的损失项
# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_data(data):
    x = torch.from_numpy(data['x'].values.astype(np.float32)).to(device).reshape(-1, 1)
    y = torch.from_numpy(data['y'].values.astype(np.float32)).to(device).reshape(-1, 1)
    T_actual = torch.from_numpy(data['T'].values.astype(np.float32)).to(device).reshape(-1, 1)
    return x, y, T_actual


# 数据预处理
train_x, train_y, train_T_actual = preprocess_data(train_data)
test_x, test_y, test_T_actual = preprocess_data(test_data)
gaussian_x, gaussian_y, gaussian_T_actual = preprocess_data(gaussian_data)  # 预处理高斯数据

# 合并高斯数据与训练数据
train_x = torch.cat([train_x, gaussian_x], dim=0)
train_y = torch.cat([train_y, gaussian_y], dim=0)
train_T_actual = torch.cat([train_T_actual, gaussian_T_actual], dim=0)

parser = get_parser()
args = parser.parse_args()


# 预测温度 T

def Tloss(u0, data_batch):
    x = data_batch['x'].to(device).reshape(-1, 1)
    y = data_batch['y'].to(device).reshape(-1, 1)
    T = u0(torch.cat([x, y], dim=1)).requires_grad_(True)
    T_actual = data_batch['T_actual'].reshape(-1, 1)
    T_actual = T_actual.to(device).requires_grad_(True)
    T_loss = loss(T, T_actual)
    return {'loss_T': T_loss}, T_loss


def pde_loss(x, y, PINN, n_f, positions, units, phi, device):
    x_f = ((x[0] + x[1]) / 2 + (x[1] - x[0]) *
           (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
           ).requires_grad_(True)
    y_f = ((y[0] + y[1]) / 2 + (y[1] - y[0]) *
           (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
           ).requires_grad_(True)

    u_f = PINN(torch.cat([x_f, y_f], dim=1))
    PDE_, out = PDE(u_f, x_f, y_f, positions, units, phi)
    mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))
    return mse_PDE, out


def pde_loss_chip(x, y, PINN, n_f, positions, units, phi, device):
    x_f = ((x[0] + x[1]) / 2 + (x[1] - x[0]) *
           (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
           ).requires_grad_(True)
    y_f = ((y[0] + y[1]) / 2 + (y[1] - y[0]) *
           (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
           ).requires_grad_(True)

    u_f = PINN(torch.cat([x_f, y_f], dim=1))
    PDE_, out = PDE1(u_f, x_f, y_f, positions, units, phi)
    mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))
    return mse_PDE, out


def pde_loss_nochip(x, y, PINN, n_f, positions, units, phi, device, x_locs, y_locs, margin=0.0):
    x_f = ((x[0] + x[1]) / 2 + (x[1] - x[0]) *
           (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
           ).requires_grad_(True)
    y_f = ((y[0] + y[1]) / 2 + (y[1] - y[0]) *
           (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
           ).requires_grad_(True)

    # 判定生成的点是否在 chip 范围内
    is_chip_data = is_chip(x_f, y_f, x_locs, y_locs, margin=margin)

    # 只计算不属于 chip 范围的部分
    x_f_nochip = x_f[~is_chip_data]
    y_f_nochip = y_f[~is_chip_data]

    # 打印调试信息
    print(f"x_f shape: {x_f.shape}, y_f shape: {y_f.shape}")
    print(f"x_f_nochip shape: {x_f_nochip.shape}, y_f_nochip shape: {y_f_nochip.shape}")

    if x_f_nochip.nelement() == 0 or y_f_nochip.nelement() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), None

    # 确保 x_f_nochip 和 y_f_nochip 是二维张量
    if len(x_f_nochip.shape) == 1:
        x_f_nochip = x_f_nochip.unsqueeze(1)
    if len(y_f_nochip.shape) == 1:
        y_f_nochip = y_f_nochip.unsqueeze(1)

    # 确保 x_f_nochip 和 y_f_nochip 形状匹配
    if x_f_nochip.shape != y_f_nochip.shape:
        print(f"Shape mismatch: x_f_nochip.shape = {x_f_nochip.shape}, y_f_nochip.shape = {y_f_nochip.shape}")
        return torch.tensor(0.0, device=device, requires_grad=True), None

    u_f_nochip = PINN(torch.cat([x_f_nochip, y_f_nochip], dim=1))
    PDE_, out = PDE2(u_f_nochip, x_f_nochip, y_f_nochip, positions, units, phi)
    mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))
    return mse_PDE, out


def bc_dirichlet(x, y, PINN):
    u_b = (PINN(torch.cat([x, y], dim=1)))
    mse_BC = args.criterion(u_b, torch.ones_like(u_b))
    return mse_BC


def bc_Neumann_x(x, y, PINN):
    u_b = (PINN(torch.cat([x, y], dim=1)))
    u_BC = is_neumann_boundary_x(u_b, x, y)
    mse_BC = args.criterion(u_BC, torch.zeros_like(u_BC))
    return mse_BC


def bc_Neumann_y(x, y, PINN):
    u_b = (PINN(torch.cat([x, y], dim=1)))
    u_BC = is_neumann_boundary_y(u_b, x, y)
    mse_BC = args.criterion(u_BC, torch.zeros_like(u_BC))
    return mse_BC


def compute_smoothness_loss(T_pred, x, y):
    T_x = d(T_pred, x)
    T_y = d(T_pred, y)
    smoothness_loss = torch.mean(T_x ** 2 + T_y ** 2)
    return smoothness_loss


def compute_laplacian_loss(T_pred, x, y):
    T_x = d(T_pred, x)
    T_y = d(T_pred, y)
    T_xx = d(T_x, x)
    T_yy = d(T_y, y)
    laplacian_loss = torch.mean((T_xx + T_yy) ** 2)
    return laplacian_loss


def heat_flux_continuity_loss(PINN_chip, PINN_non_chip, x_boundary, y_boundary):
    # 计算芯片区域的温度梯度
    T_chip = PINN(torch.cat([x_boundary, y_boundary], dim=1)).requires_grad_(True)
    grad_T_chip = d(T_chip.sum(), [x_boundary, y_boundary])

    # 计算非芯片区域的温度梯度
    T_non_chip = PINN1(torch.cat([x_boundary, y_boundary], dim=1)).requires_grad_(True)
    grad_T_non_chip = d(T_non_chip.sum(), [x_boundary, y_boundary])

    # 热通量连续性损失函数
    loss_x = args.criterion(grad_T_chip[0], grad_T_non_chip[0])
    loss_y = args.criterion(grad_T_chip[1], grad_T_non_chip[1])
    return loss_x + loss_y


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
        self._initialize_weights()

    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


PINN = MLP().to(device)
PINN1 = MLP().to(device)

# 损失函数
loss = nn.MSELoss()

# 储存热源位置
positions = torch.Tensor([[0., 0.], [0.75, 0.25], [0.25, 0.75], [-0.75, -0.75], [-0.75, 0.75],
                          [0.75, -0.75], [-0.5, 0.75]])
units = torch.Tensor([[0.5, 0.5], [0.325, 0.325], [0.5, 0.5], [0.325, 0.325], [0.25, 0.25], [0.6, 0.6], [0.25, 0.25]])
x_loc, y_loc = [], []
for i in range(len(positions)):
    positions[i, 0] - units[i, 0] / 2
    x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
    y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])


# 定义位置判断函数
def is_chip(x, y, x_locs, y_locs, margin=0.001):
    """
    判断点 (x, y) 是否在特定的 chip 位置范围内，包括一个边距参数
    """
    is_chip = torch.zeros_like(x, dtype=torch.bool)
    for i in range(len(x_locs)):
        x_min, x_max = x_locs[i]
        y_min, y_max = y_locs[i]
        mask = (x >= (x_min - margin)) & (x <= (x_max + margin)) & (y >= (y_min - margin)) & (y <= (y_max + margin))
        is_chip = is_chip | mask
    return is_chip


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


XS, YS = sympy.symbols('x y')
rectan = gd2.Rectangle((-1.25, -1.25), (1.25, 1.25))

# 定义批次大小
batch_size = 32

# 将数据合并到一个 TensorDataset 中
train_dataset = TensorDataset(train_x, train_y, train_T_actual)
test_dataset = TensorDataset(test_x, test_y, test_T_actual)

# 创建 DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 创建一个新的优化器，包含热应力模型的参数
optimizer = args.optimizer(list(PINN.parameters()) + list(PINN1.parameters()) + list(awl.parameters()), args.lr)

# 学习率调度器

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

min_MAE = float('inf')
min_val_MAE = float('inf')
min_MRE = float('inf')
min_val_MRE = float('inf')
max_R2 = float('-inf')
max_val_R2 = float('-inf')

# 初始化损失列表
losses = {
    'loss_T': [],
    'MAE': [],
    'MRE': [],
    'R2': [],
    'pde_pcb_loss': [],
    'bloss_loss': []
}
val_losses = {
    'val_loss_T': [],
    'val_MAE': [],
    'val_MRE': [],
    'val_R2': []
}

# 定义变量以保存最佳模型
min_val_loss = float('inf')

num_epochs = 20000

phi = []
power1 = np.array([140])
power1 = torch.from_numpy(power1).float().cuda().requires_grad_(True)
# optimizer.add_param_group({'params': power1, 'lr': 0.001})
phi.append(power1)

power2 = np.array([177.7888])
power2 = torch.from_numpy(power2).float().cuda().requires_grad_(True)
# optimizer.add_param_group({'params': power2, 'lr': 0.001})
phi.append(power2)

power3 = np.array([140])
power3 = torch.from_numpy(power3).float().cuda().requires_grad_(True)
# optimizer.add_param_group({'params': power3, 'lr': 0.001})
phi.append(power3)

power4 = np.array([177.7888])
power4 = torch.from_numpy(power4).float().cuda().requires_grad_(True)
# optimizer.add_param_group({'params': power4, 'lr': 0.001})
phi.append(power4)

power5 = np.array([240])
power5 = torch.from_numpy(power5).float().cuda().requires_grad_(True)
# optimizer.add_param_group({'params': power5, 'lr': 0.001})
phi.append(power5)

power6 = np.array([166.67777])
power6 = torch.from_numpy(power6).float().cuda().requires_grad_(True)
# optimizer.add_param_group({'params': power6, 'lr': 0.001})
phi.append(power6)

power7 = np.array([240])
power7 = torch.from_numpy(power7).float().cuda().requires_grad_(True)
# optimizer.add_param_group({'params': power7, 'lr': 0.001})
phi.append(power7)

n_f = 10000
n_f_loc = 1000
n_b_bc = 1000
n_c_bc = 1000

for epoch in range(num_epochs):
    PINN.train()
    epoch_losses = {
        'loss_T': 0,
        'MAE': 0,
        'MRE': 0,
        'R2': 0,
        'pde_pcb_loss': 0,
        'bloss_loss': 0
    }
    print(f"Epoch {epoch + 1}/{num_epochs} - 当前损失权重: {[param.item() for param in awl.params]}")
    # 计算 PDE 损失
    mse_PDE_c, _ = pde_loss_nochip(
        x=[-1.25, 1.25],
        y=[-1.25, 1.25],
        PINN=PINN1,
        n_f=n_f,
        positions=positions,
        units=units,
        phi=phi,
        device=device,
        x_locs=x_loc,
        y_locs=y_loc,
        margin=0.001  # 设置margin
    )
    mse_PDE0, _ = pde_loss_chip(x=x_loc[0], y=y_loc[0], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                                phi=phi, device=device)
    mse_PDE1, _ = pde_loss_chip(x=x_loc[1], y=y_loc[1], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                                phi=phi, device=device)
    mse_PDE2, _ = pde_loss_chip(x=x_loc[2], y=y_loc[2], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                                phi=phi, device=device)
    mse_PDE3, _ = pde_loss_chip(x=x_loc[3], y=y_loc[3], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                                phi=phi, device=device)
    mse_PDE4, _ = pde_loss_chip(x=x_loc[4], y=y_loc[4], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                                phi=phi, device=device)
    mse_PDE5, _ = pde_loss_chip(x=x_loc[5], y=y_loc[5], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                                phi=phi, device=device)
    mse_PDE6, _ = pde_loss_chip(x=x_loc[6], y=y_loc[6], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                                phi=phi, device=device)
    mse_PDE = mse_PDE_c + mse_PDE0 + mse_PDE1 + mse_PDE2 + mse_PDE3 + mse_PDE4 + mse_PDE5 + mse_PDE6

    # 生成边界点
    n_b_bc = 1000
    boundary_x_top_bottom = torch.FloatTensor(np.random.uniform(-1.25, 1.25, (n_b_bc, 1))).to(device)
    boundary_y_top = torch.full((n_b_bc, 1), 1.25).to(device)
    boundary_y_bottom = torch.full((n_b_bc, 1), -1.25).to(device)
    boundary_y_left_right = torch.FloatTensor(np.random.uniform(-1.25, 1.25, (n_b_bc, 1))).to(device)
    boundary_x_left = torch.full((n_b_bc, 1), -1.25).to(device)
    boundary_x_right = torch.full((n_b_bc, 1), 1.25).to(device)

    # 将边界点转换为需要梯度的张量
    x_bc_left = boundary_x_left.requires_grad_(True)
    y_bc_left = boundary_y_left_right.requires_grad_(True)
    x_bc_right = boundary_x_right.requires_grad_(True)
    y_bc_right = boundary_y_left_right.requires_grad_(True)
    x_bc_down = boundary_x_top_bottom.requires_grad_(True)
    y_bc_down = boundary_y_bottom.requires_grad_(True)
    x_bc_up = boundary_x_top_bottom.requires_grad_(True)
    y_bc_up = boundary_y_top.requires_grad_(True)

    # 计算边界条件损失
    # mse_BC_left = bc_dirichlet(x=x_bc_left, y=y_bc_left, PINN=PINN1)
    # mse_BC_right = bc_dirichlet(x=x_bc_right, y=y_bc_right, PINN=PINN1)
    # mse_BC_left = bc_Neumann_x(x=x_bc_left, y=y_bc_left, PINN=PINN1)
    # mse_BC_right = bc_Neumann_x(x=x_bc_right, y=y_bc_right, PINN=PINN1)
    mse_BC_down = bc_dirichlet(x=x_bc_down, y=y_bc_down, PINN=PINN1)
    mse_BC_up = bc_dirichlet(x=x_bc_up, y=y_bc_up, PINN=PINN1)
    mse_BC = + mse_BC_down + mse_BC_up   + mse_BC_left+mse_BC_right


    # 计算热源连续性

    def heat_source_boundary_loss(PINN, PINN1, positions, x_loc, y_loc, n_b_bc=1000, device='cuda'):
        boundary_losses = []

        for i in range(len(positions)):
            # 获取当前热源区域的 x 和 y 边界
            x_min, x_max = x_loc[i]
            y_min, y_max = y_loc[i]

            # 上下边界点
            boundary_x_top_bottom = torch.FloatTensor(np.random.uniform(x_min, x_max, (n_b_bc, 1))).to(device)
            boundary_y_top = torch.full((n_b_bc, 1), y_max).to(device)
            boundary_y_bottom = torch.full((n_b_bc, 1), y_min).to(device)

            # 左右边界点
            boundary_y_left_right = torch.FloatTensor(np.random.uniform(y_min, y_max, (n_b_bc, 1))).to(device)
            boundary_x_left = torch.full((n_b_bc, 1), x_min).to(device)
            boundary_x_right = torch.full((n_b_bc, 1), x_max).to(device)

            # 将边界点转换为需要梯度的张量
            x_bc_left = boundary_x_left.requires_grad_(True)
            y_bc_left = boundary_y_left_right.requires_grad_(True)
            x_bc_right = boundary_x_right.requires_grad_(True)
            y_bc_right = boundary_y_left_right.requires_grad_(True)
            x_bc_down = boundary_x_top_bottom.requires_grad_(True)
            y_bc_down = boundary_y_bottom.requires_grad_(True)
            x_bc_up = boundary_x_top_bottom.requires_grad_(True)
            y_bc_up = boundary_y_top.requires_grad_(True)

            # 计算两个网络在热源边界点上的预测温度
            T_pred_chip_boundary = PINN(torch.cat([torch.cat([x_bc_left, x_bc_right, x_bc_up, x_bc_down], dim=0),
                                                   torch.cat([y_bc_left, y_bc_right, y_bc_up, y_bc_down], dim=0)],
                                                  dim=1))
            T_pred_non_chip_boundary = PINN1(torch.cat([torch.cat([x_bc_left, x_bc_right, x_bc_up, x_bc_down], dim=0),
                                                        torch.cat([y_bc_left, y_bc_right, y_bc_up, y_bc_down], dim=0)],
                                                       dim=1))

            # 计算热源边界上的损失
            boundary_loss = loss(T_pred_chip_boundary, T_pred_non_chip_boundary)
            boundary_losses.append(boundary_loss)

        # 合并所有热源区域的边界损失
        return sum(boundary_losses)


    # 边界热值
    boundary_loss = heat_source_boundary_loss(PINN, PINN1, positions, x_loc, y_loc, n_b_bc, device)


    def heat_flux_continuity_loss(PINN, PINN1, positions, x_loc, y_loc, n_b_bc=1000, device='cuda'):
        flux_losses = []

        for i in range(len(positions)):
            # 获取当前热源区域的 x 和 y 边界
            x_min, x_max = x_loc[i]
            y_min, y_max = y_loc[i]

            # 上下边界点 (计算 y 方向的热通量)
            boundary_x_top_bottom = torch.FloatTensor(np.random.uniform(x_min, x_max, (n_b_bc, 1))).to(device)
            boundary_y_top = torch.full((n_b_bc, 1), y_max).to(device)
            boundary_y_bottom = torch.full((n_b_bc, 1), y_min).to(device)

            boundary_x_top_bottom.requires_grad_()
            boundary_y_top.requires_grad_()
            boundary_y_bottom.requires_grad_()

            T_chip_top = PINN(torch.cat([boundary_x_top_bottom, boundary_y_top], dim=1))
            T_chip_bottom = PINN(torch.cat([boundary_x_top_bottom, boundary_y_bottom], dim=1))

            T_non_chip_top = PINN1(torch.cat([boundary_x_top_bottom, boundary_y_top], dim=1))
            T_non_chip_bottom = PINN1(torch.cat([boundary_x_top_bottom, boundary_y_bottom], dim=1))

            # 计算热通量梯度 (y方向)
            flux_chip_y_top = d(T_chip_top, boundary_y_top)
            flux_chip_y_bottom = d(T_chip_bottom, boundary_y_bottom)

            flux_non_chip_y_top = d(T_non_chip_top, boundary_y_top)
            flux_non_chip_y_bottom = d(T_non_chip_bottom, boundary_y_bottom)

            loss_y_top = args.criterion(flux_chip_y_top, flux_non_chip_y_top)
            loss_y_bottom = args.criterion(flux_chip_y_bottom, flux_non_chip_y_bottom)

            # 左右边界点 (计算 x 方向的热通量)
            boundary_y_left_right = torch.FloatTensor(np.random.uniform(y_min, y_max, (n_b_bc, 1))).to(device)
            boundary_x_left = torch.full((n_b_bc, 1), x_min).to(device)
            boundary_x_right = torch.full((n_b_bc, 1), x_max).to(device)

            boundary_y_left_right.requires_grad_()
            boundary_x_left.requires_grad_()
            boundary_x_right.requires_grad_()

            T_chip_left = PINN(torch.cat([boundary_x_left, boundary_y_left_right], dim=1))
            T_chip_right = PINN(torch.cat([boundary_x_right, boundary_y_left_right], dim=1))

            T_non_chip_left = PINN1(torch.cat([boundary_x_left, boundary_y_left_right], dim=1))
            T_non_chip_right = PINN1(torch.cat([boundary_x_right, boundary_y_left_right], dim=1))

            # 计算热通量梯度 (x方向)
            flux_chip_x_left = d(T_chip_left, boundary_x_left)
            flux_chip_x_right = d(T_chip_right, boundary_x_right)

            flux_non_chip_x_left = d(T_non_chip_left, boundary_x_left)
            flux_non_chip_x_right = d(T_non_chip_right, boundary_x_right)

            loss_x_left = args.criterion(flux_chip_x_left, flux_non_chip_x_left)
            loss_x_right = args.criterion(flux_chip_x_right, flux_non_chip_x_right)

            # 计算当前热源区域的总热通量连续性损失
            total_flux_loss = loss_y_top + loss_y_bottom + loss_x_left + loss_x_right
            flux_losses.append(total_flux_loss)

        # 返回所有热源区域的总热通量连续性损失
        return sum(flux_losses)


    heat_flux_loss = heat_flux_continuity_loss(PINN, PINN1, positions, x_loc, y_loc, n_b_bc=1000, device='cuda')
    optimizer.zero_grad()

    # 将所有训练数据合并到一个 tensor 中进行计算
    data_batch = {
        'x': train_x.to(device).requires_grad_(True),
        'y': train_y.to(device).requires_grad_(True),
        'T_actual': train_T_actual.to(device).requires_grad_(True),
    }

    # 计算温度损失
    T_loss1, _ = Tloss(PINN, data_batch)
    T_loss2, _ = Tloss(PINN1, data_batch)

    # 判定每个点是否在 chip 位置范围内
    is_chip_data = is_chip(train_x, train_y, x_loc, y_loc)
    T_pred_chip = PINN(torch.cat([train_x, train_y], dim=1))
    T_pred_non_chip = PINN1(torch.cat([train_x, train_y], dim=1))
    T_pred = torch.where(is_chip_data, T_pred_chip, T_pred_non_chip)

    mae_loss_value = calculate_mae(T_pred, train_T_actual)
    mre_loss_value = calculate_mre(T_pred, train_T_actual)
    r2_loss_value = calculate_r2(T_pred, train_T_actual)

    lambda_laplacian = 0.001  # 拉普拉斯正则项的权重超参数


    # 计算芯片平滑度损失

    def region_temperature_smoothness_loss(PINN, x_locs, y_locs, margin=0.01):
        total_loss = 0
        for i, (x_loc, y_loc) in enumerate(zip(x_locs, y_locs)):
            x_min, x_max = x_loc
            y_min, y_max = y_loc

            # 在该正方形区域内采样点
            x_samples = torch.linspace(x_min, x_max, 10).to(device).reshape(-1, 1)
            y_samples = torch.linspace(y_min, y_max, 10).to(device).reshape(-1, 1)
            x_grid, y_grid = torch.meshgrid(x_samples.squeeze(), y_samples.squeeze())

            x_flat = x_grid.flatten().reshape(-1, 1)
            y_flat = y_grid.flatten().reshape(-1, 1)

            # 预测该区域内的温度
            T_pred = PINN(torch.cat([x_flat, y_flat], dim=1))

            # 计算该区域内部温度的标准差作为损失
            region_loss = torch.std(T_pred)

            total_loss += region_loss

        return total_loss


    region = region_temperature_smoothness_loss(PINN, x_loc, y_loc, margin=0.01)




    # 合并所有损失
    total_loss = awl(
        T_loss1['loss_T'],
        T_loss2['loss_T'],
        boundary_loss,
        mse_BC
    )
    total_loss.backward()

    optimizer.step()

    # 记录损失
    epoch_losses['loss_T'] += T_loss1['loss_T'].item() + T_loss2['loss_T'].item()
    epoch_losses['MAE'] += mae_loss_value.item()
    epoch_losses['MRE'] += mre_loss_value.item()
    epoch_losses['R2'] += r2_loss_value.item()
    epoch_losses['pde_pcb_loss'] += mse_PDE.item()
    epoch_losses['bloss_loss'] += mse_BC.item()

    # 学习率调度器步进
    # scheduler.step()

    # 记录训练损失平均值
    for loss_name in losses.keys():
        epoch_avg_loss = epoch_losses[loss_name]
        losses[loss_name].append(epoch_avg_loss)

    print(f"Epoch {epoch + 1}/{num_epochs} - Training Losses:")
    for loss_name, loss_values in losses.items():
        print(f"    Avg {loss_name}: {loss_values[-1]:.8f}")
    print(f"    MAE: {min_MAE:.8f}")
    print(f"    MRE: {min_MRE:.8f}")
    print(f"    R2: {max_R2:.8f}")

    # 验证部分
    PINN.eval()
    PINN1.eval()
    with torch.no_grad():
        val_epoch_losses = {
            'val_loss_T': 0,
            'val_MAE': 0,
            'val_MRE': 0,
            'val_R2': 0
        }

        data_batch = {
            'x': test_x.to(device),
            'y': test_y.to(device),
            'T_actual': test_T_actual.to(device)
        }

        T_loss1, _ = Tloss(PINN, data_batch)
        T_loss2, _ = Tloss(PINN1, data_batch)

        # 判定每个点是否在 chip 位置范围内
        is_chip_data = is_chip(test_x, test_y, x_loc, y_loc)
        T_pred_chip = PINN(torch.cat([test_x, test_y], dim=1))
        T_pred_non_chip = PINN1(torch.cat([test_x, test_y], dim=1))
        T_pred = torch.where(is_chip_data, T_pred_chip, T_pred_non_chip)

        mae_loss_value = calculate_mae(T_pred, test_T_actual)
        mre_loss_value = calculate_mre(T_pred, test_T_actual)
        r2_loss_value = calculate_r2(T_pred, test_T_actual)

        min_val_MAE = min(min_val_MAE, mae_loss_value.item())
        min_val_MRE = min(min_val_MRE, mre_loss_value.item())
        max_val_R2 = max(max_val_R2, r2_loss_value.item())

        # 记录验证损失
        val_epoch_losses['val_loss_T'] += T_loss1['loss_T'].item() + T_loss2['loss_T'].item()
        val_epoch_losses['val_MAE'] += mae_loss_value.item()
        val_epoch_losses['val_MRE'] += mre_loss_value.item()
        val_epoch_losses['val_R2'] += r2_loss_value.item()

        # 记录和输出验证损失平均值
        for loss_name in val_losses.keys():
            val_avg_loss = val_epoch_losses[loss_name]
            val_losses[loss_name].append(val_avg_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Losses:")
        for loss_name, loss_values in val_losses.items():
            print(f"    Avg {loss_name}: {loss_values[-1]:.8f}")
        print(f"    最小验证MAE: {min_val_MAE:.8f}")
        print(f"    最小验证MRE: {min_val_MRE:.8f}")
        print(f"    最大验证R2: {max_val_R2:.8f}")

        # 保存最佳模型权重
        current_val_loss = val_losses['val_loss_T'][-1]
        if current_val_loss < min_val_loss:
            min_val_loss = current_val_loss
            torch.save(PINN.state_dict(), 'D:.pth')
            torch.save(PINN1.state_dict(), 'D:.pth')

# 保存最佳模型权重


# 绘制损失曲线
plt.figure(figsize=(10, 7))
# 绘制训练损失
plt.plot(range(1, num_epochs + 1), losses['loss_T'], label='Training loss_T')
plt.plot(range(1, num_epochs + 1), losses['pde_pcb_loss'], label='Training pde_pcb_loss')
plt.plot(range(1, num_epochs + 1), losses['bloss_loss'], label='Training bloss_loss')
plt.plot(range(1, num_epochs + 1), val_losses['val_loss_T'], label='Validation loss_T')

# 标记最低点
min_val_loss = min(val_losses['val_loss_T'])
min_val_epoch = val_losses['val_loss_T'].index(min_val_loss)
plt.scatter(min_val_epoch + 1, min_val_loss, color='red')
plt.annotate(f'{min_val_loss:.4f}', xy=(min_val_epoch + 1, min_val_loss),
             xytext=(min_val_epoch + 1 + 10, min_val_loss + 0.1),
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 7))
# 绘制MAE和MRE
plt.plot(range(1, num_epochs + 1), losses['MAE'], label='Training MAE')
plt.plot(range(1, num_epochs + 1), losses['MRE'], label='Training MRE')
plt.plot(range(1, num_epochs + 1), losses['R2'], label='Training R2')
plt.plot(range(1, num_epochs + 1), val_losses['val_MAE'], label='Validation MAE')
plt.plot(range(1, num_epochs + 1), val_losses['val_MRE'], label='Validation MRE')
plt.plot(range(1, num_epochs + 1), val_losses['val_R2'], label='Validation R2')

# 验证MAE
min_val_MAE_val = min(val_losses['val_MAE'])
min_val_MAE_epoch = val_losses['val_MAE'].index(min_val_MAE_val)
plt.scatter(min_val_MAE_epoch + 1, min_val_MAE_val, color='red')
plt.annotate(f'{min_val_MAE_val:.4f}', xy=(min_val_MAE_epoch + 1, min_val_MAE_val),
             xytext=(min_val_MAE_epoch + 1 + 10, min_val_MAE_val + 0.1),
             arrowprops=dict(facecolor='red', shrink=0.05))

# 验证MRE
min_val_MRE_val = min(val_losses['val_MRE'])
min_val_MRE_epoch = val_losses['val_MRE'].index(min_val_MRE_val)
plt.scatter(min_val_MRE_epoch + 1, min_val_MRE_val, color='blue')
plt.annotate(f'{min_val_MRE_val:.4f}', xy=(min_val_MRE_epoch + 1, min_val_MRE_val),
             xytext=(min_val_MRE_epoch + 1 + 10, min_val_MRE_val + 0.1),
             arrowprops=dict(facecolor='blue', shrink=0.05))

# 验证R2
max_val_R2_val = max(val_losses['val_R2'])
max_val_R2_epoch = val_losses['val_R2'].index(max_val_R2_val)
plt.scatter(max_val_R2_epoch + 1, max_val_R2_val, color='green')
plt.annotate(f'{max_val_R2_val:.4f}', xy=(max_val_R2_epoch + 1, max_val_R2_val),
             xytext=(max_val_R2_epoch + 1 + 10, max_val_R2_val + 0.1),
             arrowprops=dict(facecolor='green', shrink=0.05))

plt.xlabel('Epochs')
plt.ylabel('MAE/MRE/R2')
plt.title('Training and Validation MAE/MRE/R2 Curves')
plt.legend()
plt.grid(True)
plt.show()
