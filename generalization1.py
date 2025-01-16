import numpy as np
import torch.nn as nn
from math import sin, cos, pi
import torch
import os
import scipy.io as sio
from scipy.io import savemat
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
seq_time = 4
class lstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.05, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #out, _ = self.gru(x)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        out, _ = self.lstm(x[:, :seq_time, :])
        query = out[:, -1:, :]
        # out = self.linear(out[:, -1, :])
        seq_embed, _ = self.attention(query, out, out)
        out = self.linear(seq_embed[:, 0, :])
        return out
#数据归一化
def guiyihua(Yn,X_max,X_min):
    Yn_1 = (Yn - X_min) / (X_max - X_min)
    return Yn_1
#数据反归一化
def fanguiyihua(Yn, X_max, X_min):
    Yn_11 = Yn * (X_max - X_min) + X_min
    return Yn_11
def runge_kutta_step(f, xn, Yn, h):
    K1 = f(xn, Yn)
    K2 = f(xn, Yn + K1 * h / 2.0)
    K3 = f(xn, Yn + K2 * h / 2.0)
    K4 = f(xn, Yn + K3 * h)

    Yn1 = Yn + (K1 + 2 * K2 + 2 * K3 + K4) * h / 6.0
    return Yn1


# 例如二元方程组
def f(xn, Yn):  #在这个微分方程中，x为控制量，y为状态量
    y1, y2, y3, y4, y5, y6 = Yn
    # y5 = y5 % (2 * pi)
    # if y5 > pi and y5 <= 2 * pi:
    #     y5 = y5 - 2 * pi
    # y6 = y6 % (2 * pi)
    # if y6 > pi and y6 <= 2 * pi:
    #     y6 = y6 - 2 * pi
    x1, x2, x3 = xn
    DD = 0.5 * 1.225 * (1-0.00688*(y3*3.2808/1000))**4.256*(y4**2)*27.871*(0.0476-0.1462*x2+0.0491*(x2**2)+12.8046*(x2**3)-12.6985*(x2**4))

    LL = 0.5 * 1.225 * (1-0.00688*(y3*3.2808/1000))**4.256*(y4**2)*27.871*(0.0174+4.3329*x2-1.3048*(x2**2)+2.2442*(x2**3)-5.8517*(x2**4))

    f1 = y4 * cos(y6) *cos(y5)
    f2 = y4 * cos(y6) *sin(y5)
    f3 = y4 * sin(y6)
    f4 = (91130 * x1 * cos(x2)-DD)/9299 - 9.8 * sin(y6)
    f5 = (91130 * x1 * sin(x2)+LL) * sin(x3)/9299/y4/cos(y6)
    f6 = (91130 * x1 *sin(x2)+LL) * cos(x3)/9299/y4 - 9.8 * cos(y6)/y4

    return np.array([f1, f2, f3, f4, f5, f6])

# 自定义RBF模块
class RBF(nn.Module):
    def __init__(self, num_centers, input_dim, center, variance=1.0):  #平滑指数
        super(RBF, self).__init__()
        self.num_centers = num_centers
        self.input_dim = input_dim
        self.variance = variance

        self.centers = center

    def forward(self, x):
        x = x.reshape(1,6)
        # print(x.shape)
        size = (1, self.num_centers, self.input_dim)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = x.expand(size)
        # print(x.shape)
        c = self.centers.unsqueeze(0).expand(size)
        c = c.to("cpu")
        # print(x.device,c.device)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.variance

        out = torch.exp(-distances)
        return out
# 定义网络
class RBFNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, centerss):
        super(RBFNet, self).__init__()
        # self.input_layer = nn.Linear(input_dim, hidden_dim)
        # self.hidden_layer = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     RBF(hidden_dim, input_dim, center=centerss)
        # )
        self.hidden_layer = RBF(hidden_dim, input_dim, centerss)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.input_layer(x)
        x = torch.from_numpy(x).float()
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

#遍历得到初始状态
def calculate_cube_range(center, side_length):
    """
    计算以给定点为中心的正方体的坐标范围。

    参数:
    center (tuple): 中心点的坐标 (x, y, z)
    side_length (int): 正方体的边长

    返回:
    tuple: 正方体的坐标范围 (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    x, y, z = center
    half_length = side_length / 2

    x_min = x - half_length
    x_max = x + half_length
    y_min = y - half_length
    y_max = y + half_length
    z_min = z - half_length
    z_max = z + half_length

    return (x_min, x_max, y_min, y_max, z_min, z_max)

#主程序
#将两个网络模型引入
model1 = torch.load('model_path_rbf')
model1.to('cpu')
X_max1 = np.array([3200, 2400, 640.0266, 213.5, 0.98149, 0.40458])
X_min1 = np.array([-3959.99487, -6359.99968, 200, 122.10758, -0.00001, -0.21532])

model2 = torch.load('model_path_lstmattn')
# print(model2)
X_max2 = np.array([-3061.63633, -6187.26767, 366.25146, 200, 0.70864, 0.39824])
X_min2 = np.array([-3959.99934, -6359.99968, 219.50803, 164.54751, -0.00001, -0.01396])



# 中心点坐标
center = (-3955, -6355, 225)

# 输入边长
side_length = 19.2

# 计算坐标范围
x_min, x_max, y_min, y_max, z_min, z_max = calculate_cube_range(center, side_length)
# x_min, x_max = -3964, -3946
# y_min, y_max = -6364, -6346
# z_min, z_max = 216, 234
# # 初始化数组
# coords = []
#
# jump_step = 2
#
# # 遍历x-y面
# for x in range(x_min, x_max + 1, jump_step):
#     # if x == x_max-1:
#     #     x = x-1
#     for y in range(y_min, y_max + 1, jump_step):
#         # if y == y_max-1:
#         #     y = y - 1
#         coords.append([x, y, z_min])
#         coords.append([x, y, z_max])
#
# # 遍历x-z面
# for x in range(x_min, x_max + 1, jump_step):
#     # if x == x_max-1:
#     #     x = x-1
#     for z in range(z_min+1, z_max, jump_step):
#         # if z == z_max-2:
#         #     z = z-1
#         coords.append([x, y_min, z])
#         coords.append([x, y_max, z])
#
# # 遍历y-z面
# for y in range(y_min+1, y_max, jump_step):
#     # if y == y_max-2:
#     #     y = y - 1
#     for z in range(z_min+1, z_max, jump_step):
#         # if z == z_max-2:
#         #     z = z - 1
#         coords.append([x_min, y, z])
#         coords.append([x_max, y, z])

# 初始化数组
coords = []

jump_step = 3  # 步长（可以是浮点数）

# 遍历x-y面（z_min和z_max）
x_values = np.arange(x_min, x_max + 1, jump_step).tolist()
if x_values[-1] != x_max:  # 确保x_max被采集到
    x_values.append(x_max)

y_values = np.arange(y_min, y_max + 1, jump_step).tolist()
if y_values[-1] != y_max:  # 确保y_max被采集到
    y_values.append(y_max)

for x in x_values:
    for y in y_values:
        # 添加z_min和z_max的点
        coords.append([x, y, z_min])
        coords.append([x, y, z_max])

# 遍历x-z面（y_min和y_max）
z_values = np.arange(z_min + jump_step, z_max, jump_step).tolist()
if z_values[-1] != z_max - 1:  # 确保z_max被采集到
    z_values.append(z_max)

for x in x_values:
    for z in z_values:
        # 添加y_min和y_max的点
        coords.append([x, y_min, z])
        coords.append([x, y_max, z])

# 遍历y-z面（x_min和x_max）
for y in y_values[1:-1]:  # 跳过y_min和y_max，避免重复
    for z in z_values:
        # 添加x_min和x_max的点
        coords.append([x_min, y, z])
        coords.append([x_max, y, z])

# 去重（虽然逻辑上已经避免重复，但可以进一步确保）
unique_coords = []
seen = set()
for coord in coords:
    coord_tuple = tuple(coord)
    if coord_tuple not in seen:
        seen.add(coord_tuple)
        unique_coords.append(coord)


coords = unique_coords

coords = np.array(coords)
print(coords.shape)
hang1 = coords.shape[0]
# print(coords[1,:].shape)
print("------")
print(hang1)

h1 = 0.001
h2 = 0.01


for n in range(0,hang1):
    print(n)
    T = 0
    staterbf2 =  np.concatenate([coords[n,:], [200, 0, 0]])
    conrbf2 = np.array([0, 0, 0])  # 为了对齐保存，首行控制不要，初始化为0
    stateattn1 = np.concatenate([coords[n,:], [200, 0, 0]])
    conattn1 = np.array([0, 0, 0])
    # x = [0, 0]
    # print(x.shape)
    # y = data0[0, 1]
    # z = data0[0, 2]
    for i in range(550):
        Yn2 =  np.concatenate([coords[n,:], [200, 0, 0]])
        # print(Yn2)
        # print(Yn2.shape,X_max2.shape,X_min2.shape)
        Yn2 = guiyihua(Yn2, X_max2, X_min2)
        xn2 = model2(Yn2)
        # print(xn.shape)
        # print(Yn)
        xn2 = xn2.detach().numpy()
        xn2 = xn2[0]
        Yn2 = fanguiyihua(Yn2, X_max2, X_min2)
        Yn22 = runge_kutta_step(f, xn2, Yn2, h2)
        # state1 = np.append(state1, [Yn1], axis=0)
        # con1 = np.append(con1, [xn], axis=0)
        staterbf2 = np.vstack((staterbf2, Yn22))
        conrbf2 = np.vstack((conrbf2, xn2))
        Yn2 = Yn22
        T = T + h2
    #暂定15个间隔组成启动序列
    for i in range(0,3):
        stateattn1 = np.vstack((stateattn1, staterbf2[(i+1)*15,:]))
        conattn1 = np.vstack((conattn1, conrbf2[(i+1)*15,:]))
    Yn1 = np.empty((1, 4, 6))
    # Yn[0,0,:] = [-3951.25674835049,	-6356.61787025357,	228.555790051595,	200.000000000000,	0,	0]
    # print(Yn[0,0,:].shape)
    Yn1[0, 0, :] = stateattn1[0, 0:6]
    Yn1[0, 1, :] = stateattn1[1, 0:6]
    Yn1[0, 2, :] = stateattn1[2, 0:6]
    Yn1[0, 3, :] = stateattn1[3, 0:6]
    start_time = time.time()
    for i in range(75000):
        # start_time = time.time()
        Yn1 = guiyihua(Yn1,X_max1,X_min1)
        xn1 = model1(Yn1)
        # print(xn.shape)
        xn1 = xn1[0]
        xn1 = xn1.detach().numpy()
        Yn1 = fanguiyihua(Yn1,X_max1,X_min1)
        Yn11 = runge_kutta_step(f, xn1, Yn1[-1,-1,:], h1)
        stateattn1 = np.append(stateattn1,[Yn11],axis=0)
        conattn1 = np.append(conattn1,[xn1],axis=0)
        YY = Yn1[:, 1:, :]
        # print(YY.shape)
        Yn1[0, 0:3, :] = YY
        Yn1[0, 3, :] = Yn11.reshape(1, 6)
    end_time = time.time()
    time1 = end_time - start_time
    save_pathnpytime = 'save_time_path'
    filenametime = f"geneltrajnettime{n}.npy"
    full_pathtime = save_pathnpytime + '/' + filenametime
    np.save(full_pathtime, time1)
    print(stateattn1.shape, conattn1.shape)
    trajectory1 = np.concatenate((stateattn1, conattn1), axis=1)
    save_pathnpymat = 'save_trajectory_path'
    filenamenpy = f"geneltrajnet{n}.npy"
    full_pathnpy = save_pathnpymat + '/' + filenamenpy
    np.save(full_pathnpy, trajectory1)
    filenamemat = f"geneltrajnet{n}.mat"
    full_pathmat = save_pathnpymat + '/' + filenamemat
    savemat(full_pathmat, {'trajectory': trajectory1})
    # np.save('state1.npy', state1)
    # np.save('con1.npy', con1)
    # 绘制轨迹
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(stateattn1[:,0], stateattn1[:,1], stateattn1[:,2], label='line 1')
    # ax.plot(x0, y0, z0, label='line 2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    save_path = 'save_fig_path'
    figfilename = f"geneltraj{n}.png"
    full_path = save_path + '/' + figfilename
    plt.savefig(full_path)
    # plt.show()
    plt.close()




