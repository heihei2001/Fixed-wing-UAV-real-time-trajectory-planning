import numpy as np
import torch.nn as nn
from math import sin, cos, pi
import torch
import os
import scipy.io as sio
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

seq_time = 4


class lstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.05, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out, _ = self.gru(x)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        out, _ = self.lstm(x[:, :seq_time, :])
        query = out[:, -1:, :]
        # out = self.linear(out[:, -1, :])
        seq_embed, _ = self.attention(query, out, out)
        out = self.linear(seq_embed[:, 0, :])
        return out


# 数据归一化
def guiyihua(Yn, X_max, X_min):
    Yn_1 = (Yn - X_min) / (X_max - X_min)
    return Yn_1


# 数据反归一化
def fanguiyihua(Yn, X_max, X_min):
    Yn_11 = Yn * (X_max - X_min) + X_min
    return Yn_11


def runge_kutta_step(f, xn, Yn, h, wind_x, wind_y, wind_z):
    K1 = f(xn, Yn, wind_x, wind_y, wind_z)
    K2 = f(xn, Yn + K1 * h / 2.0, wind_x, wind_y, wind_z)
    K3 = f(xn, Yn + K2 * h / 2.0, wind_x, wind_y, wind_z)
    K4 = f(xn, Yn + K3 * h, wind_x, wind_y, wind_z)

    Yn1 = Yn + (K1 + 2 * K2 + 2 * K3 + K4) * h / 6.0
    return Yn1


# 例如二元方程组
def f(xn, Yn, wind_x, wind_y, wind_z):  # 在这个微分方程中，x为控制量，y为状态量
    y1, y2, y3, y4, y5, y6 = Yn

    x1, x2, x3 = xn
    DD = 0.5 * 1.225 * (1 - 0.00688 * (y3 * 3.2808 / 1000)) ** 4.256 * (y4 ** 2) * 27.871 * (
                0.0476 - 0.1462 * x2 + 0.0491 * (x2 ** 2) + 12.8046 * (x2 ** 3) - 12.6985 * (x2 ** 4))

    LL = 0.5 * 1.225 * (1 - 0.00688 * (y3 * 3.2808 / 1000)) ** 4.256 * (y4 ** 2) * 27.871 * (
                0.0174 + 4.3329 * x2 - 1.3048 * (x2 ** 2) + 2.2442 * (x2 ** 3) - 5.8517 * (x2 ** 4))

    f1 = y4 * cos(y6) * cos(y5)-wind_x
    f2 = y4 * cos(y6) * sin(y5)-wind_y
    f3 = y4 * sin(y6)-wind_z
    f4 = (91130 * x1 * cos(x2) - DD) / 9299 - 9.8 * sin(y6)
    f5 = (91130 * x1 * sin(x2) + LL) * sin(x3) / 9299 / y4 / cos(y6)
    f6 = (91130 * x1 * sin(x2) + LL) * cos(x3) / 9299 / y4 - 9.8 * cos(y6) / y4

    return np.array([f1, f2, f3, f4, f5, f6])


# 主程序
# 加载训练好的模型

# 文件夹路径
model_folder_path = 'your_model_path'

# 基础路径和文件名模板
base_path = "trajectory_path"
file_template = "traj{}.mat"  # 文件名模板
n_traj = 1

data_file_path = os.path.join(base_path, file_template.format(n_traj))


output_folder_path = 'results_path'
destination_folder_path = 'model_move_path'

#读取风速文件
data_file_path_windx = "wind_x_path"
data_file_path_windy = "wind_y_path"
data_file_path_windz = "wind_z_path"


data_wind_x = sio.loadmat(data_file_path_windx)
data_wind_x = data_wind_x['x_wind3']
data_wind_x = data_wind_x[n_traj-1, 0]
print(data_wind_x.shape)

data_wind_y = sio.loadmat(data_file_path_windy)
data_wind_y = data_wind_y['y_wind3']
data_wind_y = data_wind_y[n_traj-1, 0]

data_wind_z = sio.loadmat(data_file_path_windz)
data_wind_z = data_wind_z['z_wind3']
data_wind_z = data_wind_z[n_traj-1, 0]

data0 = sio.loadmat(data_file_path)
data0 = data0['trajectory']
X_max = np.load('lstmattngru_wind_max.npy')
X_min = np.load('lstmattngru_wind_min.npy')
x0 = data0[:, 0]
y0 = data0[:, 1]
z0 = data0[:, 2]
print(x0.shape)
h = 0.04
for model_file in os.listdir(model_folder_path):
    if model_file.endswith('.pth'):
        # 加载训练好的模型
        # 检查当前可用的CUDA设备数量
        device_count = torch.cuda.device_count()

        # 使用torch.load加载模型时，指定map_location参数将模型加载到CPU上
        if device_count > 0:
            map_location = 'cuda:0'  # 将模型加载到第一个CUDA设备上
        else:
            map_location = 'cpu'  # 如果没有CUDA设备可用，则加载到CPU上

        print(model_file)
        model_path = os.path.join(model_folder_path, model_file)
        model = torch.load(model_path, map_location=map_location)
        model.to('cpu')
        # 移动文件到目标文件夹
        destination_path = os.path.join(destination_folder_path, model_file)
        shutil.move(model_path, destination_path)
        T = 0

        x = data0[0:4, 0]
        y = data0[0:4, 1]
        z = data0[0:4, 2]
        Yn = np.empty((1, 4, 6))

        Yn[0, 0, :] = data0[0, 0:6]
        Yn[0, 1, :] = data0[1, 0:6]
        Yn[0, 2, :] = data0[2, 0:6]
        Yn[0, 3, :] = data0[3, 0:6]
        print(Yn.shape)

        for i in range(2000):
            Yn = guiyihua(Yn, X_max, X_min)
            xn = model(Yn)
            # print(xn.shape)
            xn = xn[0]
            xn = xn.detach().numpy()
            Yn = fanguiyihua(Yn, X_max, X_min)
            Yn1 = runge_kutta_step(f, xn, Yn[-1, -1, :], h, data_wind_x, data_wind_y, data_wind_z)  # 6  1 的

            x = np.append(x, Yn1[0])
            y = np.append(y, Yn1[1])
            z = np.append(z, Yn1[2])
            YY = Yn[:, 1:, :]

            Yn[0, 0:3, :] = YY
            Yn[0, 3, :] = Yn1.reshape(1, 6)
            T = T + h
        print(T)

        # 绘制轨迹

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x, y, z, color='blue', label='Net')
        ax.plot(x0, y0, z0, color='red', label='Trajectory0')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True)

        # plt.show()

        # plt.axis('equal')
        # 保存图像
        output_file_path = os.path.join(output_folder_path, f'{os.path.splitext(model_file)[0]}.png')
        plt.savefig(output_file_path)
        plt.close(fig)






