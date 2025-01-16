import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 生成数据
# x_train = torch.rand(1000, 6)
# y_train = torch.rand(1000, 3)
#
# x_test = torch.rand(100, 6)
# y_test = torch.rand(100, 3)
#数据处理，将数据作为可以处理的形式：数据读取与数据分类，分成状态和控制
folder_path = "dataset_path"
controldata = []  #m*3
statedata = []   #m*6
for file_name in os.listdir(folder_path):
    # print(file_name)
    if file_name.endswith(".mat"):
        file_path = os.path.join(folder_path, file_name)
        data = scipy.io.loadmat(file_path)   #已经得到traj_n，包括状态6+3+3
        data1 = data['trajectory']
        # Do something with the data here
        hang = len(data1)
        for i in range(0,hang-1):   #不要最后一行
            statedata1 = data1[i,:6]
            controldata1 = data1[i,6:9]
            controldata.append(controldata1)
            statedata.append(statedata1)
            if data1[i, 2] > 350:
                break
controldata = np.array(controldata)
statedata = np.array(statedata)
print(controldata.shape)
print(statedata.shape)
#对数据进行归一化处理，只针对输入，并记录6个状态的最值并保存 这时候只有开头部分
# 求最小最大值
X_min = np.min(statedata, axis=0)
X_max = np.max(statedata, axis=0)
# print(X_max,X_min)
# Min-Max规范化
statedata = (statedata - X_min) / (X_max - X_min)
# print(statedata.shape)
# print(statedata[24:26,:])
#划分数据，进行数据包装
num_samples = statedata.shape[0]
num_test = int(num_samples * 0.1)   #以9：1划分训练集和测试集

train_dataX, test_dataX = statedata[:-num_test], statedata[-num_test:]
train_dataY, test_dataY = controldata[:-num_test], controldata[-num_test:]
print(train_dataX.shape)
train_dataX= torch.from_numpy(train_dataX).float()
train_dataY= torch.from_numpy(train_dataY).float()
test_dataX= torch.from_numpy(test_dataX).float()
test_dataY= torch.from_numpy(test_dataY).float()
# K-Means提取5个聚类中心
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_dataX)
centers = torch.from_numpy(kmeans.cluster_centers_)
np.save('centers.npy', centers)
centers = np.load('centers.npy')
centers = torch.from_numpy(centers).float()
centers = centers.to(device)
# 自定义RBF模块
class RBF(nn.Module):
    def __init__(self, num_centers, input_dim, center, variance=1.0):  #平滑指数
        super(RBF, self).__init__()
        self.num_centers = num_centers
        self.input_dim = input_dim
        self.variance = variance

        self.centers = center

    def forward(self, x):
        # print(x.shape)
        size = (x.size(0), self.num_centers, self.input_dim)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = x.expand(size)
        # print(x.shape)
        c = self.centers.unsqueeze(0).expand(size)
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
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


net = RBFNet(6, 5, 3,centers).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())


# 训练函数
def train(net, train_loader, optimizer, criterion):
    net.train()
    epoch_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


# 测试函数
def test(net, test_loader, criterion):
    net.eval()
    epoch_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            epoch_loss += criterion(output, y).item()

    return epoch_loss / len(test_loader)


# 训练网络
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_dataX, train_dataY), batch_size=512,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_dataX, test_dataY), batch_size=512, shuffle=False)

epochs = 100
train_losses = []
test_losses = []
best_model = net
best_val_loss = float('inf')
for epoch in range(epochs):
    train_loss = train(net, train_loader, optimizer, criterion)
    test_loss = test(net, test_loader, criterion)
    train_losses = np.append(train_losses,train_loss)
    test_losses = np.append(test_losses, test_loss)
    print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}')
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        best_model = net
    print(best_val_loss)
save_path1 = "save_model_path"
net.to("cpu")
torch.save(best_model, save_path1)
plt.plot(train_losses)
plt.plot(test_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()