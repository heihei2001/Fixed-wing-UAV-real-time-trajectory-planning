import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# 定义Dataset加载表格数据
class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        x = self.X[index]  # x shape (seq_len, feature_dim)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.X)


# 定义GRU网络
class RegnnModel(nn.Module):
    # def __init__(self, in_dim, out_dim):
    #     super(RegnnModel, self).__init__()
    #     self.layer1 = nn.Linear(in_dim, 256)
    #     self.layer2 = nn.Linear(256, out_dim)
    #
    # def forward(self, x):
    #     x = x.float()
    #     x = torch.relu(self.layer1(x))
    #     x = self.layer2(x)
    #     return x
    def __init__(self, input_dim, output_dim):
        super(RegnnModel, self).__init__()

        # 定义网络层
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_dim)
        # # 定义BatchNorm层
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(64)
        # 激活函数
        self.relu = nn.ReLU()
        self.linear = nn.Identity()

    def forward(self, x):
        # 前向传播
        x = x.float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.linear(self.fc4(x))

        return x


# 训练函数
def train(model, train_loader, optimizer):
    criterion = nn.MSELoss(reduction='mean')
    model.train()
    for x, y in train_loader:
        # print(x.dtype)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        y = y.float()
        loss = criterion(y_hat, y)
        # print(y_hat.dtype)
        # print(y.dtype)
        # print(loss.dtype)
        loss.backward()
        optimizer.step()
    losscpu = loss.to("cpu")

    return losscpu


# 测试函数
def test(model, test_loader):
    criterion = nn.MSELoss(reduction='mean')
    model.eval()
    test_loss = 0.
    test_loss = torch.tensor(test_loss)
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            test_loss += criterion(pred, y).item()

    test_loss /= len(test_loader)
    test_loss = test_loss.to("cpu")
    print('Test loss: {:.8f}'.format(test_loss))

    return test_loss
# def test(model, test_loader):
#     model.eval()
#     outputs = []
#     for x, y in test_loader:
#         x = x.to(device)
#         y = y.to(device)
#         with torch.no_grad():
#             y_hat = model(x)
#             y_hat = y_hat.to("cpu")
#             outputs.append(y_hat.numpy())
#     return np.concatenate(outputs, axis=0)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #定义数据
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
    controldata = np.array(controldata)
    statedata = np.array(statedata)
    print(controldata.shape)
    print(statedata.shape)
    #对数据进行归一化处理，只针对输入，并记录6个状态的最值并保存
    # 求最小最大值
    X_min = np.min(statedata, axis=0)
    X_max = np.max(statedata, axis=0)
    print(X_max,X_min)
    # Min-Max规范化
    statedata = (statedata - X_min) / (X_max - X_min)
    # print(statedata.shape)
    # print(statedata[24:26,:])
    #划分数据，进行数据包装
    num_samples = statedata.shape[0]
    num_test = int(num_samples * 0.1)   #以9：1划分训练集和测试集

    train_dataX, test_dataX = statedata[:-num_test], statedata[-num_test:]
    train_dataY, test_dataY = controldata[:-num_test], controldata[-num_test:]
    #转换为张量形式
    # train_dataX = train_dataX.float()
    # train_dataY = train_dataY.float()
    # test_dataX = test_dataX.float()
    # test_dataY = test_dataY.float()

    train_dataset = MyDataset(train_dataX,train_dataY)
    test_dataset = MyDataset(test_dataX, test_dataY)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    # 训练模型
    input_size = 6  # X的特征数量
    hidden_size = 256
    output_size = 3  # Y的特征数量
    #model = GRUModel(input_size, hidden_size, output_size)
    model = RegnnModel(input_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr= 0.00001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    num_epochs = 400   #200  0.0008
    trainloss_epo = []
    testloss_epo = []
    best_model = model
    best_val_loss = float('inf')
    for epoch in range(num_epochs):

        losstrain = train(model, train_loader, optimizer)
        trainloss_epo.append(losstrain.item())
        print('Epoch [{}/{}], Loss: {:.8f}'
              .format(epoch + 1, num_epochs, losstrain.item()))
        losstest = test(model,test_loader)
        testloss_epo.append(losstest.item())
        if losstest < best_val_loss:
            best_val_loss = losstest.item()
            best_model = model
        print(best_val_loss)

    #保存网络
    save_path1 = "save_model_path"
    torch.save(best_model, save_path1)

    plt.plot(trainloss_epo)
    plt.plot(testloss_epo)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    # 测试模型
    # outputs = test(model, test_loader)

    # 做预测
    # X_predict =  # 待预测的X
    # sequence = torch.from_numpy(X_predict)
    # model.eval()
    # with torch.no_grad():
    #     pred = model(sequence)