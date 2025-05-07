import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F  
from sklearn.model_selection import train_test_split
import time



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


class lstmModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # 替换为RNN层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 类型检查与转换（保持输入处理不变）
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()
        
        # 通过RNN层
        out, _ = self.rnn(x)  # RNN输出形状：(batch, seq_len, hidden_size)
        
        # 取最后一个时间步的输出并经过线性层
        out = self.linear(out[:, -1, :])  # 保持输出形式不变
        return out


# 训练函数
def train(model, train_loader, optimizer):
    criterion = nn.MSELoss(reduction='mean')
    model.train()
    total_loss = 0.  # 累加所有批次的损失
    total_loss = torch.tensor(total_loss)
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        y = y.float()
        

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # 累加当前批次的损失
    avg_loss = total_loss / len(train_loader)    
    losscpu = avg_loss.to("cpu")

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
    # print('Test loss: {:.8f}'.format(test_loss))

    return test_loss

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #定义数据
    seq_time = 4
    #数据处理，将数据作为可以处理的形式：数据读取与数据分类，分成状态和控制
    # 数据处理，将数据作为可以处理的形式：数据读取与数据分类，分成状态和控制
    folder_paths = [
         "dataset_path"
    ]
    controldata = []  #m*seq_time=4*3
    statedata = []   #m*seq_time=4*6
    for folder_path in folder_paths:
        for file_name in os.listdir(folder_path):
            # print(file_name)
            if file_name.endswith(".mat"):
                file_path = os.path.join(folder_path, file_name)
                data = scipy.io.loadmat(file_path)   #已经得到traj_n，包括状态6+3+3
                data1 = data['trajectory']
                hang = len(data1)
                for i in range(0,hang-seq_time):
                    statedata1 = data1[i:i+seq_time,:6]
                    controldata1 = data1[i+seq_time-1,6:9]
                    controldata.append(controldata1)
                    statedata.append(statedata1)
    controldata = np.array(controldata)
    statedata = np.array(statedata)
    print(statedata.shape,controldata.shape)

    #对数据进行归一化处理，只针对输入，并记录6个状态的最值并保存
    # 求最小最大值
    X_min = np.min(statedata, axis=(0,1))
    X_max = np.max(statedata, axis=(0,1))
    
    # Min-Max规范化
    statedata = (statedata - X_min) / (X_max - X_min)
    
    # 使用train_test_split随机划分训练集和测试集
    train_dataX, test_dataX, train_dataY, test_dataY = train_test_split(
        statedata, controldata, test_size=0.1, random_state=42)

    train_dataset = MyDataset(train_dataX,train_dataY)
    test_dataset = MyDataset(test_dataX, test_dataY)

    train_loader = DataLoader(train_dataset, batch_size=1024*5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024*5)

    # 训练模型
    input_size = 6  # X的特征数量
    hidden_size = 128
    output_size = 3  # Y的特征数量
    
    model = lstmModel(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 350   #200  0.0008
    trainloss_epo = []
    testloss_epo = []
    epoch_times = []    # 保存每个epoch的时间
    best_model = model
    best_val_loss = float('inf')

    # 定义保存模型的函数
    def save_model(model, path):
        torch.save(model, path)

    for epoch in range(num_epochs):
        # 记录当前epoch的开始时间
        epoch_start_time = time.perf_counter()
        losstrain = train(model, train_loader, optimizer)
        trainloss_epo.append(losstrain.item())
        # print('Epoch [{}/{}], Loss: {:.8f}'
        #       .format(epoch + 1, num_epochs, losstrain.item()))
        losstest = test(model,test_loader)
        testloss_epo.append(losstest.item())
        if losstest < best_val_loss:
            best_val_loss = losstest.item()
            best_model = model
        # print(best_val_loss)
        # print(f'Best Validation Loss so far: {best_val_loss:.8f}')
        # 记录当前epoch的结束时间并计算耗时
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        # 每50个epoch保存一次模型
        if (epoch + 1) % 20 == 0:
            current_model_path = f"uav_epoch{epoch + 1}.pth"
            best_model_path = f"uav_best_epoch{epoch + 1}.pth"
            save_model(model, current_model_path)
            if best_model is not None:
                save_model(best_model, best_model_path)
    
    # 最后保存最佳模型和最终模型
    final_best_model_path = "uav_bestrdnn1.pth"
    final_model_path = "uav_rdnn1.pth"
    if best_model is not None:
        save_model(best_model, final_best_model_path)
    save_model(model, final_model_path)

    # 保存 trainloss_epo 和 testloss_epo 到 .mat 文件
    loss_data = {
        'trainloss_epo': trainloss_epo,
        'testloss_epo': testloss_epo,
        'epoch_times': epoch_times  # 保存每个epoch的时间
    }
    scipy.io.savemat('loss_data_rdnn.mat', loss_data)

    # 绘制并保存loss图
    plt.plot(trainloss_epo, label='Train Loss')
    plt.plot(testloss_epo, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()

    # 保存loss图到文件
    loss_plot_path = "loss/loss_rdnn.png"
    plt.savefig(loss_plot_path)
