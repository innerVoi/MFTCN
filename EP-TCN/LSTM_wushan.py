
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import math
import torch.nn.functional as F

global window_size
window_size = 7
global n_classes
n_classes= 1

# 归一化
def normalization(dataset):
    global scalar
    min_value = np.min(dataset)
    max_value = np.max(dataset)
    scalar = max_value - min_value
    return list(map(lambda x: x / scalar, dataset))

def MSE(y1, y2):
    L = len(y1)
    S_error = 0
    for i in range(L):
        S_error += (y1[i] - y2[i])**2
    return S_error / L

# 计算均方根误差
def RMSE(y1, y2):
    L = len(y1)
    S_error = 0
    for i in range(L):
        S_error += (y1[i] - y2[i])**2
    return math.sqrt(S_error / L)

# 计算平均绝对百分比误差
def MAPE(y1, y2):
    L = len(y1)
    S_error = 0
    for i in range(L):
        S_error += (abs((y1[i]-y2[i])/y1[i]) * (100 / L))
    return S_error

# 计算平均绝对误差
def MAE(y1, y2):
    L = len(y1)
    S_error = 0
    for i in range(L):
        S_error += abs(y1[i]-y2[i])
    return S_error / L

# 计算拟合优度
def R_square(y1, y2):
    L = len(y1)
    m = 0
    for i in range(L):
        m += y1[i]
    m /= L
    s1, s2 = 0, 0
    for i in range(L):
        s1 += (y1[i]-y2[i])**2
        s2 += (y1[i]-m)**2
    return 1 - (s1 / s2)

# 划分训练集和验证集
def create_data(dataset, window=window_size, pred_step=n_classes):
    dataX, dataY = [], []
    for i in range(len(dataset) - window - pred_step + 1):
        x = dataset[i:i + window]
        dataX.append(x)
        dataY.append(dataset[i + window: i + window + pred_step])
    return np.array(dataX), np.array(dataY)

print('----------------1. Producing Data--------------')
# 读取数据
# dataset = pd.read_csv('../data/data_wushan.csv')
dataset = pd.read_csv('./data_wushan.csv')
dataset = dataset[dataset['avg_Temp']<45]
data = dataset['avg_Temp']

# 归一化数据
# data = normalization(data)
data_length = len(data)
print(data_length)

# 划分数据集
dataX, dataY = create_data(data)
print(len(dataX), len(dataY))
test_Y = dataY[3313:]

# 划分train, valid, test
trainX = dataX[0:2950]
trainY = dataY[0:2950]
validX = dataX[2950:3313]
validY = dataY[2950:3313]
testX = dataX[3313:]
testY = dataY[3313:]
print('train:%d, valid:%d, test:%d' % (len(trainY), len(validX), len(testY)))

# 改变数据形状
trainX = trainX.reshape(-1, 1, window_size)
# trainY = trainY.reshape(-1, n_classes, 1)
trainY = trainY.reshape(-1, 1, n_classes)
validX = validX.reshape(-1, 1, window_size)
# validY = validY.reshape(-1, n_classes, 1)
validY = validY.reshape(-1, 1, n_classes)
testX = testX.reshape(-1, 1, window_size)
# testY = testY.reshape(-1, n_classes, 1)
testY = testY.reshape(-1, 1, n_classes)

# 将数据转化为张量
trainX = torch.from_numpy(trainX)
trainY = torch.from_numpy(trainY)
validX = torch.from_numpy(validX)
validY = torch.from_numpy(validY)
testX = torch.from_numpy(testX)
testY = torch.from_numpy(testY)

print('----------------2. LSTM--------------')
# LSTM网络
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=n_classes, num_layers=2):
        # super用于多层继承使用
        super(lstm_reg, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # 两层网络
        self.reg = nn.Linear(hidden_size, output_size)  # 将隐层的输出向量作为输入向量，回归到output_size维度的输出向量

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

# 模型训练
epochs = 1000
lstm = lstm_reg(window_size, 16)
lstm = lstm.cuda(1)
# criterion = nn.MSELoss().cuda()
# criterion = F.mse_loss().cuda()
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-2)
for ep in range(epochs):
    tra_x = trainX.float().cuda(1)
    tra_y = trainY.float().cuda(1)
    out = lstm(tra_x)
    loss = F.mse_loss(out, tra_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (ep + 1) % 50 == 0:
        print('epoch: %d, loss: %.5f' % (ep + 1, loss.item()))
# torch.save(lstm.state_dict(), '../model/lstm_{}.pkl'.format(epochs))


prediction_train = lstm(trainX.float().cuda(1)).cpu().view(-1).data.numpy()
prediction_valid = lstm(validX.float().cuda(1)).cpu().view(-1).data.numpy()
prediction_test = lstm(testX.float().cuda(1)).cpu().view(-1).data.numpy().tolist()
# print(prediction_test)
print(len(prediction_test))
# print(test_Y)
actual_Y = test_Y.flatten().tolist()
print(len(actual_Y))
# print(actual_Y)
final_test_mse_imf = MSE(actual_Y, prediction_test)
print('final test mse={}'.format(final_test_mse_imf))

final_test_rmse = RMSE(actual_Y, prediction_test)
print('final test rmse={}'.format(final_test_rmse))

final_test_mae = MAE(actual_Y, prediction_test)
print('final test mae={}'.format(final_test_mae))

final_test_mape = MAPE(actual_Y, prediction_test)
print('final test mape={}'.format(final_test_mape))

final_test_r = R_square(actual_Y, prediction_test)
print('final test r={}'.format(final_test_r))

# # 计算训练误差
# lstm.load_state_dict(torch.load('../model/lstm_{}.pkl'.format(epochs)))
# train_out = lstm(trainX.float().cuda())
# train_pred = train_out.cpu().view(-1).data.numpy()
# yTrain =  trainY.view(-1).data.numpy() * scalar
# train_pred = train_pred * scalar
# mse_Train = MSE(train_pred, yTrain)
#
# # 计算验证误差
# validX = validX.float().cuda()
# valid_out = lstm(validX)
# valid_pred = valid_out.cpu().view(-1).data.numpy()
# yValid = validY * scalar
# valid_pred = valid_pred * scalar
# mse_Valid = MSE(yValid, valid_pred)
#
# # 计算测试误差
# testX = testX.float().cuda()
# test_out = lstm(testX)
# test_pred = test_out.cpu().view(-1).data.numpy()
# yTest = testY * scalar
# test_pred = test_pred * scalar
# mse_Test = MSE(yTest, test_pred)
# print('mse_Train={}, mse_Valid={}, mse_Test={}'.format(train_loss, valid_loss, test_loss))
# # 测试集预测和实际绘图
# plt.plot(yTest, color = 'blue', label='actual')
# plt.plot(test_pred, color = 'red', label='LSTM')
# plt.legend()
# plt.show()
# plt.savefig('../picture/LSTM.png')
# result = pd.DataFrame(test_pred)
# result.to_csv('../comparison/LSTM_pred.csv', index=False)