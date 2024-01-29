import numpy as np
from PyEMD import EEMD, EMD, Visualisation
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import math
import torch.nn.functional as F

# 读取数据
torch.cuda.set_device(1)
global window_size
window_size = 7
global n_classes
n_classes= 7
print(f'window_size = {window_size}, n_classes = {n_classes}')
# dataset = pd.read_csv('../data/data_wushan.csv')
dataset = pd.read_csv('./data_wushan.csv')
dataset = dataset[dataset['avg_Temp']<45]
data = dataset['avg_Temp']

# EEMD分解
def eemd_data(dataset):
    max_imf = -1
    global E_IMFs

    tMin, tMax = 0, dataset.shape[0] - 1
    T = np.linspace(tMin, tMax, dataset.shape[0])
    data = list(dataset)
    eemd = EEMD()
    eemd.trials = 100
    eemd.noise_seed(2021)
    E_IMFs = eemd.eemd(data, T, max_imf)
    # print(E_IMFs)
    # print(len(E_IMFs[1]))
    imfNo = E_IMFs.shape[0]

    c = 1
    r = imfNo + 1
    # 画图展示
    #plt.ioff()
    #plt.figure(figsize=(8, 15))
    #plt.subplot(r, c, 1)
    #plt.plot(T, data, 'r')
    #plt.xlim((tMin, tMax))
    #plt.title("Original signal")
    #plt.tight_layout()

    #for num in range(imfNo):
    #    plt.subplot(r, c, num + 2)
    #    plt.plot(T, E_IMFs[num], 'g')
    #    plt.xlim((tMin, tMax))
    #    plt.title("Imf " + str(num + 1))
    #    plt.tight_layout()
    #plt.savefig('../picture/eemd.png')
    #plt.close()

# 划分训练集和验证集
def create_data(dataset, window=window_size, pred_step=n_classes):
    dataX, dataY = [], []
    for i in range(len(dataset) - window - pred_step + 1):
        x = dataset[i:i + window]
        dataX.append(x)
        dataY.append(dataset[i + window: i + window + pred_step])
    return np.array(dataX), np.array(dataY)

# 计算均方根误差
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


if __name__ == '__main__':
    X, Y = create_data(data)
    test_Y = Y[3313:]
    print(len(test_Y))
    eemd_data(data)
    print(len(E_IMFs))
    global prediction_test
    prediction_test = [0 for i in range(len(test_Y) * n_classes)]
    # print(len(prediction_test))
    for i in range(len(E_IMFs)):
        print('-' * 24)
        print('this is the %d IMF' % i)
        data_i = E_IMFs[i]
        # data_i = normalization(data_i)
        x, y = create_data(data_i)
        # 划分训练集、验证集、测试集
        trainX = x[0:2950]
        trainY = y[0:2950]
        validX = x[2950:3313]
        validY = y[2950:3313]
        testX = x[3313:]
        testY = y[3313:]
        # 改变数据形状
        trainX = trainX.reshape(-1, 1, window_size)
        trainY = trainY.reshape(-1, 1, n_classes)
        validX = validX.reshape(-1, 1, window_size)
        validY = validY.reshape(-1, 1, n_classes)
        testX = testX.reshape(-1, 1, window_size)
        testY = testY.reshape(-1, 1, n_classes)

        # 将数据转化为张量
        trainX = torch.from_numpy(trainX)
        trainY = torch.from_numpy(trainY)
        validX = torch.from_numpy(validX)
        validY = torch.from_numpy(validY)
        testX = torch.from_numpy(testX)
        testY = torch.from_numpy(testY)

        lstm = lstm_reg(window_size, 16)
        lstm = lstm.cuda()
        criterion = nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=5e-3)

        iterations = 600 - i * 50
        lstm.train()
        for epoch in range(iterations):
            trainX = trainX.float().cuda()
            trainY = trainY.float().cuda()
            validX = validX.float().cuda()
            validY = validY.float().cuda()
            optimizer.zero_grad()
            out = lstm(trainX)
            loss = criterion(out, trainY)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
               print('epoch: %d, loss: %.5f' % (epoch + 1, loss.item()))
        # name = 'eemd-lstm_iterations_' + str(i) + '.pkl'
        # torch.save(lstm.state_dict(), '../model/EEMD_LSTM_Models/' + name)
        #
        # lstm.load_state_dict(torch.load('../model/EEMD_LSTM_Models/' + name))
        # 验证集误差计算
        lstm.eval()
        valid_pred = lstm(validX)
        valid_loss = F.mse_loss(valid_pred, validY)
        # pred_v = valid_pred.cpu().view(-1).data.numpy()

        testX = testX.float().cuda()
        testY = testY.float().cuda()
        test_pred = lstm(testX)
        test_loss = F.mse_loss(test_pred, testY)

        print(f'the {i}-th IMF: valid)loss = {valid_loss}, test_loss = {test_loss}')

        # actual_test = [actual_test[i] + testY[i] for i in range(len(actual_test))]
        test_pred = test_pred.cpu().view(-1).data.numpy()
        # print(testY)
        # print(test_pred)
        prediction_test = [prediction_test[i] + test_pred[i] for i in range(len(prediction_test))]
        # print(' actual_valid[0]={}, prediction_valid[0]={}\n actual_test[0]={}, prediction_test[0]={}'.format(actual_valid[0], prediction_valid[0], actual_test[0], prediction_test[0]))

        #plt.plot(testY, color='blue', label='test' + str(i))
        #plt.plot(pred_t, color='red', label='pred_t' + str(i))
        #plt.legend()
        #plt.savefig('../picture/EEMD-LSTM/test/iterations_{}.png'.format(i))
        #plt.close()
        # if i==7:
        #     print('valid_IMF_errors：{}, test_IMF_errors:{}'.format(valid_IMF_errors, test_IMF_errors))

    actual_Y = test_Y.flatten().tolist()
    # print(actual_Y)
    # print(prediction_test)
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

    # final_valid_mse = MSE(actual_valid, prediction_valid)
    # plt.plot(actual_valid, color='blue', label='actual_valid')
    # plt.plot(prediction_valid, color='red', label='prediction_valid')
    # plt.legend()
    # plt.title('EEMD-LSTM_valid MSE: %.5f' % final_valid_mse)
    # plt.savefig('../picture/EEMD-LSTM/valid/EEMD-LSTM_valid.png')
    # plt.close()
    #
    # final_test_mse = MSE(actual_test, prediction_test)
    # plt.plot(actual_test, color='blue', label='actual_test')
    # plt.plot(prediction_test, color='red', label='prediction_test')
    # plt.legend()
    # plt.title('EEMD-LSTM_test MSE: %.5f' % final_test_mse)
    # plt.savefig('../picture/EEMD-LSTM/test/EEMD-LSTM_test.png')
    # plt.close()
    # result = pd.DataFrame(prediction_test)
    # result.to_csv('../comparison/EEMD-LSTM_pred.csv',index=False)