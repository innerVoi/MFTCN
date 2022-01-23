import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from PyEMD import EEMD
from torch import nn
# sys.path.append("../../")
sys.path.append("./")
from model import TCN
import math

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit (default: 50)')
parser.add_argument('--ksize', type=int, default=2,
                    help='kernel size (default: 2)')
parser.add_argument('--levels', type=int, default=2,
                    help='# of levels (default: 1)')
#parser.add_argument('--seq_len', type=int, default=400,
#                    help='sequence length (default: 400)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

input_channels = 1
torch.cuda.set_device(1)
global window_size
window_size = 7
global n_classes
n_classes= 7
print(f'window_size = {window_size}, n_classes = {n_classes}')
batch_size = args.batch_size
#seq_length = args.seq_len
epoch = args.epochs
print(args)
# 读取数据
print("----------------Read data----------------")
# dataset = pd.read_csv('../data/data_wushan.csv')
dataset = pd.read_csv('./data_wushan.csv')
dataset = dataset[dataset['avg_Temp']<45]
data = dataset['avg_Temp']

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

    # c = 1
    # r = imfNo + 1
    # # 画图展示
    # plt.ioff()
    # plt.figure(figsize=(8, 15))
    # plt.subplot(r, c, 1)
    # plt.plot(T, data, 'r')
    # plt.xlim((tMin, tMax))
    # plt.title("Original signal")
    # plt.tight_layout()
    #
    # for num in range(imfNo):
    #     plt.subplot(r, c, num + 2)
    #     plt.plot(T, E_IMFs[num], 'g')
    #     plt.xlim((tMin, tMax))
    #     plt.title("Imf " + str(num + 1))
    #     plt.tight_layout()
    # plt.savefig('./figures/eemd-pso-tcn/eemd.png')
    # plt.close()


def tcn_train(epochs, batch_size, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, plot_i):
    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
    if args.cuda:
        model.cuda()
        X_train = X_train.float().cuda()
        Y_train = Y_train.float().cuda()
        X_valid = X_valid.float().cuda()
        Y_valid = Y_valid.float().cuda()
        X_test = X_test.float().cuda()
        Y_test = Y_test.float().cuda()
    global lr
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    print('-'*24)
    # print('cur_epochs={}, cur_batch={}'.format(epochs, batch_size))
    for epoch in range(1, epochs + 1):
        model.train()
        batch_idx = 1
        total_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            if i + batch_size > X_train.size(0):
                x, y = X_train[i:], Y_train[i:]
            else:
                x, y = X_train[i:(i + batch_size)], Y_train[i:(i + batch_size)]
            # print(y.shape)
            optimizer.zero_grad()
            output = model(x)
            # print(output.shape)
            loss = F.mse_loss(output, y)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            batch_idx += 1
            total_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                cur_loss = total_loss / args.log_interval
                processed = min(i + batch_size, X_train.size(0))
                if epoch % 20 == 0:
                    print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, processed, X_train.size(0), 100. * processed / X_train.size(0), cur_loss))
                total_loss = 0
    model.eval()
    with torch.no_grad():
        output_train = model(X_train)
        output_valid = model(X_valid)
        output_test = model(X_test)
        train_loss = F.mse_loss(output_train, Y_train)
        valid_loss = F.mse_loss(output_valid, Y_valid)
        test_loss = F.mse_loss(output_test, Y_test)
        print('Train loss: {:.6f}, Valid loss: {:.6f}, Test loss: {:.6f}\n'.format(train_loss.item(), valid_loss.item(), test_loss.item()))

        # pred_t = output_test.cpu().view(-1).data.numpy() * scalar
        # actual_t = Y_test.cpu().view(-1).data.numpy() * scalar
        # plt.plot(actual_t, color='blue', label= 'actual_test')
        # plt.plot(pred_t, color='red', label='EEMD-PSO-TCN')
        # plt.title('IMF {} MSE:{}'.format(plot_i, MSE(pred_t, actual_t)))
        # plt.legend()
        # plt.savefig('./figures/eemd-pso-tcn/EEMD-PSO-TCN_pred_{}.png'.format(plot_i))
        # plt.close()

        return Y_test.cpu().view(-1).data.numpy(), output_test.cpu().view(-1).data.numpy()


if __name__ == '__main__':
    print('----------------EEMD_TCN-----------------')
    X, Y = create_data(data)
    test_Y = Y[3313:]
    print(len(test_Y))
    eemd_data(data)
    print(len(E_IMFs))
    test_IMF_errors = []
    global prediction_test
    prediction_test = [0 for i in range(len(test_Y) * n_classes)]
    for i in range(len(E_IMFs)):
        print('-' * 24)
        print('this is the %d IMF' % i)
        data_i = E_IMFs[i]
        dataX, dataY = create_data(data_i)
        # 划分train, valid, test
        X_train = dataX[0:2950]
        Y_train = dataY[0:2950]
        X_valid = dataX[2950:3313]
        Y_valid = dataY[2950:3313]
        X_test = dataX[3313:]
        Y_test = dataY[3313:]
        # 改变数据形状
        X_train = X_train.reshape(-1, 1, window_size)
        Y_train = Y_train.reshape(-1, n_classes, 1)
        X_valid = X_valid.reshape(-1, 1, window_size)
        Y_valid = Y_valid.reshape(-1, n_classes, 1)
        X_test = X_test.reshape(-1, 1, window_size)
        Y_test = Y_test.reshape(-1, n_classes, 1)
        # 将数据转化为张量
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_valid = torch.from_numpy(X_valid)
        Y_valid = torch.from_numpy(Y_valid)
        X_test = torch.from_numpy(X_test)
        Y_test = torch.from_numpy(Y_test)

        # 学习当前的imf_i
        actual_i, pred_i = tcn_train(epoch, batch_size, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, i)
        prediction_test = [prediction_test[i] + pred_i[i] for i in range(len(prediction_test))]

    actual_Y = test_Y.flatten().tolist()
    print(len(actual_Y), len(prediction_test))

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
