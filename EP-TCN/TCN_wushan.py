import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch import nn
# sys.path.append("../../")
sys.path.append("./")
from model import TCN
import  math

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 50)')
parser.add_argument('--ksize', type=int, default=2,
                    help='kernel size (default: 2)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
#parser.add_argument('--seq_len', type=int, default=400,
#                    help='sequence length (default: 400)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=2022,
                    help='random seed (default: 2022)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

torch.cuda.set_device(1)
input_channels = 1
global window_size
window_size = 7
global n_classes
n_classes= 7
print(f'window_size = {window_size}, n_classes = {n_classes}')

batch_size = args.batch_size
#seq_length = args.seq_len
epochs = args.epochs
print(args)
print("Producing data...")


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


# 归一化
def normalization(dataset):
    global scalar
    min_value = np.min(dataset)
    max_value = np.max(dataset)
    scalar = max_value - min_value
    return list(map(lambda x: x / scalar, dataset))


# 读取数据
# dataset = pd.read_csv('../data/data_wushan.csv')
dataset = pd.read_csv('./data_wushan.csv')
dataset = dataset[dataset['avg_Temp']<45]
data = dataset['avg_Temp']
# data = normalization(data)
data_length = len(data)

dataX, dataY = create_data(data)
print(len(dataX), len(dataY))
test_Y = dataY[3313:]
print(len(test_Y))

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

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
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

#print(X_train.shape,'\n',Y_train.shape)
#print(X_test.shape,'\n',Y_test.shape)

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        if i + batch_size > X_train.size(0):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        #print(y.shape)
        optimizer.zero_grad()
        output = model(x)
        #print(output.shape)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
            total_loss = 0


def evaluate():
    model.eval()
    with torch.no_grad():
        output = model(X_valid)
        valid_loss = F.mse_loss(output, Y_valid)
        print('\nValid set: Average loss: {:.6f}\n'.format(valid_loss.item()))
        return valid_loss.item()


def test():
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.mse_loss(output, Y_test)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))

        prediction_test = output.cpu().view(-1).data.numpy().tolist()
        # b = Y_test.cpu().view(-1).data.numpy()
        # a = a.tolist()
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



        # tcn_result = pd.DataFrame(a)
        # tcn_result.to_csv('../../comparison/tcn_pred.csv',index=False)
        # b = b.tolist()
        # print(b[0:10],'\n',a[0:10])
        # mse = MSE(a, b)
        # plt.plot(b, color='#FA8072', label='actual_test')
        # plt.plot(a, color='#00FFFF', label='TCN', linestyle='--')
        # plt.title('TCN mse:%.5f' % (mse))
        # plt.legend()
        # plt.savefig('./figures/TCN_pred_1.png')
        # plt.close()
        return test_loss.item()


for ep in range(1, epochs+1):
    train(ep)
    vloss = evaluate()

tloss = test()
