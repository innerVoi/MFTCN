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

parser = argparse.ArgumentParser(description='Sequence Modeling - Air Temperature Forecasting')
# parser.add_argument('--batch_size', type=int, default=4, metavar='N',
#                     help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
# parser.add_argument('--epochs', type=int, default=100,
#                     help='upper epoch limit (default: 50)')
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
torch.cuda.set_device(0)
global window_size
window_size = 7
global n_classes
n_classes= 5
print(f'window_size = {window_size}, n_classes = {n_classes}')
# batch_size = args.batch_size
#seq_length = args.seq_len
# epochs = args.epochs
print(args)

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


# 读取数据
print("----------------Read data----------------")
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


def tcn_train(epochs, batch_size, X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
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
    print('cur_epochs={}, cur_batch={}'.format(epochs, batch_size))
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
                    print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                        epoch, processed, X_train.size(0), 100. * processed / X_train.size(0), lr, cur_loss))
                total_loss = 0
    model.eval()
    with torch.no_grad():
        output_train = model(X_train)
        output_valid = model(X_valid)
        output_test = model(X_test)
        train_loss = F.mse_loss(output_train, Y_train)
        valid_loss = F.mse_loss(output_valid, Y_valid)
        test_loss = F.mse_loss(output_test, Y_test)
        print('Train loss: {:.6f}, Valid loss: {:.6f}, Test loss: {:.6f}\n'.format(train_loss.item(), \
                valid_loss.item(), test_loss.item()))
        return (train_loss.item() + valid_loss.item()) / 2

def tcn(epochs, batch_size, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, plot_i):
    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    print(f'epoch={epochs}, batch_size={batch_size}')
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

        return Y_test.cpu().view(-1).data.numpy(), output_test.cpu().view(-1).data.numpy()

## 2. PSO优化算法
class PSO(object):
    def __init__(self, particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value, plot_i):
        '''参数初始化
        particle_num(int):粒子群的粒子数量
        particle_dim(int):粒子维度，对应待寻优参数的个数
        iter_num(int):最大迭代次数
        c1(float):局部学习因子，表示粒子移动到该粒子历史最优位置(pbest)的加速项的权重
        c2(float):全局学习因子，表示粒子移动到所有粒子最优位置(gbest)的加速项的权重
        w(float):惯性因子，表示粒子之前运动方向在本次方向上的惯性
        max_value(float):参数的最大值
        min_value(float):参数的最小值
        '''
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.c1 = c1  ##通常设为2.0
        self.c2 = c2  ##通常设为2.0
        self.w = w
        self.max_value = max_value
        self.min_value = min_value
        self.plot_i = plot_i

    ### 2.1 粒子群初始化
    def swarm_origin(self):
        '''粒子群初始化
        input:self(object):PSO类
        output:particle_loc(list):粒子群位置列表
               particle_dir(list):粒子群方向列表
        '''
        particle_loc = []
        particle_dir = []
        for i in range(self.particle_num):
            tmp1 = []
            tmp2 = []
            for j in range(self.particle_dim):
                a = random.random()
                b = random.random()
                tmp1.append(int(a * (self.max_value - self.min_value) + self.min_value))
                tmp2.append(b)
            particle_loc.append(tmp1)
            particle_dir.append(tmp2)

        return particle_loc, particle_dir

    ## 2.2 计算适应度函数数值列表;初始化pbest_parameters和gbest_parameter
    def fitness(self, particle_loc):
        '''计算适应度函数值
        input:self(object):PSO类
              particle_loc(list):粒子群位置列表
        output:fitness_value(list):适应度函数值列表
        '''

        fitness_value = []
        ### 1.适应度函数
        for i in range(self.particle_num):
            mse_final = tcn_train(int(particle_loc[i][0]), int(particle_loc[i][1]), X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
            fitness_value.append(mse_final)
        ### 2. 当前粒子群最优适应度函数值和对应的参数
        current_fitness = 100.0
        current_parameter = []
        for i in range(self.particle_num):
            if current_fitness > fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]

        return fitness_value, current_fitness, current_parameter

    ## 2.3  粒子位置更新
    def updata(self, particle_loc, particle_dir, gbest_parameter, pbest_parameters):
        '''粒子群位置更新
        input:self(object):PSO类
              particle_loc(list):粒子群位置列表
              particle_dir(list):粒子群方向列表
              gbest_parameter(list):全局最优参数
              pbest_parameters(list):每个粒子的历史最优值
        output:particle_loc(list):新的粒子群位置列表
               particle_dir(list):新的粒子群方向列表
        '''
        ## 1.计算新的量子群方向和粒子群位置
        for i in range(self.particle_num):
            a1 = [x * self.w for x in particle_dir[i]]
            a2 = [y * self.c1 * random.random() for y in
                  list(np.array(pbest_parameters[i]) - np.array(particle_loc[i]))]
            a3 = [z * self.c2 * random.random() for z in list(np.array(gbest_parameter) - np.array(particle_dir[i]))]
            particle_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
            #            particle_dir[i] = self.w * particle_dir[i] + self.c1 * random.random() * (pbest_parameters[i] - particle_loc[i]) + self.c2 * random.random() * (gbest_parameter - particle_dir[i])
            particle_loc[i] = list(np.array(particle_loc[i]) + np.array(particle_dir[i]))

        ## 2.将更新后的量子位置参数固定在[min_value,max_value]内
        ### 2.1 每个参数的取值列表
        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        ### 2.2 每个参数取值的最大值、最小值、平均值
        value = []
        for i in range(self.particle_dim):
            tmp2 = []
            tmp2.append(max(parameter_list[i]))
            tmp2.append(min(parameter_list[i]))
            value.append(tmp2)

        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                particle_loc[i][j] = (particle_loc[i][j] - value[j][1]) / (value[j][0] - value[j][1]) * (
                        self.max_value - self.min_value) + self.min_value

        return particle_loc, particle_dir

    ## 2.4 画出适应度函数值变化图
    def plot(self, results):
        '''画图
        '''
        X = []
        Y = []
        for i in range(self.iter_num):
            X.append(i + 1)
            Y.append(results[i])
        plt.plot(X, Y)
        plt.xlabel('Number of iteration')
        plt.ylabel('(train_mse+valid_mse) / 2')
        plt.title('PSO-TCN IMF {}'.format(self.plot_i))
        plt.savefig('./figures/wushan_imf_{}.png'.format(self.plot_i))

    ## 2.5 主函数
    def main(self):
        '''主函数
        '''
        results = []
        best_fitness = 100.0
        ## 1、粒子群初始化
        particle_loc, particle_dir = self.swarm_origin()
        ## 2、初始化gbest_parameter、pbest_parameters、fitness_value列表
        ### 2.1 gbest_parameter
        gbest_parameter = []
        for i in range(self.particle_dim):
            gbest_parameter.append(0.0)
        ### 2.2 pbest_parameters
        pbest_parameters = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                tmp1.append(0.0)
            pbest_parameters.append(tmp1)
        ### 2.3 fitness_value
        fitness_value = []
        for i in range(self.particle_num):
            fitness_value.append(100.0)

        ## 3.迭代
        for i in range(self.iter_num):
            ### 3.1 计算当前适应度函数值列表
            current_fitness_value, current_best_fitness, current_best_parameter = self.fitness(particle_loc)
            ### 3.2 求当前的gbest_parameter、pbest_parameters和best_fitness
            for j in range(self.particle_num):
                if current_fitness_value[j] < fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter

            print('\n iteration is :', i + 1, ';Best parameters:', gbest_parameter, ';Best fitness', best_fitness, '\n\n')
            results.append(best_fitness)
            ### 3.3 更新fitness_value
            fitness_value = current_fitness_value
            ### 3.4 更新粒子群
            particle_loc, particle_dir = self.updata(particle_loc, particle_dir, gbest_parameter, pbest_parameters)
            # print(particle_loc,'\n',particle_dir)

        ## 4.结果展示
        # results.sort()
        # results.reverse()
        self.plot(results)
        print('Final parameters are :', gbest_parameter)
        return round(gbest_parameter[0]), round(gbest_parameter[1])

if __name__ == '__main__':
    print('----------------2.Parameter Seting------------')
    particle_num = 10
    particle_dim = 2
    iter_num = 10
    c1 = 2
    c2 = 2
    w = 0.8
    max_value = 100
    min_value = 10
    print('particle_num={}, particle_dim={}, iter_num={}, c1={}, c2={}, w={}, max_value={}, min_value={}' \
          .format(particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value))
    print('----------------3.PSO_TCN-----------------')
    X, Y = create_data(data)
    test_Y = Y[3313:]
    print(len(test_Y))
    eemd_data(data)
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

        pso = PSO(particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value, i)
        epoch, batch_size = pso.main()

        # 学习当前的imf_i
        actual_i, pred_i = tcn(epoch, batch_size, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, i)
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