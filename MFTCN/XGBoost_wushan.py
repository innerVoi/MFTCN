import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb
import math

global window_size
window_size = 5
global n_classes
n_classes= 1
print(f'window_size = {window_size}, n_classes = {n_classes}')

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


print('----------------Read Data--------------')
# 读取数据
# dataset = pd.read_csv('../data/data_wushan.csv')
dataset = pd.read_csv('./data_wushan.csv')
dataset = dataset[dataset['avg_Temp']<45]
data = dataset['avg_Temp']

# 归一化数据, 划分数据集
dataX, dataY = create_data(data)
train_x = dataX[0:2950]
train_y = dataY[0:2950]
valid_x = dataX[2950:3313]
valid_y = dataY[2950:3313]
test_x = dataX[3313:]
test_y = dataY[3313:]
print(type(train_x))
print('train:%d, valid:%d, test:%d' % (len(train_x), len(valid_x), len(test_x)))

print('----------split data----------')
dtrain = xgb.DMatrix(train_x, train_y)
dvalid = xgb.DMatrix(valid_x, valid_y)
dtest = xgb.DMatrix(test_x)

params = {
    'booster': 'gbtree',
    'eval_metric': 'rmse',
    'gamma': 0.03,
    'max_depth': 7,
    'alpha': 0,
    'lambda': 0.08,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'min_child_weight': 2,
    'silent': 0,
    'eta': 0.03,
    'nthread': -1,
    'seed': 2021,
}

print('----------XGBoost----------')
watchlist = [(dtrain,'train'), ((dvalid, 'valid'))]
model = xgb.train(params, dtrain, num_boost_round=20000, evals=watchlist,
                  verbose_eval=1000, early_stopping_rounds=500)
#输出概率
prediction_test = model.predict(dtest)
actual_Y = test_y.flatten().tolist()
# print(len(actual_Y), len(prediction_test))

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