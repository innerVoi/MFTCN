from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings("ignore")


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
dataset = pd.read_csv('../data/data_wushan.csv')
dataset = dataset[dataset['avg_Temp']<45]
data = dataset['avg_Temp'].ffill()
actual_Y = data[-363:]


def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()

# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()

    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

# Dickey-Fuller test:
def teststationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput

# 平稳性检验方法1
ts = data
ts_log = np.log(ts)
# draw_trend(ts,7)
#通过上图，我们可以发现数据的移动平均值/标准差有越来越大的趋势，是不稳定的。

# 平稳性检验方法2
print("---------------ts--------------")
print(teststationarity(ts))

def draw_moving(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = timeSeries.ewm(halflife=size,min_periods=0,adjust=True,ignore_na=False).mean()

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Weighted Rollig Mean')
    plt.show()
# draw_moving(ts_log, 7)

def draw_acf_pacf(ts,lags):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
# draw_acf_pacf(ts_log, 7)


model = ARIMA(ts, order=(1,0,1))
result_arima = model.fit( disp=-1,)
result_arima.forecast()
predict_ts = result_arima.predict()
# print(predict_ts)

# 测试集的评估
prediction_test = predict_ts[-363:]
ts = ts[prediction_test.index]  # 过滤没有预测的记录plt.figure(facecolor='white')

prediction_test = prediction_test.tolist()
actual_Y = actual_Y.tolist()

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