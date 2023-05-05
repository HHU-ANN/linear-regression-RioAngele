# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    X,y = read_data()
    alpha= 0.1
    # weight = np.dot(np.linalg.inv((np.dot(x.T,x)+np.dot(alpha,np.eye(6)))),np.dot(x.T,y))
    XtX = np.dot(X.T, X)
    n = X.shape[0]
    return data @ np.linalg.solve(XtX + alpha * n * np.identity(X.shape[1]), np.dot(X.T, y))
    # return weight @ data


def lasso(data):
    x, Y = read_data()
    weight = data
    y = np.dot(weight, x.T)
    alpha = 3
    rate = 0.000000079824
    weight =model_lasso(x,Y,alpha,rate,weight)
    return weight @ data



def model_lasso(X,y,alpha,rate,weight):
    # for i in range(int(186452)):
    #     y = np.dot(weight, x.T)
    #     dw = np.dot(y - Y, x) + alpha * np.sign(weight)
    #     weight = weight * (1 - (rate * alpha / 6)) - dw * rate
    n = X.shape[0]
    for i in range(8000):
        # 计算梯度
        gradient = np.dot(X.T, np.dot(X, weight) - y) + alpha * np.sign(weight)
        # 更新权重
        weight -= rate * gradient / n
    return weight

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
