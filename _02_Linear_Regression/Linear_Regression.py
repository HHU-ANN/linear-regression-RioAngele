# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x,y = read_data()
    alpha= -0.1
    weight = np.dot(np.linalg.inv((np.dot(x.T,x)+np.dot(alpha,np.eye(6)))),np.dot(x.T,y))
    return weight @ data

 

def lasso(data):
    x, Y = read_data()
    weight = data
    y = np.dot(weight, x.T)
    alpha = 3000
    rate = e-6
    weight =model_lasso(x,Y,alpha,rate,weight)
    return weight @ data

def model_lasso(x,Y,alpha,rate,weight):
    for i in range(int(2e5)):
        y = np.dot(weight, x.T)
        dw = np.dot(y - Y, x) + alpha * np.sign(weight)
        weight = weight * (1 - (rate * alpha / 6)) - dw * rate
    
    return weight

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
