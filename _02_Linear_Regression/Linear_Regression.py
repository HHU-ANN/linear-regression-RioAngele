# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X,y=read_data()
    alpha = 0.1
    XtX = np.dot(X.T, X)
    n = X.shape[0]
    return np.linalg.solve(XtX + alpha * n * np.identity(X.shape[1]), np.dot(X.T, y))


def lasso(data, lr=0.01, max_iter=1000):
    X,y=read_data()
    alpha = 0.1
    w = np.zeros(X.shape[1])
    n = X.shape[0]
    for i in range(max_iter):
        gradient = np.dot(X.T, np.dot(X, w) - y) + alpha * np.sign(w)
        w -= lr * gradient / n
    return w

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
