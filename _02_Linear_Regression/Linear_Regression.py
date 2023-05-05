# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def main(data):
    x,y=read_data
    alpha = 0.1
    w1 = ridge(x, y, alpha)

    w2 = lasso(x, y, alpha)

    x_new = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    y1 = np.dot(x_new, w1)

    y2 = np.dot(x_new, w2)

    return y1,y1

def ridge(X, y, alpha):
    XtX = np.dot(X.T, X)
    n = X.shape[0]
    return np.linalg.solve(XtX + alpha * n * np.identity(X.shape[1]), np.dot(X.T, y))


def lasso(X, y, alpha, lr=0.01, max_iter=1000):
    w = np.zeros(X.shape[1])
    n = X.shape[0]
    for i in range(max_iter):

        gradient = np.dot(X.T, np.dot(X, w) - y) + alpha * np.sign(w)

        w -= lr * gradient / n
    return w

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, yy
