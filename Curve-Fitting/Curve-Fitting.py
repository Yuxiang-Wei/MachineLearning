import numpy as np
import matplotlib.pyplot as plt
import math


def load_data(order, data_num, start=0, end=1):
    """
    给定阶数和样本数,生成从start到end去年上的指定数目的样本点
    :param order: 多项式阶数
    :param data_num: 样本点数目
    :param start: 区间起点
    :param end: 区间终点
    :return: 返回生成数据集, 矩阵,形状(data_num, order+2)
    """
    x = np.arange(start, end, 1.0 * (end - start) / data_num)
    y = [math.sin(2 * math.pi * z / (end - start)) + np.random.normal(0, 0.1) for z in x]
    x_vec = np.mat(x).T
    data_set = np.ones(np.shape(x)[0])
    for i in range(1, order + 1, 1):
        data_set = np.column_stack((data_set, np.power(x_vec, i)))
    data_set = np.column_stack((data_set, np.mat(y).T))
    np.random.shuffle(data_set)  # 打乱数据
    return data_set


# 计算给定参数和样本情况下的loss值, 此时的loss未开根号
def get_loss(w, x, y, _lambda=0.0):
    hx = np.dot(x, w) - y
    loss = (0.5 * np.dot(hx.T, hx) + _lambda * 0.5 * np.dot(w.T, w)) / np.shape(y)[0]
    return loss


def get_analytical_solution(x_train, y_train):
    """
    求取解析解
    :param x_train: 训练集的x, 矩阵, 每行形如 [1, x, x^2, ... x^n]
    :param y_train: 训练集的y, 列向量
    :return: 返回求解得带的参数w, 列向量
    """
    w = np.dot(np.dot(np.dot(x_train.T, x_train).I, x_train.T), y_train)
    return w


def get_analytical_solution_with_regular(x_train, y_train, _lambda=0.1):
    """
    带正则项的解析解
    :param x_train: 训练集的x, 矩阵, 每行形如 [1, x, x^2, ... x^n]
    :param y_train: 训练集的y, 列向量
    :param _lambda: 正则项的lambda系数
    :return: 返回求解得带的参数w, 列向量
    """
    order = np.shape(x_train)[1] - 1
    w = np.dot(np.dot((np.dot(x_train.T, x_train) + _lambda * np.eye(order + 1)).I, x_train.T), y_train)
    return w


def gradient_descent(x_train, y_train, lr=1e-6, epochs=100000, _lambda=0.0):
    """
    梯度下降拟合
    :param x_train: 训练集的x, 矩阵, 每行形如 [1, x, x^2, ... x^n]
    :param y_train: 训练集的y, 列向量
    :param lr: 学习率
    :param epochs: 迭代轮数
    :param _lambda: 正则项的lambda系数, 默认不含正则项
    :return: 返回求解得带的参数w, 列向量
    """
    order = np.shape(x_train)[1] - 1
    train_num = np.shape(x_train)[0]
    w = np.random.normal(0, 0.01, size=order + 1)
    w = np.mat(w).T
    for epoch in range(epochs):
        hx = np.dot(x_train, w) - y_train
        loss0 = get_loss(w, x_train, y_train, _lambda)
        w = w - lr / train_num * (np.dot(x_train.T, hx) + _lambda * w)
        loss1 = get_loss(w, x_train, y_train, _lambda)
        if loss0 < loss1:  # 学习率衰减
            lr = lr / 10
        if (epoch + 1) % 1000 == 0:  # 每隔1000轮输出一次loss
            print('loss = %f, order = %d, train_num = %d, epoch = %d, lr = %e, lambda = %f' % (
                loss1, order, train_num, epoch + 1, lr, _lambda))
    return w


def newton_method(x_train, y_train, max_epoch=10000, epsilon=1e-4, _lambda=0.0):
    """
    牛顿法求解
    :param x_train: 训练集的x, 矩阵, 每行形如 [1, x, x^2, ... x^n]
    :param y_train: 训练集的y, 列向量
    :param max_epoch: 最大迭代轮数,若到达该轮还未到达min_loss, 返回
    :param epsilon: 精度阈值
    :param _lambda: 正则项的lambda系数, 默认不含正则项
    :return: 返回求解得带的参数w, 列向量
    """
    order = np.shape(x_train)[1] - 1
    w = np.random.normal(0, 0.01, size=order + 1)
    w = np.mat(w).T
    for epoch in range(max_epoch):
        loss0 = get_loss(w, x_train, y_train, _lambda)
        # 最外层的np.dot()前半部分为二阶导数的逆, 后半部分为一阶导数
        w = w - np.dot((np.dot(x_train.T, x_train) + _lambda * np.eye(order + 1)).I,
                       np.dot(x_train.T, np.dot(x_train, w) - y_train) + _lambda * w)
        loss1 = get_loss(w, x_train, y_train, _lambda)
        if math.fabs(loss1 - loss0) < epsilon:
            return w  # 若满足给定精度阈值，则返回
    return w


def conjugate_gradient_method(x_train, y_train, max_epoch=10000, min_loss=1e-6, _lambda=0.0):
    """
    共轭梯度法求解
    :param x_train: 训练集的x, 矩阵, 每行形如 [1, x, x^2, ... x^n]
    :param y_train: 训练集的y, 列向量
    :param max_epoch: 最大迭代轮数,超出则返回
    :param min_loss: 满足条件的最小loss值,loss小于该值则返回
    :param _lambda: 正则项的lambda系数, 默认不含正则项
    :return: 返回求解得带的参数w, 列向量
    """
    order = np.shape(x_train)[1] - 1
    w = np.mat(np.zeros(order + 1)).T

    # 此处往下代码可以对照报告或wiki中的伪代码看
    A = np.dot(x_train.T, x_train) + _lambda * np.eye(order + 1)
    r = np.dot(x_train.T, y_train) - np.dot(A, w)
    p = r
    for epoch in range(max_epoch):
        alpha = np.dot(r.T, r) / np.dot(np.dot(p.T, A), p)
        w1 = w + np.dot(p, alpha)
        r1 = r - np.dot(np.dot(A, p), alpha)
        if np.dot(r1.T, r1) < min_loss:
            return w1, epoch + 1
        beta = np.dot(r1.T, r1) / np.dot(r.T, r)
        p1 = r1 + np.dot(p, beta)
        r, p, w = r1, p1, w1
    return w, max_epoch


def plot(w, x_test, y_test, start, end, _lambda=0.0, title='1'):
    """
    画图所用的函数
    :param w:
    :param x_test:
    :param y_test:
    :param start:
    :param end:
    :param _lambda:
    :param title:
    :return:
    """
    func = np.poly1d(np.array(w.T).flatten()[::-1])  # 生成多项式，注意这里的w需要倒过来
    x = np.linspace(start, end, 1000)
    y = func(x)
    x1 = np.array(x_test[:, 1]).tolist()
    y1 = np.array(y_test).tolist()
    plt.scatter(x1, y1, color='m')  # 画出test样本的散点图
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
