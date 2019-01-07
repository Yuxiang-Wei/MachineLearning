import numpy as np
import matplotlib.pyplot as plt


def generate_random_data(mean1, cov1, size1, mean2, cov2, size2):
    """
    辅助函数生成随机数据样本
    使用两个高斯函数，生成代表两个不同类别的点集，合在一起组成我们的数据集
    :param mean1: 类别1样本点的均值
    :param cov1: 类别1样本点的协方差矩阵
    :param size1: 类别1样本点的数目
    :param mean2: 类别2样本点的均值
    :param cov2: 类别2样本点的协方差矩阵
    :param size2: 类别2样本点的数目
    :return: 返回生成的数据集
    """
    # 使用NumPy创建合成数据
    # 假设各个X相互独立，且服从均值为u,方差为delta的高斯分布
    X1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=size1)
    X2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=size2)
    Y1 = np.ones(size1)
    Y2 = np.zeros(size2)
    plt.scatter(X1[:, 0], X1[:,1], c='r')
    plt.scatter(X2[:, 0], X2[:, 1], c='b')
    X1, X2, Y1, Y2 = np.mat(X1), np.mat(X2), np.mat(Y1).T, np.mat(Y2).T
    X = np.row_stack((X1, X2))
    Y = np.row_stack((Y1, Y2))
    data = np.column_stack((X,Y))
    np.random.shuffle(data) # 打乱数据集
    return data

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# 计算给定参数和样本情况下的loss值, 此时的loss未开根号
def get_loss(w, x, y, _lambda=0.0):
    loss = y.T * np.log(sigmoid(np.dot(x, w))) + (1 - y).T * np.log(1 - sigmoid(np.dot(x, w)))
    return - float(loss) / np.shape(y)[0]

# 计算正确率
def get_accuracy(w, x, y):
    sum = 0.0
    pre = sigmoid(np.dot(x, w))
    for i in range(np.shape(y)[0]):
        if pre[i] >= 0.5 and y[i] == 1:
            sum = sum + 1
        elif pre[i] < 0.5 and y[i] == 0:
            sum = sum + 1
    return sum/(1.0 * np.shape(y)[0])

def lr_with_gradient_decent(x_train, y_train, lr, epoch_num, _lambda=0.0):
    """
    logistic回归的梯度下降法实现
    :param x_train: 训练集特征矩阵
    :param y_train: 训练集标签向量
    :param lr: 学习率
    :param epoch_num: 迭代次数
    :param _lambda: 正则项系数
    :return: 返回训练得到的参数w
    """
    data_size, theta_num = np.shape(x_train)
    # 为x矩阵加上一列全为1的列组成X
    x_train = np.column_stack((np.mat(np.ones(data_size)).T, x_train))
    w = np.random.normal(0, 0.01, size=theta_num+1)
    w = np.mat(w).T
    for epoch in range(epoch_num):
        hx = sigmoid(np.dot(x_train, w)) - y_train #求 y(x) - y
        # 更新w,更新规则可见实验报告
        w = w - lr / data_size * (np.dot(x_train.T, hx) + _lambda * w)
    return w


def lr_with_newton_method(x_train, y_train, epoch_num, _lambda=0.0):
    """
    logistic回归的梯度下降法实现
    :param x_train: 训练集特征矩阵
    :param y_train: 训练集标签向量
    :param epoch_num: 最大迭代次数
    :param _lambda: 正则项的系数
    :return: 返回训练得到的参数w
    """
    data_size, theta_num = np.shape(x_train)
    # 为x矩阵加上一列全为1的列组成X
    x_train = np.column_stack((np.mat(np.ones(data_size)).T, x_train))
    w = np.random.normal(0, 0, size=theta_num + 1)
    w = np.mat(w).T
    count = 0
    for epoch in range(epoch_num):
        fx = sigmoid(np.dot(x_train, w))
        hx = fx - y_train
        # 求Hessian所用
        A = np.multiply(fx, 1-fx)
        # 将A展成对角阵，方便后面使用矩阵运算求Hessian阵
        A = np.diagflat(A.T)
        # 求取Hessian阵，具体推导可见报告
        H = np.dot(x_train.T,np.dot(A, x_train))
        # 判断Hessian阵是否奇异，奇异则退出
        if np.linalg.det(H) == 0:
            break
        accuracy0 = get_accuracy(w, x_train, y_train)
        w = w - np.dot((H + _lambda * np.eye(theta_num+1)).I, np.dot(x_train.T, hx) + _lambda * w)
        accuracy1 = get_accuracy(w, x_train, y_train)
        if accuracy1 == accuracy0:
            count += 1
            if count == 5:
                # 5次迭代acc不变，说明已收敛，退出
                break
            continue
        count = 0
    return w

def uci_test():
    """
    使用真实的UCI数据集来测试
    :return:
    """
    f = open('mammographic_masses.data', 'r')
    data = []
    lines = f.readlines()
    for line in lines:
        if '?' in line:
            continue # 有数据缺少的项直接扔掉
        attributes = line.split(',')
        row = [float(attribute) for attribute in attributes]
        data.append(row)
    data = np.mat(data)
    np.random.shuffle(data)
    # 划分训练集和测试集， 200个样本用作测试集
    x_train = data[:-200, :-1]
    y_train = data[:-200, -1]
    x_test = data[-200:, :-1]
    y_test = data[-200:, -1]
    w = lr_with_gradient_decent(x_train, y_train, 1e-3, 100000)
    x_test = np.column_stack((np.mat(np.ones(x_test.shape[0])).T, x_test))
    accuracy = get_accuracy(w, x_test, y_test)
    print('uci test accuracy:', accuracy)

if __name__ == '__main__':
    uci_test()
