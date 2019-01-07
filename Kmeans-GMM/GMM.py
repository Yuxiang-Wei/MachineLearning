import random
import numpy as np
import matplotlib.pyplot as plt
from kmeans import * # 修改为具体的包

######################################################
# 给定样本和均值，求解协方差阵
# x表示样本， mean表示数据
######################################################
def calSigma(x, mean):
    data_size, dim = np.shape(x) # 维度
    sigma = np.zeros(dim)
    for i in range(dim):
        # 计算第i个属性的方差，一共有dim个
        for j in range(data_size):
            sigma[i] += (float(x[j][i]) - float(mean[:, i]))**2
        sigma[i] = sigma[i]/data_size
    return np.diagflat(sigma) # 将sigma变为对角阵

######################################################
# 给定样本和kmeans聚类的中心，求取EM聚类的参数，mu，alpha和sigma
# data表示样本， centers表示kmeans聚类的中心
######################################################
def calParam(data, centers):
    data_size = data.shape[0]
    k = centers.shape[0]
    alpha = np.zeros(k)
    distance = get_distanse(data, centers)
    label = np.argmin(distance, axis=1).tolist()  # 按每行求出最小值的索引
    # 根据标签对每类进行划分
    clusters = [[] for i in range(k)]
    for i in range(data_size):
        clusters[label[i]].append(data[i].tolist()[0])
    # 计算alpha
    for i in range(k):
        alpha[i] = 1.0 * len(clusters[i]) / data_size
    # center 即为 mu
    mu = centers
    # 计算sigma
    sigma = [np.eye(k) for i in range(k)]
    for i in range(k):
        sigma[i] = calSigma(clusters[i], centers[i, :])
    return alpha, mu, sigma

######################################################
# 随机初始化模型参数
# shape 是表示样本规模的二元组，(样本数, 特征数)
# K 表示模型个数
######################################################
def init_params(shape, k):
    data_size, dim = shape
    mu = np.mat(np.random.rand(k, dim))
    cov = [np.mat(np.eye(dim)) for i in range(k)]
    alpha = np.array([1.0 / k] * k)
    return mu, cov, alpha

######################################################
# 第 k 个模型的高斯分布密度函数
# 每 i 行表示第 i 个样本在各模型中的出现概率
# 返回一维列表
######################################################
def pdf(x, mu, sigma):
    n = np.shape(mu)[1]
    # 求取系数部分
    cof = np.power(1.0 / (2 * np.pi), n / 2.0) * np.sqrt(np.linalg.det(sigma))
    # 求取指数部分
    exp = np.exp(-0.5 * np.dot(np.dot((x - mu), np.mat(sigma).I), (x - mu).T))
    exp = np.diag(exp) # 提取对角线元素
    return 1.0 / cof * exp

######################################################
# 绘制聚类结果
######################################################
def plotCluster1(data, gamma, centers):
    colors = ['r', 'g', 'b', 'brown', 'gray', 'crimson', 'darkturquoise', 'darkgreen']
    marker = ['+', '>', 'x', 'v', '^', '<', '1', '2', 'h', '.']
    label = np.argmax(gamma, axis=1).tolist()  # 按每行求出最大值的索引，即为聚类结果
    col = []
    for i in range(np.shape(data)[0]):
        col.append(colors[label[i][0]]) # label[i][0]即为第i个样本的类别，为数字
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), c=col)
    for i in range(np.shape(centers)[0]):
        plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], marker=marker[i])
    plt.title('EM')
    plt.show()


######################################################
# EM算法求解GMM
# data表示数据,k表示聚类数
# 最终返回模型参数
######################################################
def gmm(data, k):
    data_size = np.shape(data)[0]
    #　可以使用kmeans聚类结果作为初始化参数
    centers = k_means(data, k)
    alpha, mu, sigma = calParam(data, centers)
    plotCluster(data, centers) # 打印kmeans的聚类结果
    # 也可以随机初始化参数
    # mu, sigma, alpha = init_params(data.shape, k)
    gamma = np.mat(np.zeros((data_size, k)))
    log_likelihood0 = 0
    likelihood = [] # 用于保存每一步的似然，方便绘制似然曲线
    for z in range(1000):
        # 计算各模型中所有样本出现的概率，行对应样本，列对应模型
        prob = np.zeros((data_size, k))
        for i in range(k):
            prob[:, i] = pdf(data, mu[i, :], sigma[i])
        prob = np.mat(prob)

        # 计算每个模型对每个样本gamma
        for i in range(k):
            gamma[:, i] = alpha[i] * prob[:, i]
        # 计算lld
        log_likelihood1 = np.sum(np.log(np.sum(gamma, axis=1)))
        for i in range(data_size):
            gamma[i, :] /= np.sum(gamma[i, :])
        dim = np.shape(data)[1]
        # 更新每个模型的参数
        for i in range(k):
            # 第 k 个模型对所有样本的gamma之和
            Nk = np.sum(gamma[:, i])
            # 更新 mu
            # 对每个特征求均值
            for d in range(dim):
                mu[i, d] = np.sum(np.multiply(gamma[:, i], data[:, d])) / Nk
            # 更新 cov
            cov_k = np.mat(np.zeros((dim, dim)))
            for j in range(data_size):
                cov_k += gamma[j, i] * np.dot((data[j, :] - mu[i,:]).T, (data[j, :] - mu[i,:])) / Nk
            sigma[i] = cov_k
            # 更新 alpha
            alpha[i] = Nk / data_size
        print(z, '  ' , log_likelihood0)
        if abs(log_likelihood1 - log_likelihood0) < 1e-5:
            break # 若似然变化在指定范围内，则退出
        log_likelihood0 = log_likelihood1
        likelihood.append(log_likelihood0)
    return gamma, mu, sigma, alpha, likelihood


######################################################
# 加载UCI数据集，该数据集为鸢尾花数据集，有四个实值属性，3个类
# 返回属性组成的数据矩阵和量化后的标签列表
######################################################
def read_file():
    f = open('bezdekIris.data', 'r')
    lines = f.readlines()
    data = []
    label = []
    for line in lines:
        line = line.strip('\n')
        attrs = line.split(',')
        row = [float(x) for x in attrs[:-1]]
        if len(row) == 0:
            continue
        data.append(row)
        # 量化标签
        if attrs[-1] == 'Iris-setosa':
            label.append(0)
        elif attrs[-1] == 'Iris-versicolor':
            label.append(1)
        else:
            label.append(2)
    return np.mat(data), label


######################################################
# 使用UCI数据测试EM算法，绘制原样本标签，似然变化以及聚类结果
######################################################
def test_uci():
    data,label = read_file()
    gamma, mu, sigma, alpha, likelihood = gmm(data, 3)
    step = range(1, len(likelihood) + 1)
    plt.plot(step, likelihood)
    plt.ylabel('likelihood')
    plt.xlabel('step')
    plt.title('k = 3')
    plt.show()
    x = range(1, np.shape(data)[0]+ 1)
    colors = ['r' if l == 0 else 'g' if l == 1 else 'b' for l in label]
    plt.scatter(x, label, c=colors)
    plt.show()
    label = np.argmax(gamma, axis=1).tolist()  # 按每行求出最大值的索引
    colors = ['r' if int(l[0]) == 0 else 'g' if int(l[0]) == 1 else 'b' for l in label]
    plt.scatter(x, label, c=colors)
    plt.show()

if __name__ == '__main__':
    test_uci()
