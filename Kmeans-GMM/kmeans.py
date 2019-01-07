import random
import numpy as np
import matplotlib.pyplot as plt


######################################################
# 生成训练用的数据，共由四个高斯分布组成，4个均值分布在4个象限内
# 通过协方差阵可以控制混合程度
######################################################
def generate_data():
    mean1 = [-5, -5]
    mean2 = [4, 4]
    mean3 = [-5, 4]
    mean4 = [4, -5]
    cov1 = [[4, 0], [0, 4]]  # 待改变项
    cov2 = [[2, 0], [0, 2]]  # 待改变项
    size1 = 100
    size2 = 100
    x1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=size1)
    x2 = np.random.multivariate_normal(mean=mean2, cov=cov1, size=size2)
    x3 = np.random.multivariate_normal(mean=mean3, cov=cov1, size=size1)
    x4 = np.random.multivariate_normal(mean=mean4, cov=cov1, size=size2)
    plt.scatter(x1[:, 0], x1[:, 1], c='r')
    plt.scatter(x2[:, 0], x2[:, 1], c='g')
    plt.scatter(x3[:, 0], x3[:, 1], c='b')
    plt.scatter(x4[:, 0], x4[:, 1], c='brown')
    plt.show()
    x1, x2, x3, x4 = np.mat(x1), np.mat(x2), np.mat(x3), np.mat(x4)
    x = np.row_stack((x1,x2))
    x = np.row_stack((x, x3))
    x = np.row_stack((x, x4))
    np.random.shuffle(x)
    return x

######################################################
# 绘制聚类完后的结果图
######################################################
def plotCluster(data, centers):
    colors = ['r', 'g', 'b', 'brown', 'gray', 'crimson', 'darkturquoise', 'darkgreen']
    marker = ['+', '>', 'x', 'v', '^', '<', '1', '2', 'h', '.']
    result = get_distanse(data, centers)
    label = np.argmin(result, axis=1).tolist()  # 按每行求出最小值的索引,作为聚类结果
    col =[]
    for i in range(np.shape(data)[0]):
        col.append(colors[label[i]]) # 为每个点确定颜色
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), c=col)
    for i in range(np.shape(centers)[0]):
        plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], marker=marker[i], s=200)
    plt.title('kmeans')
    plt.show()

######################################################
# 计算两个向量x1，x2的欧式距离
######################################################
def d(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    tmp = np.sum((x1 - x2) ** 2) # 差值的平方
    return np.sqrt(tmp)

######################################################
# 从数据中随机选取k个点，并返回
######################################################
def rand_center(data, k):
    data_size = range(data.shape[0])
    index = random.sample(data_size,k) # 随机选取k个点
    centers = [data[i, :].tolist()[0] for i in index]
    return np.mat(centers)

######################################################
# 计算样本和每个中心点的距离
# 返回一个矩阵，行表示样本，列表示与第i和均值的距离
######################################################
def get_distanse(data, centers):
    data_size = np.shape(data)[0]
    k = np.shape(centers)[0]
    result = np.zeros((data_size, k))
    for i in range(data_size):
        for j in range(k):
            result[i, j] = d(data[i, :], centers[j, :])
    return result

######################################################
# kmeans算法
######################################################
def k_means(data, k):
    centers = rand_center(data, k) #获得中心点
    data_size = np.shape(data)[0]
    for num in range(500):
        distance = get_distanse(data, centers)
        # 按每行求出最小值的索引,即为该点标签
        label = np.argmin(distance, axis=1).tolist()
        # 根据标签对每类进行划分
        clusters = [[] for i in range(k)]
        for i in range(data_size):
            clusters[label[i]].append(data[i].tolist()[0])
        # 计算新的均值
        new_centers = [np.mean(np.mat(clusters[i]), axis=0).tolist()[0] for i in range(k)]
        new_centers = np.mat(new_centers)
        # 若均值未变，则结束
        if (new_centers == centers).all():
            print('结束')
            break
        centers = new_centers # 更新均值
    return centers

# 验证kmeans
if __name__ == '__main__':
    data = generate_data()
    centers = k_means(data, 5)
    plotCluster(data, centers)