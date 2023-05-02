# -*- coding: utf-8 -*-
"""
__project_ = 'genetic_algorithm'
__file_name__ = 'SpectralClustering_with_kendalltau_measure.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/2 14:02'


2023年5月2日20:09:30
搞了一堆方法, 都不能比较相似, 比如说这个kendalltau 相似度度量, 他度量出来的曲线并没有多相似,反而非常消耗算力
暂时探索到此吧, 我以后需要多看看, 序列/排列相似性理论

"""
import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from scipy.stats import kendalltau

# 计算 kendalltau 距离
def kendalltau_distance_matrix(data: pd.DataFrame) -> tuple[float,np.ndarray]:
    """
    data: pd.DataFrame - 输入数据，每行表示一个对象，每列表示一个特征
    kendalltau  计算 两个向量的排列相关度,计算方法是 相对顺序数-相对逆序数比上一些归一化设计, -1表示不相关,1表示非常相关,0表示无关
    pdist 逐点距离创建, 接受多个向量, 逐点计算他们的距离, 返回距离值, n个向量返回n^2个值
    squareform 将一维n^2个向量转成n*n的方阵
    """
    dist_array = pdist(data, lambda u, v: kendalltau_metric(u,v))
    dist_mean:float = np.mean(dist_array)
    dist_var:float = np.std(dist_array)
    dist_matrix = squareform(dist_array)
    return dist_var,dist_mean,dist_matrix

def kendalltau_metric(x,y):
    # kendalltau  计算 两个向量的排列相关度,计算方法是 相对顺序数-相对逆序数比上一些归一化设计, 所以 -1全是差异,0表示差异和相同各半, 1表示完全相同
    return 1 - 0.5*(1+kendalltau(x,y,variant="b").statistic) # 加1表示向上抬,此时完全差异也大于0, 完全相同时1+1=2所以要除以2,得1,

def generate_shuffled_matrix(elements, num_rows):
    """
    生成一个矩阵，矩阵的每一行都是同一个集合元素的打乱重排。

    参数:
        elements (list): 待排集合
        num_rows (int): 矩阵的行数

    返回:
        shuffled_matrix (numpy.ndarray): 生成的矩阵
    """

    shuffled_matrix = np.empty((num_rows, len(elements)), dtype=type(elements[0]))

    for i in range(num_rows):
        shuffled_row = elements.copy()
        random.shuffle(shuffled_row)
        shuffled_matrix[i] = shuffled_row

    return shuffled_matrix


if __name__ == "__main__":

    pass

    keep_working = True
    count=0
    while keep_working:
        # 输入数据
        count+=1
        data:"np.ndarray" = generate_shuffled_matrix([i for i in range(20)],100)  # 序列长度会影响聚类效果, 因为序列越长, 他们相似的可能性就越低,目前还不知道怎么消除这个问题
        # data: "np.ndarray" = pd.DataFrame([
        #         [0, 1, 2, 3, 4, 5, 6],
        #         [0, 1, 2, 3, 5, 4, 6],
        #         [0, 1, 2, 5, 3, 4, 6],
        #         [0, 1, 5, 2, 3, 4, 6],  #
        #         [0, 5, 1, 2, 3, 4, 6],  #
        #         [5, 0, 1, 2, 3, 4, 6],
        #         [5, 0, 1, 2, 4, 3, 6],
        #         [5, 0, 1, 4, 2, 3, 6],
        #         [5, 0, 4, 1, 2, 3, 6],
        #         [5, 4, 0, 1, 2, 3, 6],
        #         [5, 4, 0, 1, 3, 2, 6],
        #         [5, 4, 0, 3, 1, 2, 6],
        #         [5, 4, 3, 0, 2, 1, 6],
        #         [5, 4, 3, 2, 0, 1, 6],
        #         [5, 4, 3, 2, 1, 0, 6],
        # ]).values


        print("begin count=",count)
        print(data)
        # 计算 kendalltau 距离矩阵
        dist_var,dist_mean,distance_matrix = kendalltau_distance_matrix(data)
        print(dist_var,dist_mean)
        print(distance_matrix)
        # 应用 DBSCAN 聚类
        eps =max(0.2,1/len(data[0]) ) # float - DBSCAN中的半径参数，表示领域的大小
        print("eps=",eps)
        min_samples = 2  # int - DBSCAN中的最小点数参数，表示形成高密度区域所需的最少点数
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=kendalltau_metric)
        clusters = dbscan.fit_predict(data)
        #
        print(clusters)
        for i in range(len(clusters)):
            if clusters[i]==0:
                print(data[i])
                keep_working=False
    print("count=",count)