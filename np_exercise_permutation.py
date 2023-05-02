# -*- coding: utf-8 -*-
"""
__project_ = 'genetic_algorithm'
__file_name__ = 'np_exercise_permutation.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/2 14:38'
"""
import numpy as np
import random


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

    # 示例
    elements = [i for i in range(10)]
    num_rows = 10
    shuffled_matrix = generate_shuffled_matrix(elements, num_rows)

    print("生成的矩阵:")
    print(shuffled_matrix)