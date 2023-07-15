import numpy as np
from scipy.optimize import linear_sum_assignment

from typing import Callable


def perform_matching(A:np.ndarray, B:np.ndarray, threshold:float, cost_func:Callable):
    '''
    A: [N, ...]
    B: [M, ...]
    '''
    # 计算A和B之间的距离矩阵
    dist_matrix = cost_func(A, B)

    # 执行匈牙利算法进行二分图匹配
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # 根据匹配结果在距离矩阵中查询距离
    distances = dist_matrix[row_ind, col_ind]

    # 找到距离小于等于阈值的匹配
    valid_matches = distances <= threshold

    # 提取匹配成功的索引对
    matched_pairs = np.column_stack((row_ind[valid_matches], col_ind[valid_matches]))

    # 找到未匹配的索引
    unmatched_A = np.setdiff1d(np.arange(A.shape[0]), matched_pairs[:, 0])
    unmatched_B = np.setdiff1d(np.arange(B.shape[0]), matched_pairs[:, 1])

    return matched_pairs, unmatched_A, unmatched_B