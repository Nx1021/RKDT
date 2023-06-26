from models.loss import calculate_scores
import numpy as np
import torch
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

alpha = 0.15
beta = 0.4
eps = 1e-4

_center = np.array([0.5, 0.5])
_cov = np.array([[0.082/30, 0], [0, 0.082/30]])

# def f(points):
#     mean = _center  # 均值
#     cov = _cov  # 协方差矩阵
#     mvn = multivariate_normal(mean=mean, cov=cov)
#     pdf_values = mvn.pdf(points)
#     return pdf_values

def f(points):
    mean = _center
    cov = _cov
    # 计算协方差矩阵的逆和行列式
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    # 计算均值差
    mean_diff = points - mean

    # 计算指数部分
    exponent = -0.5 * np.einsum('ij,ij->i', np.dot(mean_diff, inv_cov), mean_diff)

    # 计算系数
    coefficient = 1 / (2 * np.pi * np.sqrt(det_cov))

    # 计算概率密度函数值
    pdf_values = coefficient * np.exp(exponent)

    return pdf_values

def w(pk):
    points = np.expand_dims(_center, 0)
    return calculate_scores(torch.Tensor(pk), torch.Tensor(points), alpha, beta, eps)[:, 0].numpy()

x = np.linspace(0, 1, 1001)
y = np.linspace(0, 1, 1001)
X, Y = np.meshgrid(x, y)
points = np.stack([X.flatten(), Y.flatten()], axis=1)

Zf = f(points)
Zf = Zf.reshape(X.shape)
Wf = w(points)
Wf = Wf.reshape(X.shape)

plt.subplot(1,2,1)
plt.imshow(Zf, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
plt.colorbar()
plt.xticks(np.arange(0, 1, 0.1))
plt.yticks(np.arange(0, 1, 0.1))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Function f')
plt.grid(True)

plt.subplot(1,2,2)
plt.imshow(Wf, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
plt.colorbar()
plt.xticks(np.arange(0, 1, 0.1))
plt.yticks(np.arange(0, 1, 0.1))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Function w')
plt.grid(True)

plt.show()

# 设置参数
iter_num = 10000
d = np.zeros(iter_num)
for i in range(iter_num):
    n = 2  # 点的数量

    # 生成随机点坐标
    points = np.random.multivariate_normal(_center, _cov, size=n)

    # 计算权重函数值
    weights = w(points)  # 这里假设w()是一个已知的权重函数

    # 计算均值
    prob = f(points)
    weighted_sum = np.sum(points.T * prob * weights, axis=-1) / np.sum(prob * weights)
    mean = weighted_sum
    d[i] = np.linalg.norm(mean - _center)
mask = np.isnan(d) # 创建布尔掩码
filtered_array = d[~mask]  # 应用掩码，排除包含NaN的行
print("Mean:", np.mean(filtered_array))
