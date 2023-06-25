import matplotlib.pyplot as plt
import numpy as np
import torch
from models.loss import calculate_scores

# sample_num = 101
# gt_point_num = 24
# x = np.linspace(0, 1.0, sample_num)  # 生成x坐标序列，步距为0.05
# y = np.linspace(0, 1.0, sample_num)  # 生成y坐标序列，步距为0.05

# xx, yy = np.meshgrid(x, y)  # 生成坐标网格

# coordinates = np.column_stack((xx.ravel(), yy.ravel()))  # 将x和y坐标合并为一个坐标序列

# points = torch.Tensor(np.random.random((gt_point_num, 2)))


# score_map = np.zeros((sample_num*sample_num, gt_point_num+1))
# for i, pk in enumerate(coordinates):
#     print(f"\r{i}", end="")
#     pk = torch.Tensor(np.expand_dims(pk, 0))
#     score = calculate_scores(pk, points)
#     score_map[i] = score.squeeze().numpy()
# score_map = score_map.reshape(sample_num, sample_num, gt_point_num+1)
# for i in range(24 + 1):
#     plt.subplot(5,5,i+1)
#     plt.imshow(score_map[:,:,i])
#     plt.scatter(points[:,0] * (sample_num-1), points[:,1]*(sample_num-1), c = 'r', s = 3)
# plt.show()
# print()


points = torch.Tensor(np.array([[0.5, 0.5]]))

coordinates = np.linspace(0.3, 0.7, 10000 + 1)
score_map = np.zeros((10000 + 1, 2))
for i, c in enumerate(coordinates):
    pk = torch.Tensor([[0.5, c]])
    score = calculate_scores(pk, points)
    score_map[i] = score.squeeze().numpy()

plt.plot(coordinates, score_map[:,0])
plt.plot(coordinates, score_map[:,1])
plt.show()