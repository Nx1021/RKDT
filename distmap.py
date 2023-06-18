import numpy as np
import matplotlib.pyplot as plt
import torch
def softmax(x):
    e_x = np.exp(x - np.max(x))  # 减去最大值，避免数值溢出
    return e_x / np.sum(e_x, axis=0)

def calculate_scores(points, pk):
    alpha = 0.15
    beta = 0.4
    eps = 1e-4

    distances = np.linalg.norm(points - pk, axis=1)  # 计算Pk与其他点之间的欧几里得距离
    bg_distances = np.clip(alpha - distances.min(), eps, alpha)
    distances = np.append(distances, bg_distances)
    scores = np.zeros(len(points))  # 初始化得分向量为0
    
    scores = beta / distances
    scores = np.clip(scores, 0, 1/eps)

    scores = softmax(scores)
    
    return scores

# def calculate_scores(pk, points):
#     alpha = 0.15
#     beta = 0.4
#     eps = 1e-4

#     distances = torch.cdist(pk, points)  # 计算每个点与每个 pk 之间的欧几里得距离
#     bg_distances = torch.clamp( -distances.min(dim=-1)[0] + alpha, eps, alpha)
#     distances = torch.cat((distances, bg_distances.unsqueeze(-1)), dim= -1)

#     scores = beta / distances
#     scores = torch.clamp(scores, 0, 1/eps)

#     scores = torch.softmax(scores, dim=-1)

#     return scores

# # 示例
# points = torch.tensor([[[1.0, 2.0], [1.0, 4.0], [2.0, 1.0]]]) / 10
# pk = torch.tensor([[[1.0, 2.0], [1.0, 3.0], [2.0, 2.0]]]) / 10

# score_matrix = calculate_scores(pk, points)
# print(score_matrix)

# 生成随机点
np.random.seed(0)
num = 24
points = np.random.rand(num, 2)

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 左图
ax1.scatter(points[:, 0], points[:, 1])
for i, p in enumerate(points):
    ax1.text(p[0], p[1], str(i))
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_title('Random Points')

# 右图
heatmap = ax2.imshow(np.zeros((1, num+1)), cmap='hot', interpolation='nearest', origin='lower')
ax2.set_title('Score Heatmap')
text_objects = []
def update_heatmap(event):
    if event.xdata is not None and event.ydata is not None:
        pk = np.array([event.xdata, event.ydata])  # 获取鼠标位置
        scores = calculate_scores(points, pk)  # 计算得分向量
        
        # 清空打印的文字
        for text_obj in text_objects:
            text_obj.remove()
        text_objects.clear()
        
        # 更新热图数据
        heatmap.set_data(scores.reshape(1, num+1))
        heatmap.autoscale()
        
        # 在热图上打印得分向量的值
        for i, score in enumerate(scores):
            text_obj = ax2.text(i, 0, f'{score:.2f}', color='blue', ha='center', va='center')
            text_objects.append(text_obj)
        fig.canvas.draw()

# 连接鼠标移动事件
fig.canvas.mpl_connect('motion_notify_event', update_heatmap)

plt.show()
