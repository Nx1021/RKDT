import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 减去最大值，避免数值溢出
    return e_x / np.sum(e_x, axis=0)

def calculate_scores(points, pk, alpha = 0.15, beta = 0.4):
    alpha = alpha
    beta = beta
    eps = 1e-4

    distances = np.linalg.norm(points - pk, axis=1)  # 计算Pk与其他点之间的欧几里得距离
    bg_distances = np.clip(alpha - distances.min(), eps, alpha)
    distances = np.append(distances, bg_distances)
    scores = np.zeros(len(points))  # 初始化得分向量为0
    
    scores = beta / distances
    scores = np.clip(scores, 0, 1/eps)

    scores = softmax(scores)
    
    return scores

def show(image_res = 200):
    points = np.array(
    [[0.5, 0.6], 
     [0.6, 0.4],
     [0.65, 0.45]]
)

    fig, axs = plt.subplots(3, 4, figsize=(10, 10))  # 创建2x2的子图数组

    # 生成网格点
    x = np.linspace(0, 1, image_res)
    y = np.linspace(0, 1, image_res)
    x, y = np.meshgrid(x, y)
    grid_points = np.stack((x, y), axis=-1).reshape(-1, 2)

    ### 计算得分
    scores = []
    for pk in grid_points:
        score = calculate_scores(points, pk)
        scores.append(score)
    scores = np.array(scores)
    scores = scores.reshape(image_res, image_res, -1)

    # 绘制热图
    for i in range(4):
        plt.subplot(3,4,1+i)
        plt.imshow(scores[:,:,i],  interpolation='nearest', origin='lower')
        plt.scatter(points[:,0]*image_res - 0.5, points[:,1]*image_res - 0.5, c='r', s=10)
        plt.xticks(np.linspace(0, image_res, 3), [str(round(x,1)) for x in np.linspace(0, 1, 3)])
        plt.yticks(np.linspace(0, image_res, 3), [str(round(x,1)) for x in np.linspace(0, 1, 3)])

    ### 计算得分2 
    scores_2 = []
    for pk in grid_points:
        score = calculate_scores(points, pk, alpha=0.25, beta=0.4)
        scores_2.append(score)
    scores_2 = np.array(scores_2)
    scores_2 = scores_2.reshape(image_res, image_res, -1)
    # 绘制热图
    for i in range(4):
        plt.subplot(3,4,5+i)
        plt.imshow(scores_2[:,:,i],  interpolation='nearest', origin='lower')
        plt.scatter(points[:,0]*image_res - 0.5, points[:,1]*image_res - 0.5, c='r', s=10)
        plt.xticks(np.linspace(0, image_res, 3), [str(round(x,1)) for x in np.linspace(0, 1, 3)])
        plt.yticks(np.linspace(0, image_res, 3), [str(round(x,1)) for x in np.linspace(0, 1, 3)])

    ### 计算得分3 
    scores_3 = []
    for pk in grid_points:
        score = calculate_scores(points, pk, alpha=0.15, beta=1.0)
        scores_3.append(score)
    scores_3 = np.array(scores_3)
    scores_3 = scores_3.reshape(image_res, image_res, -1)
    # 绘制热图
    for i in range(4):
        ax = plt.subplot(3,4,9+i)
        im = plt.imshow(scores_3[:,:,i],  interpolation='nearest', origin='lower')
        plt.scatter(points[:,0]*image_res - 0.5, points[:,1]*image_res - 0.5, c='r', s=10)
        plt.xticks(np.linspace(0, image_res, 3), [str(round(x,1)) for x in np.linspace(0, 1, 3)])
        plt.yticks(np.linspace(0, image_res, 3), [str(round(x,1)) for x in np.linspace(0, 1, 3)])
    
    # divider = make_axes_locatable(axs)

    # # 添加颜色条到新的坐标轴
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # colorbar = plt.colorbar(im, cax=cax)

    # # 设置颜色条的标签
    # colorbar.set_label('Intensity')  # 你可以根据需要设置不同的标签

    plt.show()
    

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
# np.random.seed(0)
# num = 24
# points = np.random.rand(num, 2)

points = np.array(
    [[0.3, 0.65], 
     [0.6, 0.2],
     [0.65, 0.25]]
)

show(50)

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
