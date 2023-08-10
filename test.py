from post_processer.PostProcesser import PostProcesser
from post_processer.pnpsolver import PnPSolver
from posture_6d.posture import Posture
import numpy as np
import matplotlib.pyplot as plt
# 在还没有指定TypeVar类型，其类型为bound指定的类型时： Generic、参数正确类型输入、属性注释缺一不可
# 在子类指定了Generic的类型后，其类型被绑定，子类的子类无法再修改已经绑定的TypeVar
# 如果子类未绑定，则可以在子类的子类再指定

cfgfile = "cfg/oldt_linemod.yaml"
pnpsolver = PnPSolver(cfg_file=cfgfile)
pp = PostProcesser(pnpsolver)

landmarks = pp.mesh_manager.get_ldmk_3d(0)
ldmk_num_list = [4, 8, 12, 16, 20, 24]
z_list = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
error_list = np.linspace(1, 41, 11)
error_mat = np.zeros((len(ldmk_num_list), len(z_list), len(error_list)), np.float32)
# 测试不同数量的ldmk，在不同的误差下，pnp求解的结果
for i, ldmk_num in enumerate(ldmk_num_list):
    ldmk = landmarks[:ldmk_num]
    for j, z in enumerate(z_list):
        rvec = np.array([[0,0,0]], np.float32)
        tvec = np.array([[0,0,z]], np.float32)
        proj = pp.pnpsolver.calc_reproj(rvec, tvec ,ldmk)

        for k, e in enumerate(error_list):
            print("\rldmk_num: {:>6}, z: {:>6}, error: {:>6}".format(ldmk_num, z, e), end="")
            for _ in range(1000):
                noise = np.random.random((len(proj), 2)) * e - e/2
                p_proj = proj + noise
                p_rvec, p_tvec = pp.pnpsolver.solvepnp(p_proj, ldmk)

                error_mat[i, j, k] += np.abs(np.squeeze(p_tvec)[2] - np.squeeze(tvec)[2]) / 1000

np.save("error_mat.npy", error_mat)

for i in range(6):
    plt.subplot(3, 2, i+1)
    plt.title("ldmk_num: {}".format(ldmk_num_list[i]))
    plt.imshow(error_mat[i, :, :], vmin=error_mat[1:].min(), vmax=error_mat[1:].max())
    plt.xlabel("z")
    # 设置X轴的刻度
    plt.xticks(np.arange(len(z_list)), z_list)
    plt.ylabel("error")
    # 设置Y轴的刻度
    plt.yticks(np.arange(len(error_list)), error_list)

    # 设置颜色的最大值和最小值
plt.show()
