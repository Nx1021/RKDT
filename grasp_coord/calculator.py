from MyLib.posture import Posture
from post_processer.model_manager import ModelInfo

import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage
import os
import cv2
import time
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sko.GA import GA
import itertools
from tqdm import tqdm

from grasp_coord.object_pcd import ObjectPcd, create_ObjectPcd_from_file
from grasp_coord.gripper import Gripper, MyThreeFingerGripper
from grasp_coord import MODELS_DIR

from utils.yaml import yaml_load

def compute_angle(v1, v2):
    """
    计算两组向量的夹角

    参数:
        v1: 形状为 [..., 3] 的向量
        v2: 形状为 [..., 3] 的向量

    返回值:
        形状为 [...] 的夹角值，单位为弧度
    """
    # 归一化向量
    v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True)
    v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True)
    v1_normalized = v1 / v1_norm
    v2_normalized = v2 / v2_norm

    # 计算夹角的余弦值
    cos_angle = np.sum(v1_normalized * v2_normalized, axis=-1)

    # 夹角的弧度值
    angle = np.arccos(cos_angle)

    return angle

class InitAngle:
    CONE_ANGLE = 0.6542
    def __init__(self) -> None:
        phi = (np.sqrt(5) - 1)/2
        vertexs = []
        for frac_count in range(3):
            count = (frac_count + 1) % 3
            for i in [-1, 1]:
                for j in [-1, 1]:
                    frac_value = i * (1/phi)
                    value = j * phi
                    vertex = np.zeros(3)
                    vertex[frac_count] = frac_value
                    vertex[count] = value
                    vertexs.append(vertex)
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1,1]:
                    vertexs.append(np.array([i, j, k]))
        
        vecs = np.array(vertexs)
        # 归一化
        length = np.linalg.norm(vecs, axis=1)[0]
        vecs = vecs/length
        self.vecs = vecs

        self.get_rvec()
        # print(vecs)
        
        # ax = plt.axes(projection='3d')  # 设置三维轴
        # ax.scatter(vecs[:,0], vecs[:,1], vecs[:,2], s=5, marker="o", c='r')
        # plt.show()

    def get_rvec(self):   
        vecs = self.vecs 
        ### 转换为旋转向量
        base = np.tile([0,0,1], [vecs.shape[0],1])
        times = np.sum( vecs * base, axis=-1)
        angle = np.arccos(times) #旋转角度
        rot = np.cross(base, vecs)
        self.rvec = rot * np.tile(np.expand_dims(angle/ np.linalg.norm(rot, axis=-1), -1), [1,3])        

class SphereAngle(InitAngle):
    def __init__(self) -> None:
        nums_points = 500
        radius = 1
        loc = np.zeros((nums_points, 3))
        ii = np.arange(1, nums_points+1, 1)
        phi_array = np.arccos(-1.0 + (2.0 * ii - 1.0) / nums_points)
        theta_array = np.sqrt(nums_points * np.pi) * phi_array
        loc[:,0] = radius * np.cos(theta_array) * np.sin(phi_array)
        loc[:,1] = radius * np.sin(theta_array) * np.sin(phi_array)
        loc[:,2] = radius * np.cos(phi_array)
        self.vecs = loc
        # ax = plt.axes(projection='3d')  # 设置三维轴
        # ax.scatter(self.vecs[:,0], self.vecs[:,1], self.vecs[:,2], s=5, marker="o", c='r')
        # plt.show()
        self.get_rvec()

class Voxelized():
    def __init__(self, 
                 entity_cube:np.ndarray,
                 restore_mat:np.ndarray,
                 orig_mesh = None) -> None:
        self.entity_cube         = entity_cube.astype(np.uint8)
        self.restore_mat = restore_mat          
        self.orig_mesh = orig_mesh
      
        self.surf_cube, self.erode_entity_cube = self.split_entity_cube(entity_cube)
        
        self.entity_indices =\
              np.array(np.where(self.entity_cube)).T #[N, 3]
        self.surf_indices =\
              np.array(np.where(self.surf_cube)).T
        
        self.surf_normals = self.calc_surf_normal()

        self.entity_query = np.full(self.entity_cube.shape, -1, np.int64)
        self.entity_query[self.entity_indices[:,0], 
                        self.entity_indices[:,1], 
                        self.entity_indices[:,2]] = np.array(range(self.entity_indices.shape[0]), np.int64)
        
        self.surf_query = np.full(self.surf_cube.shape, -1, np.int64)
        self.surf_query[self.surf_indices[:,0], 
                        self.surf_indices[:,1], 
                        self.surf_indices[:,2]] = np.array(range(self.surf_indices.shape[0]), np.int64)

    def surf_points(self):
        return self.restore_mat[:3, :3].dot(self.surf_indices.T).T + self.restore_mat[:3, 3]

    @property
    def shape(self):
        return self.entity_cube.shape
 
    def split_entity_cube(self, cube:np.ndarray):
        N = int(np.sum(cube.shape) / 20)
        surf_cube = np.zeros(cube.shape, np.uint8)
        for d in range(3):
            cube = np.swapaxes(cube, 0, d)
            surf_cube = np.swapaxes(surf_cube, 0, d)
            for i in range(cube.shape[0]):
                layer = cube[i]
                # 找外圈轮廓，并排除长度较小的轮廓
                # 查找轮廓
                contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # 绘制轮廓
                image = surf_cube[i].copy()
                for contour in contours:
                    # 排除长度小于N的轮廓
                    if len(contour) >= N:
                        cv2.drawContours(image, [contour], -1, 1, thickness=0)
                surf_cube[i] = image
            cube = np.swapaxes(cube, 0, d)
            surf_cube = np.swapaxes(surf_cube, 0, d)
        surf_cube = np.clip(surf_cube, 0, 1)
        eroded_body = (cube - surf_cube).astype(np.uint8)
        return surf_cube, eroded_body

    def calc_surf_normal(self):
        # 创建示例点云数据 A 和 B
        # points:np.ndarray, base_normal:np.ndarray
        surf = o3d.geometry.PointCloud()
        surf.points = o3d.utility.Vector3dVector(self.surf_indices)

        ref_pcd = o3d.geometry.PointCloud()
        inv_restore_mat = np.linalg.inv(self.restore_mat)
        ref_pcd.points = self.orig_mesh.vertices
        ref_pcd.normals = self.orig_mesh.vertex_normals
        ref_pcd.transform(inv_restore_mat)

        # 构建 KD 树
        kdtree = o3d.geometry.KDTreeFlann()
        kdtree.set_geometry(ref_pcd)

        # 对每个点在 A 中进行最近邻搜索并插值法向
        k = 1  # 最近邻点的数量
        interpolated_normals = np.zeros(self.surf_indices.shape)

        ref_pcd_normals = np.asarray(ref_pcd.normals)
        for i in range(len(surf.points)):
            _, indices, _ = kdtree.search_knn_vector_3d(surf.points[i], k)
            interpolated_normals[i] = ref_pcd_normals[indices]
        surf.normals = o3d.utility.Vector3dVector(interpolated_normals)

        return interpolated_normals
    
    def query_surf_normal(self, indices):
        idx = self.surf_query[indices[..., 0], indices[..., 1], indices[..., 2]]
        normals = self.surf_normals[idx]
        normals[np.any(idx == -1, -1)] = 0.0
        return normals

    @staticmethod
    def from_mesh(mesh, voxel_size):
        ### 进行体素化
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
        
        ###retore_mat
        origin = voxel_grid.origin
        restore_mat = np.eye(4)*voxel_size
        restore_mat[3, 3] = 1
        restore_mat[:3, 3] = origin

        _voxel_indices = np.asarray([x.grid_index for x in voxel_grid.get_voxels()])

        ### 创建cube
        # 计算体素网格的尺寸
        voxel_dims = np.max(_voxel_indices, 0) + 1
        # 创建全0的三维矩阵
        _entity = np.zeros(voxel_dims, dtype=np.uint8)
        _entity[_voxel_indices[:, 0], _voxel_indices[:, 1], _voxel_indices[:, 2]] = 1

        ### 开运算，填充大部分缝隙
        _entity = np.pad(_entity, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        kernel = ndimage.generate_binary_structure(3,2)
        _entity = ndimage.binary_closing(_entity, kernel, iterations=1).astype(np.uint8)
        _entity = _entity[1:-1, 1:-1, 1:-1]

        ### 填充实体       
        entity = fill_3d(_entity)

        ### 过滤部分离群点
        idx = np.array(np.where(entity)).T
        pcd = o3d.geometry.PointCloud()
        points = idx.astype(np.float32)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, _ = pcd.remove_statistical_outlier(60, 2)

        ### 
        final_indices = np.asarray(pcd.points, np.int64)

        entity = np.zeros(voxel_dims, dtype=np.uint8)
        entity[final_indices[:,0], final_indices[:,1], final_indices[:,2]] = 1

        return Voxelized(entity, restore_mat, orig_mesh = mesh)

    @staticmethod
    def from_pcd(self):
        pass

class Triangle_Hough():
    def __init__(self, resolution:float, r_max:float, r_min:float, theta_resolution:float) -> None:
        '''
        resolution: 长度单位的分辨率,包括x,y,r
        r_max: 最大半径
        r_min: 最小半径
        theta_resolution: 角度分辨率（单位是角度°）
        '''
        self.r_min = r_min
        self.r_max = r_max
        self.r_step = int((self.r_max - self.r_min) / resolution) + 1 # resolution

        self.theta_max = 2*np.pi/3
        self.theta_rsl = theta_resolution * np.pi / 180
        self.theta_step = int(np.round(self.theta_max / self.theta_rsl))

        self.resolution = resolution

    def crop_nonzero_region(self, image):
        # 找到非零像素的索引
        nonzero_indices = np.nonzero(image)
        if len(nonzero_indices[0]) == 0:
            return image[0:0, 0:0], (0,0)
        # 计算最小和最大索引
        min_row = np.min(nonzero_indices[0])
        max_row = np.max(nonzero_indices[0])
        min_col = np.min(nonzero_indices[1])
        max_col = np.max(nonzero_indices[1])

        # 裁剪出非零区域
        cropped_image = image[min_row:max_row+1, min_col:max_col+1]

        return cropped_image, (min_row, min_col)

    def plot(self, triangles:np.ndarray, image:cv2.Mat):
        def generate_triangle(center, radius, phase):
            angles = np.linspace(phase, phase + 2*np.pi, num=4)
            y = center[0] + radius * np.cos(angles)
            x = center[1] + radius * np.sin(angles)
            vertices = np.column_stack((x, y))
            return vertices

        plt.imshow(image)
        for triangle in triangles:
            center = triangle[:2]
            radius = triangle[2]
            phase = triangle[3]
            vertices = generate_triangle(center, radius, phase)
            plt.plot(vertices[:, 0], vertices[:, 1], '-')
        
        plt.show()

    def parse(self, result):
        rsls = np.array([
            self.resolution, 
            self.resolution, 
            self.resolution, 
            self.theta_rsl])
        offset = np.array([0, 0, self.r_min, 0])
        triangles = result * rsls + offset
        return triangles

    def triangles_filter(self, triangles, co_marks):
        num_groups = co_marks.shape[0]
        distances = np.zeros((num_groups, 3))

        for i in range(num_groups):
            group_points = co_marks[i]
            d1 = np.linalg.norm(group_points[0] - group_points[1])
            d2 = np.linalg.norm(group_points[0] - group_points[2])
            d3 = np.linalg.norm(group_points[1] - group_points[2])
            distances[i] = (d1, d2, d3)

        mask = triangles[:, 2] < np.min(distances, -1)

        triangles = triangles[mask]
        co_marks = co_marks[mask]

        return triangles, co_marks

    def run(self, image:cv2.Mat, ifplot = False):
        start = time.time()
        croped, org = self.crop_nonzero_region(image)

        marks = np.array(np.where(croped)).T #[M, 2]

        ### 创建采样点序列
        H = croped.shape[0] * self.resolution
        W = croped.shape[1] * self.resolution
        x_range = np.arange(W)
        y_range = np.arange(H)
        # 创建网格坐标
        x_coords, y_coords = np.meshgrid(x_range, y_range)
        # 将x和y坐标堆叠在一起
        points = np.stack((y_coords.ravel(), x_coords.ravel()), axis=1) #[P, 2]

        hough_space_list = []

        summary = np.zeros((H, W, self.r_step, self.theta_step), np.uint8)
        for mark in marks:
            hough_space = np.zeros((H, W, self.r_step + 1, self.theta_step), np.uint8) #[y, x, r, theta]
            diff = -(points - mark)
            
            r:np.ndarray = np.linalg.norm(diff, axis=-1) #[P]
            r_quant = np.round((r - self.r_min) / self.resolution).astype(np.int32)
            r_quant[r_quant > self.r_step] = self.r_step
            r_quant[r_quant < 0] = self.r_step

            theta = np.arctan2(diff[:, 1] , diff[:, 0])
            theta = np.mod(theta, 2*np.pi/3)
            theta_quant = np.round(theta / self.theta_rsl).astype(np.int32) #[P]
            theta_quant[theta_quant == self.theta_step] = 0

            hough_space[points[:,0], points[:,1], r_quant, theta_quant] = 1
            hough_space = hough_space[:, :, :-1, :]
            # hough_space = ndimage.binary_dilation(hough_space, ndimage.generate_binary_structure(4, 1))
            hough_space_list.append(hough_space)
            summary += hough_space

        condition = np.where(summary == 3)
        result = np.array(condition).T
        triangles = self.parse(result)

        ### 获取每个三角形对应的marker
        mark_idx = np.zeros((marks.shape[0], result.shape[0]))
        for i, hough_space in enumerate(hough_space_list):
            mark_idx[i] = hough_space[condition]
        mark_idx = np.where(mark_idx.T)
        mark_idx = np.array(mark_idx[1]).reshape(-1, 3)
        co_marks = marks[mark_idx] + org
        triangles[:, :2] += org
        triangles, co_marks = self.triangles_filter(triangles, co_marks)

        if ifplot:
            print(time.time() - start)
            if len(triangles) > 0:
                self.plot(triangles, image)
        
        return triangles, co_marks
        
    def _loop_method(self, image):
        '''
        do not use
        just for comparing the time cost with hough method
        '''
        graspable_index_GCS = []
        candi_graspable_index = np.array(np.where(image)).T
        min_distance =  self.r_min * 1.732
        max_distance = self.r_max*1.732
        # candi_graspable_index = np.vstack((candi_graspable_index, i * np.ones(candi_graspable_index.shape[-1]))).T
        # candi_graspable_pos = self._homo_pad(candi_graspable_index)
        # candi_graspable_pos = restore_mat.dot(candi_graspable_pos.T).T      
        # candi_graspable_pos = candi_graspable_pos[:,:3]          
        comb_2 = list(itertools.combinations(list(range(len(candi_graspable_index))), 2))
        comb_3 = list(itertools.combinations(list(range(len(candi_graspable_index))), 3))
        distance = {}
        # 先计算两两间距（握手）
        for j, partner in enumerate(comb_2): 
            # distance[j] =                     
            d = np.linalg.norm(candi_graspable_index[partner[0]] -\
                                        candi_graspable_index[partner[1]])
            if (d < max_distance and d > min_distance):
                distance.update({partner: d}) #只存储范围内的对
        # 以3个点为一组，两两间距差不超过阈值则认为是等边三角形
        for group in comb_3: 
            # distance[j] = 
            combs = list(itertools.combinations(group,2))
            try:
                group_distance = np.array([distance[combs[0]], distance[combs[1]], distance[combs[2]]])
            except KeyError:
                continue
            deltas = np.abs(np.array([
                group_distance[0] - group_distance[1],
                group_distance[0] - group_distance[2],
                group_distance[1] - group_distance[2]
            ]))
            if not np.any(deltas > 1):
                if np.any(group_distance < min_distance) or np.any(group_distance > max_distance):
                    continue
                else:
                    graspable_index_GCS.append(candi_graspable_index[np.array(group)])
        return graspable_index_GCS

class _CoordSearcher():
    '''
    接触点处理
    '''
    def __init__(self, gripper:Gripper, voxelized:Voxelized, v_friction_angle, h_friction_angle, voxel_size) -> None:
        
        self.voxelized: Voxelized = voxelized
        self.gripper = gripper

        self.v_friction_angle = v_friction_angle
        self.h_friction_angle = h_friction_angle
        self.voxel_size = voxel_size

        gripper.set_u(1)
        max_r = abs(gripper.finger_gripping_center[0])
        self.max_grasp_depth_index = int(gripper.max_grasp_depth/voxel_size) #max grasping depth in voxel
        gripper.set_u(0)
        min_r = abs(gripper.finger_gripping_center[0]) + 10
        self.hough_solver = Triangle_Hough(1, max_r/voxel_size, min_r/voxel_size, 2)

    def set_voxelized(self, voxelized:Voxelized):
        self.voxelized = voxelized

    def get_v_stable(self):
        ### 计算具有垂直法向的点的 cube
        grasp_vec = np.array([0,0,1])
        vertical_indices = self.voxelized.surf_indices[np.where(
            compute_angle(self.voxelized.surf_normals, grasp_vec) < (90 + self.v_friction_angle)*np.pi/180)[0]] # 近似垂直于抓取方向的点          
        surf_vertical_box = np.zeros(self.voxelized.entity_cube.shape, np.uint8)            
        surf_vertical_box[tuple(vertical_indices.T)] = 1
        return surf_vertical_box     

    def search_by_layer(self, surf_vertical_box:np.ndarray):
        ### search by layer
        max_proj = np.zeros(surf_vertical_box.shape[0:2], np.bool8)
        z_index = list(range(surf_vertical_box.shape[-1]))[:self.max_grasp_depth_index]
        
        triangles = []
        contacted = [] # record the graspable pos
        for i in z_index:
            surf_vertical_box_slice = surf_vertical_box[:,:,i]
            surf_vertical_box_slice = surf_vertical_box_slice * np.logical_not(max_proj) #过滤掉投影内的点

            triangle, co_marks = self.hough_solver.run(surf_vertical_box_slice, False)# [N, 4], [N, 3, 2]

            co_marks = np.concatenate((co_marks, 
                    np.full((co_marks.shape[0], co_marks.shape[1], 1), i, dtype = np.int64)), 
                    axis=-1) # [N, 3, 3]
            contacted.append(co_marks)
            triangles.append(triangle)

            max_proj = max_proj + self.voxelized.erode_entity_cube[:,:,i].astype(np.uint8) #最大投影轮廓，被包含在内的点将被排除
        contacted = np.concatenate(contacted)
        triangles = np.concatenate(triangles)
        return triangles, contacted

    def get_h_stable_grasp(self, triangles, contact_point_indices):
        '''
        triangles: [N, 4]
        contact_point_indices: [N, 3, 2]
        '''
        contact_point_indices = contact_point_indices.astype(np.int)
        ## calculate u
        gig_center = triangles[:, None, :2] #np.mean(contact_point_indices, axis = -2, keepdims=True) #[N, 1, 2]
        ## Filter those points whose angle of the horizontal grip with the gripper finger's direction is less than h, 
        ## sence these positions are unstable for grasping
        gig_normals = self.voxelized.query_surf_normal(contact_point_indices) #[N, 3, 3]
        gig_normals[..., 2] = 0
        grasp_direction = contact_point_indices[..., :2] - gig_center #[N, 3, 2]
        grasp_direction = np.pad(grasp_direction, ((0,0), (0,0), (0,1)), "constant", constant_values=0)

        h_angle = compute_angle(grasp_direction, gig_normals)
        # h_angle[h_angle > np.pi/2] = h_angle - np.pi
        mask = np.logical_not(np.any(h_angle > self.h_friction_angle * np.pi / 180, axis=-1))
        
        # self.hough_solver.plot(triangles[:1, :], self.voxelized.surf_cube[:,:, 2])

        mean_r = triangles[:, 2] * self.voxel_size
        us = self.gripper.get_u(mean_r + self.voxel_size/2)

        return triangles[mask], contact_point_indices[mask], us[mask]

    def calculate_max_depth(self, triangles, contact_point_indices, us):
        voxel_size = self.voxel_size

        vaild_graps_indices = []
        for tri, cpi, u in zip(triangles, contact_point_indices, us):
            current_grasp_depth = cpi[0, -1] * voxel_size - voxel_size/2
            self.gripper.set_u(u)
            if current_grasp_depth > self.gripper.max_grasp_depth:
                return False # grasp too deeply, which may cause inference
            else:
                # draw lines to estimate where the fingers are
                finger_mask = np.zeros(self.voxelized.entity_cube.shape[:2], np.uint8)
                finger_width = (self.gripper.finger_width + self.gripper.finger_gripping_width)/voxel_size + 1
                finger_thickness = int(max(self.gripper.finger_thickness/voxel_size, 1))

                # phase = np.linspace(tri[-1], tri[-1] + 2*np.pi, 3, endpoint=False)
                v = cpi[:, :2] - tri[:2]
                phase = np.arctan2(v[:,0], v[:,1])
                p1 = cpi[:, :2] + (finger_thickness-1) * np.array([np.sin(phase), np.cos(phase)]).T
                p2 = p1 + finger_width * np.array([np.sin(phase), np.cos(phase)]).T
                p1 = np.around(p1).astype(np.int) #[3, 2]
                p2 = np.around(p2).astype(np.int) #[3, 2]
                for pp1, pp2 in zip(p1, p2):
                    finger_mask = cv2.line(finger_mask, pp1[::-1], pp2[::-1], 1, finger_thickness) #draw a line
                finger_index = np.where(finger_mask)
                # Slice along the z-axis where the gripper's fingers are located
                # the gripper's fingers should not be interference with the object
                z_slices = self.voxelized.erode_entity_cube[finger_index[0], finger_index[1], :] #[n, N]
                z_slices_non_zero_index = np.where(z_slices)[-1]
                if z_slices_non_zero_index.size > 0:
                    potential_max_grasp_depth = z_slices_non_zero_index.min() * voxel_size - voxel_size/2
                else:
                    potential_max_grasp_depth = self.voxelized.shape[2] * voxel_size - voxel_size/2
                # preferred_grasp_depth = current_grasp_depth + self.gripper.finger_gripping_length/2 # the preferred grasp depth, which may be unreachable
                feasible_max_grasp_depth = min(potential_max_grasp_depth, self.gripper.max_grasp_depth) - voxel_size/2 # the feasible_max_grasp_depth
                if feasible_max_grasp_depth - voxel_size < current_grasp_depth:
                    # return False #can not grasp
                    continue # 留有余量
                else:
                    # if len(tri) > 0:
                    #     zi = int(np.round(feasible_max_grasp_depth / voxel_size))
                    #     self.hough_solver.plot(np.expand_dims(tri, 0), 
                    #                            np.sum(self.voxelized.erode_entity_cube[..., :zi], -1))
                    graps_indices = np.array([*tri[0:2], #[::-1], 
                                              np.round(feasible_max_grasp_depth / voxel_size), 
                                              tri[3],
                                              u])
                    vaild_graps_indices.append(graps_indices)
                # adjusted_grasp_depth = min(preferred_grasp_depth, feasible_max_grasp_depth)
        if len(vaild_graps_indices) == 0:
            vaild_graps_indices = np.zeros((0,5), np.float32)
        else:
            vaild_graps_indices = np.stack(vaild_graps_indices) #[x, y, z,]
        return vaild_graps_indices #[y, ]

    def merge_similar(self):
        pass

        # posture_gripper_trans__ = Posture(homomat = np.linalg.multi_dot( (
        #                                                         posture_t.trans_mat, 
        #                                                         posture_z_rot.trans_mat)))         
        # gripper.show_claw(posture_gripper_trans__.trans_mat, geometrys=[surf_normal_voxel_grid])  

        ## score the posture, consider about the voxels near the grasped voxel
        ## the more their normals parallel to the grasp vectors, the higher the score is.
        ## The closer the gripping center is to the center of gravity, the higher the score is.
        scores = []              
        # gather the neighborhood
        for _s in range(3):
            # normal
            _box = np.zeros(surf_box.shape, np.uint8)
            _box[tuple(gig[_s])] = 1
            structure = np.ones((1,1,3), np.uint)
            _dilated_body = ndimage.binary_dilation(_box, structure=structure, iterations=1)
            neighborhood = np.where(_dilated_body)
            neighborhood_indices = grid_indices_box[neighborhood]
            neighborhood_indices = neighborhood_indices[neighborhood_indices != -1]
            neighborhood_normals = grid_surf_normals[neighborhood_indices, :3]
            grasp_vector = grasp_direction[_s]
            grasp_vector = grasp_vector / np.linalg.norm(grasp_vector)
            inner_product = np.sum(neighborhood_normals * grasp_vector) / np.sum(_dilated_body)
            # gravity
            gravity = np.mean(np.array(np.where(body_box)).T, axis = 0)
            force_arm = np.linalg.norm(gig_center[:2] - gravity[:2])
            gravity_score = (-force_arm+self.gripper.finger_gripping_length) / (force_arm+self.gripper.finger_gripping_length)  #in [-1, 1]
            scores.append(inner_product + gravity_score)
        grasp_score = np.mean(scores)
        ## record
        local_grasp_poses[gcount, :3] = posture_gripper_trans.rvec
        local_grasp_poses[gcount, 3:6] = posture_gripper_trans.tvec
        local_grasp_poses[gcount, 6]  = u
        local_grasp_poses[gcount, 7]  = grasp_score

    def restore(self, rvec, graps_indices, us):
        '''
        graps_indices: [N, 5]
        '''
        local_grasp_poses = np.zeros((graps_indices.shape[0], 8))
        for i, gi in enumerate(graps_indices):
            self.gripper.set_u(us[i])
            ### transform index to positions
            center = gi[:3]
            center = self.voxelized.restore_mat[:3,:3].dot(center) + self.voxelized.restore_mat[:3,3]
            center[2] -= self.gripper.finger_gripping_bottom[2]
            angle = gi[3]

            posture_base_rot = Posture(rvec=rvec)
            posture_z_rot = Posture(rvec=np.array([0,0,angle]))
            posture_t = Posture(tvec=center)
            posture_gripper_trans = Posture(homomat = np.linalg.multi_dot( (posture_base_rot.trans_mat, 
                                                                            posture_t.trans_mat, 
                                                                            posture_z_rot.trans_mat)))
            local_grasp_poses[i, :3]    = posture_gripper_trans.rvec
            local_grasp_poses[i, 3:6]   = posture_gripper_trans.tvec
            local_grasp_poses[i, 6]     = us[i]
            local_grasp_poses[i, 7]     = 1.0
        return local_grasp_poses

class CandiCoordCalculator():
    def __init__(self, modelpcd:ObjectPcd, gripper:Gripper, 
                 v_friction_angle = 10, h_friction_angle = 45, voxel_size = -1) -> None:
        self.modelinfo = modelpcd
        self.gripper = gripper
        self.v_friction_angle = v_friction_angle
        self.h_friction_angle = h_friction_angle
        if voxel_size == -1:
            voxel_size = int(self.modelinfo.pointcloud_size.min() / 30) + 1
        self.voxel_size = voxel_size

    def plot(self, obj_rvec, local_grasp_pose, voxelized:Voxelized):
        rvec = local_grasp_pose[:3]
        tvec = local_grasp_pose[3:6]
        u =  local_grasp_pose[6]
        
        # 进行体素化
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxelized.surf_points())
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 3)
        # mesh = self.modelinfo.mesh
        # mesh.compute_vertex_normals()

        self.gripper.set_u(u)
        gripper_o3d_geo, gripper_trans_mat = self.gripper.get_gripper_o3d_geo()

        gripper_posture = Posture(rvec=-obj_rvec) * Posture(rvec= rvec, tvec=tvec)  # * Posture(tvec=[0,0,-gripper.finger_gripping_bottom[2]])
        for geo, mat in zip(gripper_o3d_geo, gripper_trans_mat):
            # mat = np.dot(gripper.posture_WCS.trans_mat, mat)
            geo.transform(gripper_posture.trans_mat)    
        # sence_center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=self.mesh.get_center())
        # sence_robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        gripper_frame.transform(gripper_posture.trans_mat)
        showing_geometrys = [voxel_grid] + gripper_o3d_geo + [gripper_frame]
        
        o3d.visualization.draw_geometries(showing_geometrys, width=1280, height=720)

    def calc_candidate_coord(self, show = False):
        '''
        v_friction_angle  垂直方向死锁角
        h_friction_angle  水平方向死锁角
        voxel_size: 体素化尺寸
        '''
        print("start to calculate for {}".format(self.modelinfo.name))
        ia = SphereAngle()
        voxel_size = self.voxel_size
        found_num = 0
        global_grasp_poses = []

        searcher = _CoordSearcher(self.gripper, None, self.v_friction_angle, self.h_friction_angle, self.voxel_size)
        progress = tqdm(enumerate(ia.rvec), total=len(ia.rvec), leave=True)
        for baserot_i, rvec in progress:
            ### 旋转
            transform = Posture(rvec=-rvec)
            modelinfo = self.modelinfo.transform(transform)
            ### 体素化
            voxelized:Voxelized = Voxelized.from_mesh(modelinfo.mesh, voxel_size)
            
            searcher.set_voxelized(voxelized)
            get_v_stable = searcher.get_v_stable()
            triangles, contacted = searcher.search_by_layer(get_v_stable)
            triangles, contacted, us = searcher.get_h_stable_grasp(triangles, contacted)
            vaild_graps_indices = searcher.calculate_max_depth(triangles, contacted, us)
            local_grasp_poses = searcher.restore(rvec, vaild_graps_indices, us)
            # for lp in local_grasp_poses:
            #     self.plot(rvec, lp, voxelized)
            global_grasp_poses.append(local_grasp_poses)
            found_num += len(local_grasp_poses)
            progress.set_postfix({"grasping coordinates found": found_num})

        global_grasp_poses = np.concatenate(global_grasp_poses, 0)
        save_path = os.path.join(MODELS_DIR, self.modelinfo.name + "_candi_grasp_posture" + ".npy")
        np.save(save_path, global_grasp_poses, allow_pickle= True)
        print("saved at {}".format(save_path))
        
        self.modelinfo.candi_coord_parameter = global_grasp_poses

        if show:
            self.modelinfo.draw_all(self.gripper)

