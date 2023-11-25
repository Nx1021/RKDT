from posture_6d.core.posture import Posture
from posture_6d.data.mesh_manager import MeshMeta

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

from .object_pcd import ObjectPcd, create_ObjectPcd_from_file
from .gripper import Gripper, MyThreeFingerGripper
from . import MODELS_DIR, SCRIPT_DIR, LOGS_DIR


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
    with np.errstate(divide='ignore', invalid='ignore'):
        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm

    v1_normalized[np.isnan(v1_normalized)] = 0.0
    v2_normalized[np.isnan(v2_normalized)] = 0.0

    # 计算夹角的余弦值
    cos_angle = np.sum(v1_normalized * v2_normalized, axis=-1)

    # 夹角的弧度值
    angle = np.arccos(cos_angle)

    return angle

class InitAngle:
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

    @staticmethod
    def get_rvec_from_destination(dest, base = [0,0,1]):
        ### 转换为旋转向量
        base = np.tile(base, [dest.shape[0],1]).astype(np.float32)
        times = np.sum( dest * base, axis=-1)
        angle = np.arccos(times) #旋转角度
        rot = np.cross(base, dest)
        return rot * np.tile(np.expand_dims(angle/ np.linalg.norm(rot, axis=-1), -1), [1,3])

    def get_rvec(self):   
        vecs = self.vecs 
        self.rvec = self.get_rvec_from_destination(vecs)

class SphereAngle(InitAngle):
    '''
    在球面上大致均匀分布的角度序列
    '''
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
        self.get_rvec()

class Voxelized():
    '''
    brief
    -----
    体素化对象

    attr
    -----
    * cube

    entity_cube:   实体立方, uint8 ndarray, shape = [L, W, H]

    surf_cube:     表面立方, uint8 ndarray, shape = [L, W, H]

    inner_cube:    内核立方, uint8 ndarray, shape = [L, W, H]

    entity_cube = surf_cube + inner_cube

    * indices: 在cube内为真的位置的索引序列

    entity_indices: 实体索引 int64 ndarray, shape = [-1, 3]

    surf_indices:   表面索引 int64 ndarray, shape = [-1, 3]
    
    * R³

    surf_points:    表面点的空间坐标 float32 ndarray shape = [-1, 3]

    surf_normals:   表面点的空间法矢 float32 ndarray shape = [-1, 3]

    * query: 用于查询立方索引对应的序列中的位置

    _entity_query

    _surf_query

    * restore_mat: 从surf_indices转化为surf_points的变换矩阵
    * orig_mesh: 原始的TriangleMesh对象
    '''
    def __init__(self, 
                 entity_cube:np.ndarray,
                 restore_mat:np.ndarray,
                 orig_mesh = None) -> None:
        self.entity_cube         = entity_cube.astype(np.uint8)
        self.restore_mat = restore_mat          
        self.orig_mesh = orig_mesh
      
        self.surf_cube, self.inner_cube = self.split_entity_cube(entity_cube)
        
        self.entity_indices =\
              np.array(np.where(self.entity_cube)).T #[N, 3]
        self.surf_indices =\
              np.array(np.where(self.surf_cube)).T
        
        self.surf_points =\
              self.restore_mat[:3, :3].dot(self.surf_indices.T).T + self.restore_mat[:3, 3]
        self.surf_normals = self.calc_surf_normal()

        self._entity_query = np.full(self.entity_cube.shape, -1, np.int64)
        self._entity_query[self.entity_indices[:,0], 
                        self.entity_indices[:,1], 
                        self.entity_indices[:,2]] = np.array(range(self.entity_indices.shape[0]), np.int64)
        
        self._surf_query = np.full(self.surf_cube.shape, -1, np.int64)
        self._surf_query[self.surf_indices[:,0], 
                        self.surf_indices[:,1], 
                        self.surf_indices[:,2]] = np.array(range(self.surf_indices.shape[0]), np.int64)

    @property
    def shape(self):
        return self.entity_cube.shape
 
    def split_entity_cube(self, cube:np.ndarray):
        '''
        brief
        -----
        由entity_cube生成surf_cube和inner_cube
        '''
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
        '''
        brief
        ----
        计算表面法矢，找到距离体素化表面的每个点最近的原始mesh的点，并复制其法矢
        '''
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

        # 归一化
        interpolated_normals = interpolated_normals / np.linalg.norm(interpolated_normals, axis=-1, keepdims=True)
        surf.normals = o3d.utility.Vector3dVector(interpolated_normals)

        return interpolated_normals
    
    def query_surf_normal(self, indices:np.ndarray) -> np.ndarray:
        '''
        brief
        -----
        查询表面法矢

        parameter
        -----
        indices: ndarray, shape:[..., 3]
        '''
        idx = self._surf_query[indices[..., 0], indices[..., 1], indices[..., 2]]
        normals = self.surf_normals[idx]
        normals[np.any(idx == -1, -1)] = 0.0
        return normals
    
    def query_surf_points(self, indices:np.ndarray) -> np.ndarray:
        '''
        brief
        -----
        查询表面点坐标

        parameter
        -----
        indices: ndarray, shape:[..., 3]
        '''
        idx = self._surf_query[indices[..., 0], indices[..., 1], indices[..., 2]]
        points = self.surf_points[idx]
        points[np.any(idx == -1, -1)] = 0.0
        return points

    @staticmethod
    def from_mesh(mesh, voxel_size):
        '''
        brief
        -----
        体素化TriangleMesh对象

        parameter
        -----
        mesh: open3d.geometry.TriangleMesh 单位为mm
        voxel_size: int|float 单位为mm
        '''
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
        entity = Voxelized.fill_3d(_entity)

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
    def fill_3d(voxel_array:np.ndarray):
        '''
        brief
        ----
        填充体素化点云
        沿不同方向累加，累加值为奇数的是内部
        '''
        padded_array = np.pad(voxel_array, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        # padded_array = padded_array.astype(np.int8)
        cum_list = []
        for i in range(3):
            padding = tuple([(1,0) if p == i else (0,0) for p in range(3)])
            padded_array = np.pad(voxel_array, padding, mode='constant', constant_values=0)
            diff = np.diff(padded_array, axis = i)
            # diff = np.swapaxes(diff, 0, i)[:-1]
            # diff = np.swapaxes(diff, 0, i)
            diff = diff > 0
            cum = (np.cumsum(diff, axis=i) / 2).astype(np.uint16)
            cum_list.append(cum)
        cum = np.stack(cum_list) # [3, W, H, D]
        odd = np.mod(cum, 2) == 1 # [3, W, H, D]
        in_voting = np.sum(odd, axis=0) > 2

        entity = voxel_array.copy()
        entity[in_voting] = 1
        return entity

class Triangle_Hough():
    '''
    brief
    -----
    正三角形霍夫变换
    '''
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

    def _crop_nonzero_region(self, image):
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

    def plot(self, triangles:np.ndarray, image:cv2.Mat, show = True):
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
        if show:
            plt.show()

    def _parse(self, result):
        rsls = np.array([
            self.resolution, 
            self.resolution, 
            self.resolution, 
            self.theta_rsl])
        offset = np.array([0, 0, self.r_min, 0])
        triangles = result * rsls + offset
        return triangles

    def triangles_filter(self, triangles, co_marks):
        '''
        部分结果不合理，将其过滤
        '''
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
        '''
        brief
        ----
        运行

        parameter
        -----
        image: 图像，必须是二值化的 \n
        ifplot: 是否绘制结果

        note
        -----

        x,y 表示在当前图像坐标系下的x,y

        X,Y 表示在物体坐标系下的X,Y

        y == X 
        x == Y
        '''
        start = time.time()
        croped, org = self._crop_nonzero_region(image)

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

            theta = np.arctan2(diff[:, 1] , diff[:, 0]) # 为了后续处理方便，定义theta = arctan(dx/dy) = arctan(dY/dX)
            theta = np.mod(theta, 2*np.pi/3)
            theta_quant = np.round(theta / self.theta_rsl).astype(np.int32) #[P]
            theta_quant[theta_quant == self.theta_step] = 0

            hough_space[points[:,0], points[:,1], r_quant, theta_quant] = 1
            hough_space = hough_space[:, :, :-1, :]
            hough_space_list.append(hough_space)
            summary += hough_space

        condition = np.where(summary == 3)
        result = np.array(condition).T
        triangles = self._parse(result)

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
        DO NOT USE
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

            max_proj = max_proj + self.voxelized.inner_cube[:,:,i].astype(np.uint8) #最大投影轮廓，被包含在内的点将被排除
        contacted = np.concatenate(contacted)
        triangles = np.concatenate(triangles)
        return triangles, contacted

    def get_h_stable_grasp(self, triangles, contact_point_indices):
        '''
        triangles: [N, 4]
        contact_point_indices: [N, 3, 2]
        '''
        contact_point_indices = contact_point_indices.astype(np.int32)
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
        '''
        计算夹持器最大可达深度
        '''
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
                p1 = np.around(p1).astype(np.int32) #[3, 2]
                p2 = np.around(p2).astype(np.int32) #[3, 2]
                for pp1, pp2 in zip(p1, p2):
                    finger_mask = cv2.line(finger_mask, pp1[::-1], pp2[::-1], 1, finger_thickness) #draw a line
                finger_index = np.where(finger_mask)
                # Slice along the z-axis where the gripper's fingers are located
                # the gripper's fingers should not be interference with the object
                z_slices = self.voxelized.inner_cube[finger_index[0], finger_index[1], :] #[n, N]
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
                    #                            np.sum(self.voxelized.inner_cube[..., :zi], -1))
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

    def restore(self, rvec, graps_indices, us):
        '''
        brief
        -----
        恢复夹持坐标到物体坐标系(OCS)

        parameter
        ----
        rvec: 旋转向量: shape = [3]
        graps_indices: shape = [N, 5]
        us: 夹持参数
        '''
        local_grasp_poses = np.zeros((graps_indices.shape[0], 8))
        for i, gi in enumerate(graps_indices):
            self.gripper.set_u(us[i])
            ### transform index to positions
            center = gi[:3].copy()
            center[2] += 0.5
            center = self.voxelized.restore_mat[:3,:3].dot(center) + self.voxelized.restore_mat[:3,3]
            center[2] -= self.gripper.finger_gripping_bottom[2]
            angle = gi[3]

            posture_base_rot = Posture(rvec=rvec)
            posture_z_rot = Posture(rvec=np.array([0,0,angle - self.gripper.rot_bias]))
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
        '''
        v_friction_angle  垂直方向死锁角

        h_friction_angle  水平方向死锁角

        voxel_size: 体素化尺寸
        '''        
        self.modelinfo = modelpcd
        self.gripper = gripper
        self.v_friction_angle = v_friction_angle
        self.h_friction_angle = h_friction_angle
        if voxel_size == -1:
            voxel_size = int(self.modelinfo.pointcloud_size.min() / 30) + 1
        self.voxel_size = voxel_size

    def calc_candidate_coord(self, show = False, create_process_mesh = False):
        '''
        brief
        -----
        开始计算

        parameters
        ----
        show: 计算完毕以后显示结果

        create_process_mesh: 创建过程mesh以供观阅
        '''
        def _create_process_mesh():
            '''
            绘制中间过程以供观阅
            '''
            def get_gripper_mesh(lgp):
                g_rvec = lgp[:3]
                g_tvec = lgp[3:6]
                g_u =  lgp[6]

                self.gripper.set_u(g_u)
                # self.gripper.set_u(0)
                gripper_o3d_geo, gripper_trans_mat = self.gripper.get_gripper_o3d_geo()

                gripper_posture = Posture(rvec= g_rvec, tvec=g_tvec)  # * Posture(tvec=[0,0,-gripper.finger_gripping_bottom[2]])
                for geo, mat in zip(gripper_o3d_geo, gripper_trans_mat):
                    # mat = np.dot(gripper.posture_WCS.trans_mat, mat)
                    geo.transform(gripper_posture.trans_mat)    

                gripper_combined = o3d.geometry.TriangleMesh()
                for mesh in gripper_o3d_geo:
                    gripper_combined += mesh 
                return gripper_combined 
            
            def get_arrow_mesh(start, end):
                # 创建箭头几何体对象
                diff = end - start
                normed = diff / np.linalg.norm(diff)
                arrow_length = np.linalg.norm(diff)
                arrow_rvec = InitAngle.get_rvec_from_destination(np.expand_dims(normed, 0))
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=arrow_length * 0.05,
                    cone_radius=arrow_length * 0.1,
                    cylinder_height=arrow_length * 0.8,
                    cone_height=arrow_length * 0.2
                )
                arrow.transform(Posture(rvec=arrow_rvec, tvec=start).trans_mat)
                return arrow

            orig_z = contacted[0,0,2]
            #### 创建正方体组合mesh, 为每个体素创建一个立方体网格
            # 分别创建上、中、下3个部分
            voxel_size = voxelized.restore_mat[0,0]
            mesh_cube = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
            meshes_dict:dict[str, list] = {}
            mesh_part_names = ["below", "mid", "above"]
            for voxel, ind in zip(voxelized.surf_points, voxelized.surf_indices):
                _mesh_cube = o3d.geometry.TriangleMesh(mesh_cube)
                mesh = _mesh_cube.translate(voxel - np.array([voxel_size, voxel_size, voxel_size])/2)
                if ind[2] < orig_z:
                    meshes_dict.setdefault(mesh_part_names[0], []).append(mesh)
                elif ind[2] == orig_z:
                    meshes_dict.setdefault(mesh_part_names[1], []).append(mesh)
                elif ind[2] > orig_z:
                    meshes_dict.setdefault(mesh_part_names[2], []).append(mesh)
            # 合并网格对象
            combined = {key: value for key, value in zip(mesh_part_names, [o3d.geometry.TriangleMesh() for _ in range(3)])}
            for key, meshes in meshes_dict.items():
                for mesh in meshes:
                    combined[key] += mesh
                combined[key].compute_triangle_normals()
            
            local_grasp_pose = searcher.restore(np.array([0, 0, 0.0]), vaild_graps_indices, us)
            vaild_graps_indices_orig = vaild_graps_indices.copy()
            vaild_graps_indices_orig[:, 2] = orig_z
            local_grasp_pose_orig = searcher.restore(np.array([0, 0, 0.0]), vaild_graps_indices_orig, us)
            
            grasp = get_gripper_mesh(local_grasp_pose[0])
            grasp_orig = get_gripper_mesh(local_grasp_pose_orig[0])
            rot_mesh = o3d.geometry.TriangleMesh(modelinfo.mesh)

            # 接触点法向箭头
            normals = voxelized.query_surf_normal(contacted[0])
            start = voxelized.query_surf_points(contacted[0])
            end = start + normals * voxel_size * 7
            normals_arrow = o3d.geometry.TriangleMesh()
            for s, e in zip(start, end):
                arrow = get_arrow_mesh(s, e)
                normals_arrow += arrow
            # 夹持方向箭头
            grasp_dire = contacted[0][:, :2] - triangles[0][:2]
            grasp_dire = np.pad(grasp_dire, ((0,0), (0,1)), constant_values=0.0)
            grasp_dire = grasp_dire / np.linalg.norm(grasp_dire, axis=-1, keepdims=True)
            start = voxelized.query_surf_points(contacted[0])
            end = start + grasp_dire * voxel_size * 7
            gripper_arrow = o3d.geometry.TriangleMesh()
            for s, e in zip(start, end):
                arrow = get_arrow_mesh(s, e)
                gripper_arrow += arrow

            for m in [*combined.values(), grasp, grasp_orig, rot_mesh, normals_arrow, gripper_arrow]:
                m.compute_triangle_normals()
                m.transform(Posture(rvec = [np.pi, 0, 0]).trans_mat)

            os.makedirs(f"{LOGS_DIR}/grasp_illustration", exist_ok=True)
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/voxelized_above.stl", combined[mesh_part_names[0]])
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/voxelized_mid.stl", combined[mesh_part_names[1]])
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/voxelized_below.stl", combined[mesh_part_names[2]])
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/maxdepth_grasp.stl", grasp)
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/orig_grasp.stl", grasp_orig)
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/contact_mesh_normals.stl", normals_arrow)
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/contact_gripper_normals.stl", gripper_arrow)
            # 切片
            searcher.hough_solver.plot(triangles[0:1, :], voxelized.surf_cube[:,:,orig_z], False)
            plt.savefig(f"{LOGS_DIR}/grasp_illustration/layer.svg")
            # 原模型
            self.modelinfo.mesh.compute_triangle_normals()
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/mesh.stl", self.modelinfo.mesh)
            # 旋转模型
            modelinfo.mesh.compute_triangle_normals()
            o3d.io.write_triangle_mesh(f"{LOGS_DIR}/grasp_illustration/rot_mesh.stl", rot_mesh)

        print("start to calculate for {}".format(self.modelinfo.name))
        ia = SphereAngle()
        voxel_size = self.voxel_size
        found_num = 0
        global_grasp_poses = []

        searcher = _CoordSearcher(self.gripper, None, self.v_friction_angle, self.h_friction_angle, self.voxel_size) # 搜索器
        progress = tqdm(enumerate(ia.rvec), total=len(ia.rvec), leave=True)
        
        ### 逐角度循环搜索
        for baserot_i, rvec in progress:
            ### 旋转
            transform = Posture(rvec=-rvec)
            modelinfo = self.modelinfo.transform(transform)
            ### 体素化
            voxelized:Voxelized = Voxelized.from_mesh(modelinfo.mesh, voxel_size)
            
            ### 逐层查找
            searcher.set_voxelized(voxelized)
            get_v_stable = searcher.get_v_stable()
            triangles, contacted = searcher.search_by_layer(get_v_stable)
            triangles, contacted, us = searcher.get_h_stable_grasp(triangles, contacted)
            vaild_graps_indices = searcher.calculate_max_depth(triangles, contacted, us)
            local_grasp_poses = searcher.restore(rvec, vaild_graps_indices, us)

            ### 绘制中间过程（可选的）
            if create_process_mesh and len(local_grasp_poses) > 0:
                _create_process_mesh()

            ### 收集结果
            global_grasp_poses.append(local_grasp_poses)
            found_num += len(local_grasp_poses)
            progress.set_postfix({"grasping coordinates found": found_num})

        ### 保存结果
        global_grasp_poses = np.concatenate(global_grasp_poses, 0) # 合并
        save_path = os.path.join(MODELS_DIR, self.modelinfo.name + "_candi_grasp_posture" + ".npy")
        np.save(save_path, global_grasp_poses, allow_pickle= True)
        print("saved at {}".format(save_path))
        
        self.modelinfo.candi_coord_parameter = global_grasp_poses

        ### 显示结果（可选的）
        if show:
            self.modelinfo.draw_all(self.gripper)

