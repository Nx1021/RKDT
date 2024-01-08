from posture_6d.core.posture import Posture
from posture_6d.data.mesh_manager import MeshMeta

import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage
from scipy.optimize import root
from scipy.spatial import ConvexHull
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

from typing import Optional, Callable

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
    def __init__(self, nums_points = 500) -> None:
        nums_points = nums_points
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
                 restore_mat:np.ndarray, * ,  
                 surf_cube:Optional[np.ndarray] = None,
                 entity_cube:Optional[np.ndarray] = None,
                 surf_normals_cube:Optional[np.ndarray] = None,
                 orig_mesh = None) -> None:
        assert restore_mat.shape == (4, 4), "restore_mat must be 4x4"
        assert entity_cube is not None or surf_cube is not None, "entity_cube and surf_cube cannot be None at the same time"
        
        self.orig_mesh = orig_mesh
        self.restore_mat = restore_mat          
        
        if entity_cube is not None:
            self.entity_cube = entity_cube.astype(np.uint8) if entity_cube.dtype != np.uint8 else entity_cube
            if surf_cube is None:
                self.surf_cube, self.inner_cube = self._split_entity_cube(entity_cube)
            else:
                self.surf_cube = surf_cube.astype(np.uint8) if surf_cube.dtype != np.uint8 else surf_cube
                self.inner_cube = entity_cube - surf_cube
        else:
            self.surf_cube = surf_cube.astype(np.uint8) if surf_cube.dtype != np.uint8 else surf_cube
            if entity_cube is None:
                self.entity_cube = self.fill_3d(surf_cube)
            else:
                self.entity_cube = entity_cube.astype(np.uint8) if entity_cube.dtype != np.uint8 else entity_cube
            self.inner_cube = entity_cube - surf_cube
        
        self.entity_indices =\
            np.array(np.where(self.entity_cube)).T #[N, 3]
        self.surf_indices =\
            np.array(np.where(self.surf_cube)).T
        
        self.surf_points =\
            self.restore_mat[:3, :3].dot(self.surf_indices.T).T + self.restore_mat[:3, 3] # 表面点坐标
        
        if surf_normals_cube is not None:
            self.surf_normals = surf_normals_cube[self.surf_indices[:,0], self.surf_indices[:,1], self.surf_indices[:,2]]
        else:
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
 
    @property
    def voxel_size(self):
        return self.restore_mat[0, 0]
    
    def _split_entity_cube(self, cube:np.ndarray):
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

    def visualize(self):
        voxel_size = self.voxel_size

        current_layer:int = 0
        show_layer = False

        surf_voxel_mesh_slice = {}
        surf_voxel_mesh_slice_visible = {}
        for idx in self.surf_indices:
            mesh_cube = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
            mesh_cube.compute_vertex_normals()
            surf_voxel_mesh_slice.setdefault(idx[-1], o3d.geometry.TriangleMesh())
            surf_voxel_mesh_slice[idx[-1]] += mesh_cube.translate(idx * voxel_size)
            surf_voxel_mesh_slice_visible.setdefault(idx[-1], True)

        entity_voxel_mesh_slice = {}
        entity_voxel_mesh_slice_visible = {}
        for idx in self.entity_indices:
            mesh_cube = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
            mesh_cube.compute_vertex_normals()
            entity_voxel_mesh_slice.setdefault(idx[-1], o3d.geometry.TriangleMesh())
            entity_voxel_mesh_slice[idx[-1]] += mesh_cube.translate(idx * voxel_size)
            entity_voxel_mesh_slice_visible.setdefault(idx[-1], True)

        current_mesh = surf_voxel_mesh_slice
        current_mesh_visible = surf_voxel_mesh_slice_visible

        def hide_all(vis):
            for m, v in zip((surf_voxel_mesh_slice, entity_voxel_mesh_slice), (surf_voxel_mesh_slice_visible, entity_voxel_mesh_slice_visible)):
                for idx in m.keys():
                    visible = v.get(idx)
                    if visible:
                        mesh = m.get(idx)
                        mesh.scale(1/2**10, np.array([0.0,0,0]))  
                        vis.update_geometry(mesh)
                        v[idx] = False            

        def make_visible(vis, mesh_idx):
            idx = mesh_idx
            mesh = current_mesh.get(idx)
            visible = current_mesh_visible.get(idx)
            if not visible:
                mesh.scale(2**10, np.array([0.0,0,0]))  
                vis.update_geometry(mesh)
                current_mesh_visible[idx] = True

        def display(vis):
            nonlocal current_layer, show_layer
            if not show_layer:
                # 显示完整，将所有mesh添加
                hide_all(vis)
                for mesh_idx in current_mesh.keys():
                    make_visible(vis, mesh_idx)
            else:
                # 显示单层，删除所有mesh，添加单层mesh
                hide_all(vis)
                make_visible(vis, current_layer)

        def toggle_layer_mode(vis):
            nonlocal show_layer
            if not show_layer:
                show_layer = True
            else:
                show_layer = False
            display(vis)

        def toggle_mesh_mode(vis):
            nonlocal current_mesh, current_mesh_visible
            if current_mesh == surf_voxel_mesh_slice:
                current_mesh = entity_voxel_mesh_slice
                current_mesh_visible = entity_voxel_mesh_slice_visible
            else:
                current_mesh = surf_voxel_mesh_slice
                current_mesh_visible = surf_voxel_mesh_slice_visible

            display(vis)

        # 回调函数：层数上移
        def move_up(vis):
            nonlocal current_layer, show_layer
            current_layer += 1
            if current_layer >= self.shape[-1]:
                current_layer = 0

            if show_layer:
                hide_all(vis)
                make_visible(vis, current_layer)

        # 回调函数：层数下移
        def move_down(vis):
            nonlocal current_layer
            current_layer -= 1
            if current_layer < 0:
                current_layer = self.shape[-1] - 1
            
            if show_layer:
                hide_all(vis)
                make_visible(vis, current_layer)

        key_to_callback = {}
        key_to_callback[ord("A")] = toggle_layer_mode
        key_to_callback[ord("D")] = toggle_mesh_mode
        key_to_callback[ord("W")] = move_up
        key_to_callback[ord("S")] = move_down

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=voxel_size * 5, origin=[0, 0, 0])

        o3d.visualization.draw_geometries_with_key_callbacks(list(entity_voxel_mesh_slice.values()) +list(surf_voxel_mesh_slice.values()) + [frame], 
                                                             key_to_callback ,width =640, height=480)
        
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

        ### 创建cube
        # 计算体素网格的尺寸
        voxel_dims = np.ceil((mesh.get_max_bound() - mesh.get_min_bound()) / voxel_size).astype(np.int32) + 1
        # 创建全0的三维矩阵
        surf = np.zeros(voxel_dims, dtype=np.int32) # [W, H, D]
        surf_normals = np.zeros((*list(voxel_dims), 3), dtype=np.float32) # [W, H, D, 3]

        ### 进行体素化
        origin = mesh.get_min_bound() - voxel_size / 2
        restore_mat = np.eye(4)*voxel_size
        restore_mat[3, 3] = 1
        restore_mat[:3, 3] = origin

        vertices = np.asarray(mesh.vertices) # [N, 3]
        normals = np.asarray(mesh.vertex_normals) # [N, 3]
        vertices_idx = np.floor((vertices - origin) / voxel_size).astype(np.int32) # [N, 3] int32
        
        surf[vertices_idx[:, 0], vertices_idx[:, 1], vertices_idx[:, 2]] = 1
        np.add.at(surf_normals, (vertices_idx[:, 0], vertices_idx[:, 1], vertices_idx[:, 2]), normals)
        surf_normals = surf_normals / np.linalg.norm(surf_normals, axis=-1, keepdims=True)

        ### 填充实体     
        surf = surf.astype(np.uint8)  
        entity = np.zeros_like(surf, dtype = np.uint8)
        xy_exterior = np.zeros_like(surf, dtype = np.uint8)
        for i, slice in enumerate(surf.transpose(2, 0, 1)):
            # 创建一个与输入图像大小相同的全零图像
            contours, hierarchy = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            filled_image = np.zeros_like(slice)
            entity[:, :, i] = cv2.drawContours(filled_image, contours, -1, 1, thickness=cv2.FILLED)
            xy_exterior_image = np.zeros_like(slice)
            xy_exterior[:, :, i] = cv2.drawContours(xy_exterior_image, contours, -1, 1, thickness=1)

        return Voxelized(restore_mat, surf_cube=xy_exterior, entity_cube=entity, orig_mesh = mesh, surf_normals_cube = surf_normals)

    @staticmethod
    def fill_3d(voxel_array:np.ndarray):
        '''
        brief
        ----
        填充体素化点云
        沿不同方向累加，累加值为奇数的是内部
        '''
        # padded_array = np.pad(voxel_array, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        # padded_array = padded_array.astype(np.int8)
        cum_list = []
        for i in range(3):
            padding = tuple([(1,0) if p == i else (0,0) for p in range(3)])
            padded_array = np.pad(voxel_array, padding, mode='constant', constant_values=0)
            diff = np.diff(padded_array, axis = i)
            # diff = np.swapaxes(diff, 0, i)[:-1]
            # diff = np.swapaxes(diff, 0, i)
            # diff = diff > 0
            # cum = (np.cumsum(diff, axis=i) / 2).astype(np.uint16)
            cum = np.cumsum(diff, axis=i)
            cum_list.append(cum)
        cum = np.stack(cum_list) # [3, W, H, D]
        # odd = np.mod(cum, 2) == 1 # [3, W, H, D]
        # odd = cum == 1 # [3, W, H, D]
        in_voting = np.sum(cum, axis=0) > 2

        entity = voxel_array.copy()
        entity[in_voting] = 1
        return entity

    @staticmethod
    def fill_2d(voxel_array:np.ndarray):
        '''
        brief
        -----
        逐层填充

        parameter
        ----
        voxel_array: ndarray
            shape = [L, W, H]
        '''
        entity = np.zeros_like(voxel_array)
        for i, slice in enumerate(voxel_array.transpose(2, 0, 1)):
            # 创建一个与输入图像大小相同的全零图像
            contours, hierarchy = cv2.findContours(slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            filled_image = np.zeros_like(slice)
            entity[:, :, i] = cv2.drawContours(filled_image, contours, -1, 1, thickness=cv2.FILLED)

        return entity

class VoxelizedVisualizer():
    def __init__(self, voxelized:Voxelized) -> None:
        self.voxelized = voxelized
        self.layer_num = 0

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
    def __init__(self, gripper:Gripper, voxelized:Voxelized, voxel_size, friction_angle, h_friction_angle, v_friction_angle) -> None:
        
        self.voxelized: Voxelized = voxelized
        self.gripper = gripper

        self.friction_angle = friction_angle
        self.h_friction_angle = h_friction_angle
        self.v_friction_angle = v_friction_angle
        self.voxel_size = voxel_size

        self.friction_angle = 0.0

        gripper.set_u(1)
        max_r = abs(gripper.finger_gripping_center[0])
        self.max_grasp_depth_index = int(gripper.max_grasp_depth/voxel_size) #max grasping depth in voxel
        gripper.set_u(0)
        min_r = abs(gripper.finger_gripping_center[0]) + 10
        self.hough_solver = Triangle_Hough(1, max_r/voxel_size, min_r/voxel_size, 2)

    def set_voxelized(self, voxelized:Voxelized):
        self.voxelized = voxelized

    def get_v_stable(self):
        # 摩擦锥是已知的，由于使用平行夹持器，夹持力的方向只能是水平的，因此可以先判定摩擦锥有没有于平面相交
        ### 计算具有垂直法向的点的 cube
        grasp_vec = np.array([0,0,1])
        # vertical_indices = self.voxelized.surf_indices[np.where(
        #     compute_angle(self.voxelized.surf_normals, grasp_vec) < (90 + self.v_friction_angle)*np.pi/180)[0]] # 近似垂直于抓取方向的点          
        vertical_indices = self.voxelized.surf_indices[np.where(
            np.logical_and(
                (compute_angle(self.voxelized.surf_normals, grasp_vec) < (np.pi/2 + self.v_friction_angle)),
                (compute_angle(self.voxelized.surf_normals, grasp_vec) > (np.pi/2 - self.v_friction_angle))))[0]] # 近似垂直于抓取方向的点          
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
        mask = np.logical_not(np.any(h_angle > self.h_friction_angle, axis=-1))
        
        # self.hough_solver.plot(triangles[:1, :], self.voxelized.surf_cube[:,:, 2])

        mean_r = triangles[:, 2] * self.voxel_size
        us = self.gripper.get_u(mean_r + self.voxel_size/2)

        return triangles[mask], contact_point_indices[mask], us[mask]

    def calculate_max_depth(self, triangles, contact_point_indices, us):
        '''
        计算夹持器最大可达深度
        '''
        voxel_size = self.voxel_size

        vaild_graps_indices:list[GraspCoord] = []
        for tri, cpi, u in zip(triangles, contact_point_indices, us):
            # tri [4]
            # cpi [3, 3]
            # u [1]
            current_grasp_depth = cpi[0, -1] * voxel_size - voxel_size/2
            self.gripper.set_u(u)
            if current_grasp_depth > self.gripper.max_grasp_depth:
                return False # grasp too deeply, which may cause inference
            else:
                # draw lines to estimate where the fingers are
                finger_mask = np.zeros(self.voxelized.entity_cube.shape[:2], np.uint8)
                finger_width = (self.gripper.finger_width + self.gripper.finger_gripping_width)/voxel_size + 1
                finger_thickness = int(max(self.gripper.finger_thickness/voxel_size, 1))

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
                    max_depth_idx = int(np.round(feasible_max_grasp_depth / voxel_size))
                    graspcoord = GraspCoord()
                    graspcoord.contact_points = cpi # [3, 3]
                    graspcoord.grasp_center = np.array([tri[0], tri[1], max_depth_idx]) # [3]
                    graspcoord.contact_normals = self.voxelized.query_surf_normal(graspcoord.contact_points) # [3, 3]
                    graspcoord.grasp_angle = tri[3]
                    graspcoord.u = u
                    # graps_indices = np.array([*tri[0:2],
                    #                           np.round(feasible_max_grasp_depth / voxel_size), 
                    #                           tri[3],
                    #                           u])
                    # vaild_graps_indices.append(graps_indices)
                    vaild_graps_indices.append(graspcoord)
                # adjusted_grasp_depth = min(preferred_grasp_depth, feasible_max_grasp_depth)
        # if len(vaild_graps_indices) == 0:
        #     vaild_graps_indices = np.zeros((0,5), np.float32)
        # else:
        #     vaild_graps_indices = np.stack(vaild_graps_indices) #[x, y, z,]
        return vaild_graps_indices #[y, ]

    def merge_similar(self):
        pass

    def quality_evaluation(self):
        pass
        

    def restore_as_coord(self, grapscoords:list["GraspCoord"]):
        '''
        brief
        -----
        恢复夹持坐标到临时坐标系

        parameter
        ----
        rvec: 旋转向量: shape = [3]
        graps_indices: shape = [N, 5]
        us: 夹持参数
        '''
        restored_grapscoords:list["GraspCoord"] = []

        for i, gc in enumerate(grapscoords):
            self.gripper.set_u(gc.u)
            ### transform index to positions
            center = gc.grasp_center.copy()
            center[2] += 0.5
            center = self.voxelized.restore_mat[:3,:3].dot(center) + self.voxelized.restore_mat[:3,3]
            center[2] -= self.gripper.finger_gripping_bottom[2]
            angle = gc.grasp_angle
            posture_z_rot = Posture(rvec=np.array([0,0,angle - self.gripper.rot_bias]))
            posture_t = Posture(tvec=center)
            posture_gripper_trans = Posture(homomat = np.linalg.multi_dot( (posture_t.trans_mat, 
                                                                            posture_z_rot.trans_mat)))
            # 还原接触点
            restored_grapscoord = GraspCoord(copy_from=gc)
            restored_grapscoord.contact_points  = self.voxelized.restore_mat[:3,:3].dot(gc.contact_points.T).T + self.voxelized.restore_mat[:3,3]
            restored_grapscoord.contact_normals = self.voxelized.restore_mat[:3,:3].dot(gc.contact_normals.T).T
            restored_grapscoord.grasp_center    = self.voxelized.restore_mat[:3,:3].dot(gc.grasp_center) + self.voxelized.restore_mat[:3,3]
            restored_grapscoord.grasp_angle     = gc.grasp_angle
            restored_grapscoord.grasp_posture   = posture_gripper_trans
            # restored_grapscoord.transform(posture_base_rot.trans_mat)
            # geos = self.gripper.render(posture_gripper_trans, gc.u)
            # contact_point_0 = o3d.geometry.TriangleMesh.create_sphere(radius=3).translate(restored_grapscoord.contact_points[0])
            # contact_point_1 = o3d.geometry.TriangleMesh.create_sphere(radius=3).translate(restored_grapscoord.contact_points[1])
            # contact_point_2 = o3d.geometry.TriangleMesh.create_sphere(radius=3).translate(restored_grapscoord.contact_points[2])
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
            # mesh = o3d.geometry.TriangleMesh(self.voxelized.orig_mesh)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(self.voxelized.surf_points)
            # o3d.visualization.draw_geometries(geos + [contact_point_0, contact_point_1, contact_point_2, frame, mesh, pcd])
            restored_grapscoords.append(restored_grapscoord)
        return restored_grapscoords

    def restore_to_Object(self, rvec, grapscoords:list["GraspCoord"]):
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
        local_grasp_poses = np.zeros((len(grapscoords), 8))
        restored_grapscoords_O:list["GraspCoord"] = []

        for i, gc in enumerate(grapscoords):
            posture_base_rot = Posture(rvec=rvec)
            posture_in_Temp = gc.grasp_posture
            posture_gripper_trans = Posture(homomat = np.linalg.multi_dot( (posture_base_rot.trans_mat, 
                                                                            posture_in_Temp.trans_mat)))
            local_grasp_poses[i, :3]    = posture_gripper_trans.rvec
            local_grasp_poses[i, 3:6]   = posture_gripper_trans.tvec
            local_grasp_poses[i, 6]     = gc.u
            local_grasp_poses[i, 7]     = 1.0

            # 还原接触点
            restored_grapscoord = GraspCoord(copy_from=gc)
            restored_grapscoord.transform(posture_base_rot.trans_mat)
            # geos = self.gripper.render(posture_gripper_trans, gc.u)
            # contact_point_0 = o3d.geometry.TriangleMesh.create_sphere(radius=3).translate(restored_grapscoord.contact_points[0])
            # contact_point_1 = o3d.geometry.TriangleMesh.create_sphere(radius=3).translate(restored_grapscoord.contact_points[1])
            # contact_point_2 = o3d.geometry.TriangleMesh.create_sphere(radius=3).translate(restored_grapscoord.contact_points[2])
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
            # mesh = o3d.geometry.TriangleMesh(self.voxelized.orig_mesh)
            # mesh.transform(posture_base_rot.trans_mat)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(self.voxelized.surf_points)
            # pcd.transform(posture_base_rot.trans_mat)
            # o3d.visualization.draw_geometries(geos + [contact_point_0, contact_point_1, contact_point_2, frame, mesh, pcd])
            restored_grapscoords_O.append(restored_grapscoord)
        return local_grasp_poses, restored_grapscoords_O

class ForceClosureEvaluator():
    '''
    My Method
    '''
    def __init__(self, friction_angle:float, h_friction_angle:float, v_friction_angle:float, max_finger_force:float = 1.0, gravity:float = 0.0) -> None:
        '''
        parameters
        -----
        contact_points: ndarray, shape = [N, 3]
            接触点坐标
        contact_normals: ndarray, shape = [N, 3]
            接触法向
        friction_angle: float
            摩擦角
        center: ndarray, shape = [3]
            参考原点
        '''
        self.friction_angle = friction_angle
        self.h_friction_angle = h_friction_angle
        self.v_friction_angle = v_friction_angle
        self._modelinfo:ObjectPcd = None

        self.max_finger_force = max_finger_force
        self.gravity = gravity
    
    @staticmethod
    def calc_wrench(force:np.ndarray, point:np.ndarray, center:np.ndarray = None):
        '''
        parameters
        ----
        force: ndarray
            shape = [..., 3]
        point: ndarray
            shape = [..., 3]
        center: ndarray
            shape = [3]
        '''
        center = np.array([0, 0, 0.0]) if center is None else center

        r = point - center # [..., 3]
        torque = np.cross(r, force) # [..., 3]
        wrench = np.concatenate((torque, force), axis=-1) # [..., 6]
        return wrench
    
    @staticmethod
    def get_wrench_space(bound_wrench:list[np.ndarray]):
        def func(*alphas):
            wrench = np.zeros(6)
            for alpha, bw in zip(alphas, bound_wrench):
                wrench += alpha * bw
            return wrench
        return func

    @staticmethod
    def vector_projection_on_plane(v, normal):
        """
        计算一个空间向量在一个三维空间平面上的投影。
        
        参数:
        v: ndarray, shape=(..., 3)
        要投影的向量
        
        normal: ndarray, shape=(3)
        平面的法向量
        
        返回:
        projected_vector: ndarray, shape=(3,)
        向量在平面上的投影
        """
        
        # 归一化法向量
        normal_unit = normal / np.linalg.norm(normal)
        
        # 计算向量在法向量上的投影
        projection_length = np.dot(v, normal_unit) # [N, 1]
        
        # 计算投影向量
        projected_vector = v - np.matmul( projection_length[..., np.newaxis], normal_unit[np.newaxis, ...])
        
        return projected_vector

    def random_sample_in_wrench_space(self, *wrench_cones:list[tuple[np.ndarray]], sample_num:int = 1000):
        '''
        parameters
        -----
        wrench_cones: list[tuple[np.ndarray]]
            shape = [N, ?, 6]
        
        return
        -----
        wrenches: list[np.ndarray]
            shape = [sample_num, 6]

        '''
        cones_num = len(wrench_cones)
        alphas = np.random.rand(sample_num, cones_num) # [sample_num, N]
        alphas = np.tile(np.expand_dims(alphas, -1), [1, 1, 6]) # [sample_num, N, 6]
        
        wc_samples:list[np.ndarray] = [] # [N, sample_num, 6]
        for i, cone in enumerate(wrench_cones):
            cone_bound_num = len(cone) # [?]
            cone = np.tile(np.expand_dims(cone, 0), [sample_num, 1, 1]) # [sample_num, ?, 6]
            betas = np.random.rand(sample_num, cone_bound_num) # [sample_num, ?]
            betas = np.tile(np.expand_dims(betas, -1), [1,1,6]) # [sample_num, ?, 6]
            wc_sample = np.sum(betas * cone, axis=1) # [sample_num, 6]
            wc_samples.append(wc_sample)
        
        wc_samples = np.swapaxes(np.stack(wc_samples), 0, 1) # [sample_num, N, 6]

        samples = np.sum(alphas * wc_samples, axis=1) # [sample_num, 6]

        return samples

    def get_surface_min_dist(self, samples:np.ndarray):
        '''
        parameters
        -----
        samples: ndarray
            shape = [N, 3]
        '''
        hull = ConvexHull(samples)

        # 绘制三维点集和凸包
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # 绘制点集
        # ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='b', marker='o', s=50, label='Points')

        # # 绘制凸包
        # for simplex in hull.simplices:
        #     simplex = np.append(simplex, simplex[0])  # 将闭合曲线
        #     ax.plot(samples[simplex, 0], samples[simplex, 1], samples[simplex, 2], 'r-')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()
        # plt.title('Convex Hull of 3D Point Set')
        # plt.show()

        # get vertices indices
        vertices = samples[hull.vertices]
        distances = np.linalg.norm(vertices, axis=1)
        min_distance = np.min(distances)

        return min_distance

    def get_v_sub_space_tol_angle(self, contact_normal):
        '''
        子空间容许角

        parameters
        -----
        contact_normals: ndarray, shape = [3]
            接触法向

        return
        -----
        tol_angle: tuple[float]
            tolerance angle: (lower, upper)
        '''
        # Convert the problem into simplified parameters
        contact_normal = contact_normal / np.linalg.norm(contact_normal)
        angle_with_h = np.arccos(contact_normal[2]) # angle with horizontal plane [-pi/2, pi/2]
        A = np.tan(angle_with_h) # slope of the contact normal
        f_angle = self.friction_angle # friction angle

        def calc_inter_point(f_angle, A, z):
            def inner_function(X):
                nonlocal f_angle, A, z
                F = np.zeros(2)
                x, y = X[0], X[1]
                
                F[0] = (np.sqrt(3) * y * np.sin(f_angle) + x * np.sqrt(4 - 3 * np.sin(f_angle)**2) * np.sqrt(A**2 + 1) + np.sqrt(3) * A * z * np.sin(f_angle)) / (2 * np.sqrt(A**2 + 1))
                F[1] = x**2 + y**2 + z**2 - 1
                
                return F
            return inner_function
    
        z = 1/(np.tan(f_angle)**2 + 1)**(1/2)

        initial_guess = [-1, 1]
        solution_upper = root(calc_inter_point(f_angle, A, z), initial_guess)
        initial_guess = [-1, -1]
        solution_lower = root(calc_inter_point(f_angle, A, z), initial_guess)

        v_angle_upper = np.arctan(solution_upper.x[1] / z)
        v_angle_lower = np.arctan(solution_lower.x[1] / z)
        normal_proj_v = angle_with_h

        return v_angle_lower, v_angle_upper, normal_proj_v

    def h_stable(self, contact_normals, force_direction):
        pass

    def v_stable(self, contact_normal):
        v_angle_lower, v_angle_upper, normal_proj_v = self.get_v_sub_space_tol_angle(contact_normal)
        if (normal_proj_v + v_angle_lower > 0) or (normal_proj_v + v_angle_upper < 0):
            return False
        else:
            return True

    def __normalize(self, array_ref, array):
        '''
        将array的正态分布参数修改为和array_ref一致
        '''
        mean_ref = np.mean(array_ref)
        std_ref = np.std(array_ref)
        mean = np.mean(array)
        std = np.std(array)
        array = (array - mean) / std * std_ref + mean
        return array

    def evaluate(self, contact_points:np.ndarray, contact_normals:np.ndarray, force_direction:np.ndarray, center, gripper_origin = None):
        '''
        brief
        -----
        评估抓取质量

        parameters
        ----
        contact_points: ndarray, shape = [N, 3] 共面的
            接触点坐标
        contact_normals: ndarray, shape = [N, 3]
            接触法向
        force_direction: ndarray, shape = [N, 3]
            夹持方向
        center: ndarray, shape = [3]
            参考原点
        '''
        def get_bound_vec(base, angle, axis):
            '''
            brief
            -----
            获取边界向量

            parameters
            -----
            base: ndarray, shape = [..., 3]
                基向量
            angle: float
                角度
            axis: ndarray, shape = [3]
                旋转轴
            '''
            axis = axis / np.linalg.norm(axis)
            rot_mat = Posture(rvec=angle * axis).rmat
            return rot_mat.dot(base.T).T
        
        ### 转化坐标系为：XY平面平行于contact_points构成的平面，center为原点
        contact_plane_normal = np.cross(contact_points[1] - contact_points[0], contact_points[2] - contact_points[0])
        contact_plane_normal = contact_plane_normal / np.linalg.norm(contact_plane_normal)
        if gripper_origin is not None:
            # 根据gripper原点判定方向
            if np.dot(contact_plane_normal, gripper_origin - contact_points[0]) < 0:
                contact_plane_normal = -contact_plane_normal
        TransPosture = Posture.from_vecs(contact_plane_normal, np.array([0,0,1])) * Posture(tvec=-center)# (rvec=np.cross(contact_plane_normal, [0,0,1])) * Posture(tvec=-center)

        contact_points = TransPosture * contact_points
        contact_normals = TransPosture.rmat.dot(contact_normals.T).T
        # force_direction = TransPosture.rmat.dot(force_direction.T).T

        ### h sub space: 
        ### free: x, y, z_rot
        h_projected_vector = self.vector_projection_on_plane(contact_normals, np.array([0,0,1]))
        h_projected_vector = h_projected_vector / np.linalg.norm(h_projected_vector, axis=-1, keepdims=True) # [N, 3] normalize
        h_force_lowers = get_bound_vec(h_projected_vector, -self.h_friction_angle, np.array([0,0,1])) * self.max_finger_force
        h_force_uppers = get_bound_vec(h_projected_vector, self.h_friction_angle, np.array([0,0,1])) * self.max_finger_force
        h_wrench_lower = self.calc_wrench(h_force_lowers, contact_points)
        h_wrench_upper = self.calc_wrench(h_force_uppers, contact_points)

        h_samples = self.random_sample_in_wrench_space(*list(zip(h_wrench_lower, h_wrench_upper)), sample_num=10000)
        h_samples[:, :3] = self.__normalize(h_samples[:, 3:], h_samples[:, :3])
        h_samples_sub = h_samples[:, [3, 4, 2]] # x, y, z_rot
        h_score = self.get_surface_min_dist(h_samples_sub)

        ### v sub space:
        ### free: x_rot, y_rot, z
        # 分别计算normal_i在v_i上的投影
        v_normals = np.cross(contact_normals, np.array([0,0,1]))
        v_normals = v_normals / np.linalg.norm(v_normals, axis=-1, keepdims=True) # [N, 3] normalize
        v_bound_lowers_list = []
        v_bound_uppers_list = []
        for i, cn in enumerate(contact_normals):
            v_angle_lower, v_angle_upper, normal_proj_v = self.get_v_sub_space_tol_angle(cn)
            v_force_lowers = get_bound_vec(contact_normals[i], v_angle_lower, v_normals[i])
            v_force_uppers = get_bound_vec(contact_normals[i], v_angle_upper, v_normals[i])
            v_bound_lowers_list.append(v_force_lowers)
            v_bound_uppers_list.append(v_force_uppers)
        v_force_lowers = np.stack(v_bound_lowers_list) * self.max_finger_force
        v_force_uppers = np.stack(v_bound_uppers_list) * self.max_finger_force

        v_wrench_lower = self.calc_wrench(v_force_lowers, contact_points)
        v_wrench_upper = self.calc_wrench(v_force_uppers, contact_points)

        v_samples = self.random_sample_in_wrench_space(*list(zip(v_wrench_lower, v_wrench_upper)), sample_num=10000)
        gravity_wrench = self.calc_wrench(np.array([0,0,-self.gravity]), np.array([0,0,0])) # 重力的旋量
        v_samples += gravity_wrench
        v_samples[:, :3] = self.__normalize(v_samples[:, 3:], v_samples[:, :3])
        v_samples_sub = v_samples[:, [0, 1, 5]] # x_rot, y_rot, z
        v_score = self.get_surface_min_dist(v_samples_sub)

        # 绘制
        # 创建箭头，使他们的方向沿着
        # arrows = []
        # for i in range(3):
        #     hl_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.025, cone_radius=0.05, cylinder_height=0.5, cone_height=0.5)
        #     hu_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.025, cone_radius=0.05, cylinder_height=0.5, cone_height=0.5)
        #     vl_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.025, cone_radius=0.05, cylinder_height=0.5, cone_height=0.5)
        #     vu_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.025, cone_radius=0.05, cylinder_height=0.5, cone_height=0.5)
        #     nm_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.025, cone_radius=0.05, cylinder_height=0.5, cone_height=0.5)
        #     hl_arrow.paint_uniform_color([1, 0, 0])
        #     hu_arrow.paint_uniform_color([1, 0, 0])
        #     vl_arrow.paint_uniform_color([0, 1, 0])
        #     vu_arrow.paint_uniform_color([0, 1, 0])
        #     nm_arrow.paint_uniform_color([0, 0, 1])
        #     hl_arrow.transform(Posture.from_vecs([0,0,1], h_bound_lowers[i],  tvec = contact_points[i]).trans_mat)
        #     hu_arrow.transform(Posture.from_vecs([0,0,1], h_bound_uppers[i],  tvec = contact_points[i]).trans_mat)
        #     vl_arrow.transform(Posture.from_vecs([0,0,1], v_bound_lowers[i],  tvec = contact_points[i]).trans_mat)
        #     vu_arrow.transform(Posture.from_vecs([0,0,1], v_bound_uppers[i],  tvec = contact_points[i]).trans_mat)
        #     nm_arrow.transform(Posture.from_vecs([0,0,1], contact_normals[i], tvec = contact_points[i]).trans_mat)
        #     arrows.extend([hl_arrow, hu_arrow, vl_arrow, vu_arrow, nm_arrow])
        # # 绘制接触点
        # contact_point_0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.05).translate(contact_points[0])
        # contact_point_1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.05).translate(contact_points[1])
        # contact_point_2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.05).translate(contact_points[2])
        # # 绘制frame
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        # mesh = o3d.geometry.TriangleMesh(self._modelinfo.mesh)
        # mesh.scale(0.01, center=np.array([0,0,0]))
        # mesh.transform(TransPosture.trans_mat)

        # o3d.visualization.draw_geometries(arrows + [contact_point_0, contact_point_1, contact_point_2, frame, mesh])

        return h_score, v_score

class GraspCoord():
    def __init__(self, *, 
                    contact_points:Optional[np.ndarray] = None,
                    contact_normals:Optional[np.ndarray] = None,
                    force_direction:Optional[np.ndarray] = None,
                    grasp_center:Optional[np.ndarray] = None,
                    grasp_angle:Optional[float] = None,
                    grasp_posture:Optional[Posture] = None,
                    u:Optional[float] = None,
                    score:Optional[float] = None,
                    copy_from:Optional["GraspCoord"] = None
                 ) -> None:
        def get_with_priority(*args):
            for arg in args:
                if arg is not None:
                    return arg
            return None
        
        self.contact_points         = None
        self.contact_normals        = None
        self.force_direction        = None
        self.grasp_center           = None
        self.grasp_angle            = None
        self.grasp_posture:Posture  = None
        self.u                      = None
        self.score                  = None

        if copy_from is not None:
            self.contact_points     = copy_from.contact_points # [N, 3]
            self.contact_normals    = copy_from.contact_normals # [N, 3]
            self.force_direction    = copy_from.force_direction # [N, 3]
            self.grasp_center       = copy_from.grasp_center # [3]
            self.grasp_angle        = copy_from.grasp_angle # float
            self.grasp_posture:Posture = copy_from.grasp_posture # Posture
            self.u                  = copy_from.u # float
            self.score              = copy_from.score # float
        
        self.contact_points         = get_with_priority(contact_points, self.contact_points)
        self.contact_normals        = get_with_priority(contact_normals, self.contact_normals)
        self.force_direction        = get_with_priority(force_direction, self.force_direction)
        self.grasp_center           = get_with_priority(grasp_center, self.grasp_center)
        self.grasp_angle            = get_with_priority(grasp_angle, self.grasp_angle)
        self.grasp_posture:Posture  = get_with_priority(grasp_posture, self.grasp_posture)
        self.u                      = get_with_priority(u, self.u)
        self.score                  = get_with_priority(score, self.score)

    def transform(self, trans_mat:np.ndarray):
        self.contact_points  = trans_mat[:3,:3].dot(self.contact_points.T).T + trans_mat[:3,3] if self.contact_points is not None else None
        self.contact_normals = trans_mat[:3,:3].dot(self.contact_normals.T).T if self.contact_normals is not None else None
        self.force_direction = trans_mat[:3,:3].dot(self.force_direction.T).T if self.force_direction is not None else None
        self.grasp_center    = trans_mat[:3,:3].dot(self.grasp_center) + trans_mat[:3,3] if self.grasp_center is not None else None
        self.grasp_posture   = Posture(homomat=trans_mat) * self.grasp_posture if self.grasp_posture is not None else None

class CandiCoordCalculator():
    def __init__(self, modelpcd:ObjectPcd, gripper:Gripper, 
                 friction_angle = np.pi/4, voxel_size = -1) -> None:
        '''
        v_friction_angle  垂直方向死锁角

        h_friction_angle  水平方向死锁角

        voxel_size: 体素化尺寸
        '''        
        self.modelinfo = modelpcd
        self.gripper = gripper
        self.friction_angle = friction_angle
        h_ratio = 3**(1/2)/2
        self.h_friction_angle = np.arcsin(h_ratio*np.sin(self.friction_angle))
        self.v_friction_angle = np.arcsin((1 - h_ratio**2) ** (1/2) * np.sin(self.friction_angle))

        if voxel_size == -1:
            voxel_size = int(self.modelinfo.pointcloud_size.min() / 30) + 1
        self.voxel_size = voxel_size

        self.rot_num = 500

        self.save_name = "_candi_grasp_posture"

        # calculate Centroid
        voxelized = Voxelized.from_mesh(self.modelinfo.mesh, voxel_size)
        voxelized_centroid = np.mean(voxelized.entity_indices, axis=0) # [3]
        self.centroid = voxelized.restore_mat[:3,:3].dot(voxelized_centroid) + voxelized.restore_mat[:3,3]
        self.gravity = voxel_size**3 * np.sum(voxelized.entity_cube) * 1.5 * 9.8 * 1e-6 # 重力 N

        self.fc_evaluator = ForceClosureEvaluator(self.friction_angle, self.h_friction_angle, self.v_friction_angle, 25, self.gravity)
        self.fc_evaluator._modelinfo = self.modelinfo

    def evaluate(self, grapscoords:list[GraspCoord]):
        _scale = 100
        scores = []
        for rg in grapscoords:
            contact_center = np.mean(rg.contact_points, axis=0)

            contact_points = rg.contact_points / _scale # 使得力矩和力的数值接近
            
            contact_normals = rg.contact_normals
            contact_normals = contact_normals / np.linalg.norm(contact_normals, axis=-1, keepdims=True)

            force_direction = np.tile(contact_center[np.newaxis, ...], [3, 1]) - rg.contact_points
            force_direction = force_direction / np.linalg.norm(force_direction, axis=-1, keepdims=True)

            gripper_origin = rg.grasp_center / _scale

            h_score, v_score = self.fc_evaluator.evaluate( contact_points,
                                        contact_normals,
                                        force_direction,
                                        self.centroid / _scale,
                                        gripper_origin)
            
            rg.score = min(h_score, v_score)
            # if (h_score < v_score):
            #     print("h_score", rg.score)
            # else:
            #     print("v_score", rg.score)
            scores.append(rg.score)
        return scores

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
            
            local_grasp_pose = searcher.restore_as_coord(np.array([0, 0, 0.0]), valid_grapscoords, us)
            vaild_graps_indices_orig = valid_grapscoords.copy()
            vaild_graps_indices_orig[:, 2] = orig_z
            local_grasp_pose_orig = searcher.restore_as_coord(np.array([0, 0, 0.0]), vaild_graps_indices_orig, us)
            
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
        ia = SphereAngle(self.rot_num)
        voxel_size = self.voxel_size
        found_num = 0
        global_grasp_poses = []

        searcher = _CoordSearcher(self.gripper, None, self.voxel_size, self.friction_angle, self.h_friction_angle, self.v_friction_angle) # 搜索器
        progress = tqdm(enumerate(ia.rvec), total=len(ia.rvec), leave=True)
        
        ### 三角面片密集化
        # 获取三角网格的顶点之间的最大距离
        mesh = self.modelinfo.mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        distances = pcd.compute_nearest_neighbor_distance()
        max_distance = np.max(distances)
        # 根据voxel_size计算分割次数，考虑以最小距离构成等边三角形，则必须使最小距离的2/√3小于voxel_size
        number_of_iterations = int(np.ceil(2 * max_distance / np.sqrt(3) / voxel_size))
        mesh = mesh.subdivide_midpoint(number_of_iterations=number_of_iterations)
        mesh.compute_vertex_normals()
        self.modelinfo.mesh = mesh

        ### 逐角度循环搜索
        for baserot_i, rvec in progress:
            ### 旋转
            transform = Posture(rvec=-rvec)
            modelinfo = self.modelinfo.transform(transform)
            ### 体素化
            voxelized:Voxelized = Voxelized.from_mesh(modelinfo.mesh, voxel_size)
            
            ### 逐层查找
            searcher.set_voxelized(voxelized)
            get_v_stable = searcher.get_v_stable() # 夹持力方向始终是水平的，因此可以先判定竖直方向的稳定性
            triangles, contacted = searcher.search_by_layer(get_v_stable)
            triangles, contacted, us = searcher.get_h_stable_grasp(triangles, contacted)
            valid_grapscoords = searcher.calculate_max_depth(triangles, contacted, us)
            restored_grapscoords_Temp = searcher.restore_as_coord(valid_grapscoords) # [rvec(3), tvec(3), u, quality] 还需要返回接触点坐标、法矢
            local_grasp_poses, restored_grapscoords_Object = searcher.restore_to_Object(rvec, restored_grapscoords_Temp)

            # 力封闭并不是说接触力在摩擦锥内是任意的，而是对于任意的外力，总能在摩擦锥内找到抵抗的力
            # 添加抓取质量评估，原点取物体质心。
            scores = self.evaluate(restored_grapscoords_Object)
            local_grasp_poses[:, -1] = scores
            ### 绘制中间过程（可选的）
            if create_process_mesh and len(local_grasp_poses) > 0:
                _create_process_mesh()

            ### 收集结果
            global_grasp_poses.append(local_grasp_poses)
            found_num += len(local_grasp_poses)
            progress.set_postfix({"grasping coordinates found": found_num})

        ### 保存结果
        global_grasp_poses = np.concatenate(global_grasp_poses, 0) # 合并
        save_path = os.path.join(MODELS_DIR, self.modelinfo.name + self.save_name + ".npy")
        np.save(save_path, global_grasp_poses, allow_pickle= True)
        print("saved at {}".format(save_path))
        
        self.modelinfo.candi_coord_parameter = global_grasp_poses

        ### 显示结果（可选的）
        if show:
            self.modelinfo.draw_all(self.gripper)

    def _voxelize_test(self):
        start_time = time.time()

        ia = SphereAngle()
        voxel_size = self.voxel_size

        progress = tqdm(enumerate(ia.rvec), total=len(ia.rvec), leave=True)
        
        ### 三角面片密集化
        # 获取三角网格的顶点之间的最大距离
        mesh = self.modelinfo.mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        distances = pcd.compute_nearest_neighbor_distance()
        max_distance = np.max(distances)
        # 根据voxel_size计算分割次数，考虑以最小距离构成等边三角形，则必须使最小距离的2/√3小于voxel_size
        number_of_iterations = int(np.ceil(2 * max_distance / np.sqrt(3) / voxel_size))
        mesh = mesh.subdivide_midpoint(number_of_iterations=number_of_iterations)
        mesh.compute_vertex_normals()

        self.modelinfo.mesh = mesh

        ### 逐角度循环搜索
        for baserot_i, rvec in progress:
            ### 旋转
            transform = Posture(rvec=-rvec)
            modelinfo = self.modelinfo.transform(transform)
            ### 体素化
            voxelized:Voxelized = Voxelized.from_mesh(modelinfo.mesh, voxel_size)
            # voxelized.visualize()

        print("voxelize time cost: {}".format(time.time() - start_time))