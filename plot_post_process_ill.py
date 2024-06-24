import numpy as np
import open3d as o3d
import cv2
import os
import tqdm

from posture_6d.derive import draw_one_mask
from posture_6d.core.posture import Posture
from posture_6d.core.intr import CameraIntr

from posture_6d.data.dataset_example import VocFormat_6dPosture, Spliter, Dataset
from posture_6d.data.mesh_manager import MeshMeta, MeshManager

from post_processer.PostProcesser import PostProcesser, \
    create_pnpsolver, create_mesh_manager, PnPSolver, move_obj_by_optical_link


class _Illu_Creator:
    def __init__(self, cfg_file:str, output_dir:str) -> None:
        self.mesh_manager = create_mesh_manager(cfg_file)
        self.pnpsolver    = create_pnpsolver(cfg_file)

        self.intr:CameraIntr = self.pnpsolver.intr

        self.output_dir = output_dir
    
    def create(self, rgb:np.ndarray, depth:np.ndarray, class_id:int, gt_posture:Posture):
        rgb = rgb[:, :, ::-1]
        mesh_meta = self.mesh_manager.export_meta(class_id)
        self.intr.cam_hgt = rgb.shape[0]
        self.intr.cam_wid = rgb.shape[1]

        pred_posture = Posture(rvec=gt_posture.rvec + np.random.randn(3) * 0.05 - 0.025, tvec=gt_posture.tvec)
        pred_posture = move_obj_by_optical_link(pred_posture, -80)

        mesh_meta = mesh_meta.transform(pred_posture)
        mesh_meta.mesh.compute_vertex_normals()
        # 添加颜色
        mesh_meta.mesh.paint_uniform_color([1.0, 0.0, 0.0])

        ### RGB mode
        rgb_pcd = self.rgb_to_point_cloud(rgb, presumed_depth=gt_posture.tvec[2])
        xy = np.array(rgb_pcd.points)[:, :2]
        x_size = np.max(xy[:, 0]) - np.min(xy[:, 0])
        box_size = x_size / rgb.shape[1]
        # create a triangle mesh
        rgb_mesh = self.create_cube_mesh_from_point_cloud(rgb_pcd, box_size=box_size) # TODO adaptively set box size
        ## create a new empty mesh
        rgb_scene_mesh = o3d.geometry.TriangleMesh()
        # add the mesh of the object
        rgb_scene_mesh += rgb_mesh
        rgb_scene_mesh += mesh_meta.mesh
        rgb_scene_mesh.compute_vertex_normals()

        ### RGBD mode
        rgbd_pcd = self.generate_colored_point_cloud(depth, rgb_image=rgb)
        # create a triangle mesh
        rgbd_mesh = self.create_cube_mesh_from_point_cloud(rgbd_pcd, box_size=box_size)
        # obj mesh
        proj_obj_pcd = self.generate_obj_proj_mesh(mesh_meta, Posture(), color = np.array([1.0, 0, 0.0]))
        proj_obj_mesh = self.create_cube_mesh_from_point_cloud(proj_obj_pcd, box_size=1)
        proj_obj_mesh.paint_uniform_color([1.0, 0.0, 0.0])
        # create a new empty mesh
        rgbd_scene_mesh = o3d.geometry.TriangleMesh()
        # add the mesh of the object
        rgbd_scene_mesh += proj_obj_mesh
        rgbd_scene_mesh += rgbd_mesh
        rgbd_scene_mesh.compute_vertex_normals()

        ### save sperately
        o3d.io.write_triangle_mesh(f"{self.output_dir}/rgb_scene_mesh_{class_id}.ply", rgb_scene_mesh)
        o3d.io.write_triangle_mesh(f"{self.output_dir}/rgbd_scene_mesh_{class_id}.ply", rgbd_scene_mesh)

    @staticmethod
    def create_cube_mesh_from_point_cloud(point_cloud, box_size=0.2, box_color = None):
        """
        Create cube mesh from each point in point cloud.
        
        Parameters:
            point_cloud (o3d.geometry.PointCloud): Open3D point cloud object.
            box_size (float): Size of the cube. Default is 0.2.
        
        Returns:
            o3d.geometry.TriangleMesh: Open3D triangle mesh object.
        """
        vertices = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        box_color = np.array([1.0, 0, 0]) if box_color is None else box_color
        mesh = o3d.geometry.TriangleMesh()
        
        for i, vertex in tqdm.tqdm(enumerate(vertices)):
            # Create a cube mesh centered at the current point
            cube_mesh = o3d.geometry.TriangleMesh.create_box(width=box_size, height=box_size, depth=box_size)
            
            # Translate the cube mesh to the current point
            cube_mesh.translate(vertex)
            
            # Assign color to each vertex of the cube
            try:
                color = colors[i]
            except:
                color = box_color

            cube_mesh.paint_uniform_color(color)
            
            # Merge the cube mesh with the main mesh
            mesh += cube_mesh
        
        mesh.compute_vertex_normals()

        return mesh

    def generate_colored_point_cloud(self, depth_image, rgb_image=None, color=None):
        """
        Generate colored point cloud from depth image and optionally RGB image or single color.
        
        Parameters:
            depth_image (numpy.ndarray): Depth image as numpy array.
            intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix.
            rgb_image (numpy.ndarray): Optional, RGB image as numpy array. Default is None.
            color (numpy.ndarray): Optional, single color as numpy array [R, G, B]. Default is None.
        
        Returns:
            o3d.geometry.PointCloud: Open3D point cloud object.
        """
        intrinsic_matrix = self.intr.intr_M

        # Convert depth image to point cloud
        depth = depth_image.astype(np.float32)
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        points = np.stack([u, v], axis=-1)
        points = points.reshape(-1, 2)
        
        # Apply intrinsic matrix to get 3D points
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = np.dot(np.linalg.inv(intrinsic_matrix), points.T).T

        Z = depth.flatten()
        mask = Z > 0
        points = points * Z[:, None]
        
        # If color is specified as a single color
        if color is not None:
            colors = np.full((points.shape[0], 3), color)
        
        # If RGB image is provided
        elif rgb_image is not None:
            colors = rgb_image.reshape(-1, 3)
        
        # If no color information is provided, set all points to gray
        else:
            colors = np.full((points.shape[0], 3), 128)
        
        # Normalize colors
        colors = colors / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        
        return pcd

    def rgb_to_point_cloud(self, rgb_image, presumed_depth=1.0):
        """
        Convert RGB image to point cloud in camera coordinate system.
        
        Parameters:
            rgb_image (numpy.ndarray): RGB image as numpy array.
            intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix.
        
        Returns:
            numpy.ndarray: Point cloud in camera coordinate system (shape: Nx3).
        """
        intrinsic_matrix = self.intr.intr_M

        height, width, _ = rgb_image.shape
        
        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        
        # Convert pixel coordinates to homogeneous coordinates
        pixel_coords = np.vstack((u, v, np.ones_like(u)))
        
        # Convert pixel coordinates to normalized image coordinates
        normalized_coords = np.dot(np.linalg.inv(intrinsic_matrix), pixel_coords)
        normalized_coords /= normalized_coords[2]
        
        # Extract normalized coordinates (X, Y) and depth (Z) from RGB image
        X = normalized_coords[0]
        Y = normalized_coords[1]
        Z = np.ones(X.shape)

        points = np.stack((X, Y, Z), axis=-1) * presumed_depth
        
        # Combine X, Y, Z coordinates into point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb_image.reshape(-1, 3) / 255.0)
        
        return point_cloud

    def generate_obj_proj_mesh(self, mesh_meta:MeshMeta, obj_posture:Posture, color = None):
        intr = self.intr
        raw_depth, orig_proj = draw_one_mask(mesh_meta, obj_posture, intr, tri_mode=False)
        raw_depth[raw_depth == intr.max_depth] = 0
        raw_depth = raw_depth.astype(np.float32)

        pcd = self.generate_colored_point_cloud(raw_depth, color=color)
        return pcd
            

if __name__ == "__main__":
    cfg_file = "cfg/oldt_morrison_real_voc.yaml"
    output_dir = "post_process_illu"
    os.makedirs(output_dir, exist_ok=True)

    illu = _Illu_Creator(cfg_file, output_dir)

    dataset = VocFormat_6dPosture("datasets/morrison_real_voc", lazy=True)
    data_idx = 55
    class_id = 0

    color = dataset.images_elements.read(data_idx)
    depth_scale = dataset.depth_scale_elements.read(data_idx)
    depth = dataset.depth_elements.read(data_idx)   * depth_scale
    extr  = dataset.extr_vecs_elements.read(data_idx)

    illu.create(color, depth, class_id, Posture(rvec=extr[class_id][0], tvec=extr[class_id][1]))

