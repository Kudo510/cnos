'''
Generate maks and mask_visib for xyz dataset
'''
import glob
import json
import os
import trimesh
from PIL import Image
import numpy as np
import torch
import cv2


def triangle_area(a, b, c):
    """
    Calculate the area of a triangle given three points.
    
    :param a: First point (x, y)
    :param b: Second point (x, y)
    :param c: Third point (x, y)
    :return: Area of the triangle
    """
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


class XYZ():
    def __init__(self, obj_mesh_path="datasets/bop23_challenge/datasets/daoliuzhao/models/models/obj_000001.ply", dataset_path="datasets/bop23_challenge/datasets/daoliuzhao/test"):
        self.img_folders = sorted(glob.glob(dataset_path + "/0000*"))
        self.obj_mesh = trimesh.load_mesh(obj_mesh_path)


    def generate_all_visib_masks(self):
        for img_folder in self.img_folders:
            self.generate_visib_masks(img_folder)


    def generate_visib_masks(self, img_folder):
        scene_gt = json.load(open(os.path.join(img_folder, "scene_gt.json"), 'r')) 
        scene_camera = json.load(open(os.path.join(img_folder, "scene_camera.json"), 'r'))
        mask_folder = os.path.join(img_folder, "mask")
        mask_visib_folder = os.path.join(img_folder, "mask_visib")

        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
        if not os.path.exists(mask_visib_folder):
            os.makedirs(mask_visib_folder)
        
        for img_id, camera_K_depth in scene_camera.items():
            K = np.array(camera_K_depth["cam_K"]).reshape(3,3)
            depth = camera_K_depth["depth_scale"]
            list_R = [np.array(s["cam_R_m2c"]) for s in scene_gt[img_id]]
            list_t = [np.array(s["cam_t_m2c"]).reshape(3,1) for s in scene_gt[img_id]]
            obj_id_list = [s["obj_id"] for s in scene_gt[img_id]]
            depth_map = np.array(Image.open(f"{img_folder}/depth/{int(img_id):06d}.png"))

            for i, (R, t) in enumerate(zip(list_R, list_t)):
                rendered_mask = self.project_3d_model(self.obj_mesh, K, R, t, depth_map)
                Image.fromarray(rendered_mask).save(os.path.join(mask_visib_folder, f"{int(img_id):06d}_{i:06d}.png"))


    def project_3d_model(self, model, K, R, t, depth_map, image_shape=(1544, 2064)):
        # 1. Transform 3D points from model space to camera space
        vertices = np.asarray(model.vertices)
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        camera_points = np.dot(homogeneous_vertices, np.vstack((np.hstack((R, t)), [0, 0, 0, 1])).T)

        # 2. Project 3D points onto the image plane
        projected_points = np.dot(camera_points[:, :3], K.T)
        projected_points /= projected_points[:, 2][:, np.newaxis]
        projected_points = projected_points[:, :2].astype(int)

        # 3. Create an image to render the projected model
        rendered_image = np.zeros(image_shape, dtype=np.uint8)
        
        # 4. Render the projected model with depth testing
        for face in model.faces:
            triangle = projected_points[face]
            z_values = camera_points[face][:, 2]

            # Create a mask for the triangle
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, triangle, 1)

            # Get the bounding box of the triangle
            y_min, y_max = np.min(triangle[:, 1]), np.max(triangle[:, 1])
            x_min, x_max = np.min(triangle[:, 0]), np.max(triangle[:, 0])

            # Iterate over the bounding box
            for y in range(max(0, y_min), min(image_shape[0], y_max + 1)):
                for x in range(max(0, x_min), min(image_shape[1], x_max + 1)):
                    if mask[y, x]:
                        # Compute barycentric coordinates
                        w = triangle_area((x, y), triangle[1], triangle[2])
                        v = triangle_area((x, y), triangle[0], triangle[2])
                        u = triangle_area((x, y), triangle[0], triangle[1])
                        total_area = u + v + w

                        if total_area > 0:
                            u /= total_area
                            v /= total_area
                            w /= total_area

                            # Interpolate z-value
                            z = u * z_values[0] + v * z_values[1] + w * z_values[2]

                            # Depth testing
                            if z < depth_map[y, x]:
                                rendered_image[y, x] = 255  # White color for visibility
                                depth_map[y, x] = z

        return rendered_image



         
        