import glob
import json
import os
import trimesh
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

def triangle_area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))

class XYZ():
    def __init__(self, obj_mesh_path="datasets/bop23_challenge/datasets/daoliuzhao/models/models/obj_000001.ply", dataset_path="datasets/bop23_challenge/datasets/daoliuzhao/test"):
        self.img_folders = sorted(glob.glob(dataset_path + "/0000*"))
        self.obj_mesh = trimesh.load_mesh(obj_mesh_path)

    def generate_all_data(self):
        for img_folder in tqdm(self.img_folders, desc="Loading scene"):
            self.generate_data(img_folder)

    def generate_data(self, img_folder):
        scene_gt = json.load(open(os.path.join(img_folder, "scene_gt.json"), 'r')) 
        scene_camera = json.load(open(os.path.join(img_folder, "scene_camera.json"), 'r'))
        mask_folder = os.path.join(img_folder, "mask")
        mask_visib_folder = os.path.join(img_folder, "mask_visib")
        scene_gt_info = {}

        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
        if not os.path.exists(mask_visib_folder):
            os.makedirs(mask_visib_folder)
        
        for img_id, camera_K_depth in tqdm(scene_camera.items(), desc="Processing images"):
            K = np.array(camera_K_depth["cam_K"]).reshape(3,3)
            depth = camera_K_depth["depth_scale"]
            list_R = [np.array(s["cam_R_m2c"]) for s in scene_gt[img_id]]
            list_t = [np.array(s["cam_t_m2c"]).reshape(3,1) for s in scene_gt[img_id]]
            obj_id_list = [s["obj_id"] for s in scene_gt[img_id]]
            depth_map = np.array(Image.open(f"{img_folder}/depth/{int(img_id):06d}.png"))

            scene_gt_info[img_id] = []

            for i, (R, t) in enumerate(zip(list_R, list_t)):
                mask, bbox_obj = self.project_3d_model(self.obj_mesh, K, R, t, depth_map, use_depth=False)
                Image.fromarray(mask).save(os.path.join(mask_folder, f"{int(img_id):06d}_{i:06d}.png"))
                
                mask_visib, bbox_visib = self.project_3d_model(self.obj_mesh, K, R, t, depth_map, use_depth=True)
                Image.fromarray(mask_visib).save(os.path.join(mask_visib_folder, f"{int(img_id):06d}_{i:06d}.png"))

                px_count_all = np.sum(mask > 0)
                px_count_valid = np.sum(mask > 0)  # Assuming all projected pixels are valid
                px_count_visib = np.sum(mask_visib > 0)
                visib_fract = px_count_visib / px_count_all if px_count_all > 0 else 0

                scene_gt_info[img_id].append({
                    "bbox_obj": bbox_obj,
                    "bbox_visib": bbox_visib,
                    "px_count_all": int(px_count_all),
                    "px_count_valid": int(px_count_valid),
                    "px_count_visib": int(px_count_visib),
                    "visib_fract": float(visib_fract)
                })

        # Save scene_gt_info.json
        with open(os.path.join(img_folder, "scene_gt_info.json"), 'w') as f:
            json.dump(scene_gt_info, f, indent=2)

    def project_3d_model(self, model, K, R, t, depth_map, image_shape=(1080, 1440), use_depth=True):
        vertices = np.asarray(model.vertices)
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        camera_points = np.dot(homogeneous_vertices, np.vstack((np.hstack((R, t)), [0, 0, 0, 1])).T)

        projected_points = np.dot(camera_points[:, :3], K.T)
        projected_points /= projected_points[:, 2][:, np.newaxis]
        projected_points = projected_points[:, :2].astype(int)

        rendered_image = np.zeros(image_shape, dtype=np.uint8)
        
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf

        for face in model.faces:
            triangle = projected_points[face]
            z_values = camera_points[face][:, 2]

            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, triangle, 1)

            y_min, y_max = np.min(triangle[:, 1]), np.max(triangle[:, 1])
            x_min, x_max = np.min(triangle[:, 0]), np.max(triangle[:, 0])

            min_x = min(min_x, x_min)
            min_y = min(min_y, y_min)
            max_x = max(max_x, x_max)
            max_y = max(max_y, y_max)

            for y in range(max(0, y_min), min(image_shape[0], y_max + 1)):
                for x in range(max(0, x_min), min(image_shape[1], x_max + 1)):
                    if mask[y, x]:
                        if use_depth:
                            w = triangle_area((x, y), triangle[1], triangle[2])
                            v = triangle_area((x, y), triangle[0], triangle[2])
                            u = triangle_area((x, y), triangle[0], triangle[1])
                            total_area = u + v + w

                            if total_area > 0:
                                u /= total_area
                                v /= total_area
                                w /= total_area

                                z = u * z_values[0] + v * z_values[1] + w * z_values[2]

                                if z < depth_map[y, x]:
                                    rendered_image[y, x] = 255
                        else:
                            rendered_image[y, x] = 255

        bbox = [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]
        return rendered_image, bbox
