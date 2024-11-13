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


def scale_points_to_image(points, target_height=1080, target_width=1440, 
                         source_height=None, source_width=None):
    """
    Scale points to fit target image dimensions.
    If source dimensions aren't provided, they're inferred from points.
    """
    # Remove homogeneous coordinate for scaling
    points_xy = points[:, :2]
    
    if source_height is None or source_width is None:
        # Infer source dimensions from points
        min_x, min_y = points_xy.min(axis=0)
        max_x, max_y = points_xy.max(axis=0)
        source_width = max_x - min_x
        source_height = max_y - min_y
    
    # Calculate scale factors
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    
    # Scale points
    scaled_points = points.copy()
    scaled_points[:, 0] *= scale_x
    scaled_points[:, 1] *= scale_y
    
    return scaled_points


class XYZ():
    def __init__(self, obj_mesh_path="datasets/bop23_challenge/datasets/daoliuzhao/models/models/obj_000001.ply", dataset_path="datasets/bop23_challenge/datasets/daoliuzhao/test", image_shape=(1080, 1440)):
        self.img_folders = sorted(glob.glob(dataset_path + "/0000*"))
        self.obj_mesh = trimesh.load_mesh(obj_mesh_path)
        self.image_shape = image_shape

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
                # self.project_3d_model_no_depth_images(self.obj_mesh, K, R, t)
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

    def project_3d_model(self, model, K, R, t, depth_map, use_depth=True):

        image_shape=self.image_shape
        vertices = np.asarray(model.vertices)*1000
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        camera_points = np.dot(homogeneous_vertices, np.vstack((np.hstack((R, t)), [0, 0, 0, 1])).T)

        projected_points = np.dot(camera_points[:, :3], K.T)
        projected_points /= projected_points[:, 2][:, np.newaxis]
        projected_points = (projected_points[:, :2]).astype(int)

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
    
    def project_3d_model_no_depth_images(self, model, K, R, t):
        """
        Project 3D model to 2D image space and create visibility mask without using depth images.
        
        Args:
            model: 3D model with vertices and faces
            K: Camera intrinsic matrix (3x3)
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            image_shape: Output image dimensions (height, width)
        
        Returns:
            rendered_image: Binary mask showing visible parts
            bbox: Bounding box of the rendered object [x, y, w, h]
        """
        image_shape=self.image_shape

        # Project vertices to camera space
        vertices = np.asarray(model.vertices)
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        camera_points = np.dot(homogeneous_vertices, np.vstack((np.hstack((R, t)), [0, 0, 0, 1])).T)
        
        # Project to image space
        projected_points = np.dot(camera_points[:, :3], K.T)
        projected_points /= projected_points[:, 2][:, np.newaxis]
        projected_points = projected_points[:, :2].astype(int)
        
        # Initialize output image and z-buffer
        rendered_image = np.zeros(image_shape, dtype=np.uint8)
        z_buffer = np.full(image_shape, np.inf)  # Initialize z-buffer with infinity
        
        # Track bounding box
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf
        
        # Process each face
        for face in model.faces:
            triangle = projected_points[face]
            z_values = camera_points[face][:, 2]
            
            # Skip triangles with negative z values (behind camera)
            if np.any(z_values <= 0):
                continue
            
            # Create mask for current triangle
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, triangle, 1)
            
            # Get triangle bounds
            y_min, y_max = np.min(triangle[:, 1]), np.max(triangle[:, 1])
            x_min, x_max = np.min(triangle[:, 0]), np.max(triangle[:, 0])
            
            # Update bounding box
            min_x = min(min_x, x_min)
            min_y = min(min_y, y_min)
            max_x = max(max_x, x_max)
            max_y = max(max_y, y_max)
            
            # Clip bounds to image dimensions
            y_min = max(0, y_min)
            y_max = min(image_shape[0], y_max + 1)
            x_min = max(0, x_min)
            x_max = min(image_shape[1], x_max + 1)
            
            # Process pixels within triangle bounds
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if mask[y, x]:
                        # Calculate barycentric coordinates
                        w = triangle_area((x, y), triangle[1], triangle[2])
                        v = triangle_area((x, y), triangle[0], triangle[2])
                        u = triangle_area((x, y), triangle[0], triangle[1])
                        total_area = u + v + w
                        
                        if total_area > 0:
                            # Normalize barycentric coordinates
                            u /= total_area
                            v /= total_area
                            w /= total_area
                            
                            # Interpolate z value
                            z = u * z_values[0] + v * z_values[1] + w * z_values[2]
                            
                            # Update pixel if it's closer than current z-buffer value
                            if z < z_buffer[y, x]:
                                z_buffer[y, x] = z
                                rendered_image[y, x] = 255
        
        # Calculate bounding box
        if min_x == np.inf:  # No visible pixels
            bbox = [0, 0, 0, 0]
        else:
            bbox = [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]
        
        return rendered_image, bbox
    
    def calculate_total_area(self, model, K, R, t):
        """Calculate total visible area if nothing was occluded"""
        vertices = np.asarray(model.vertices)
        homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        camera_points = np.dot(homogeneous_vertices, np.vstack((np.hstack((R, t)), [0, 0, 0, 1])).T)
        
        # Project to image space
        projected_points = np.dot(camera_points[:, :3], K.T)
        projected_points /= projected_points[:, 2][:, np.newaxis]
        projected_points = projected_points[:, :2].astype(int)
        
        total_area = 0
        for face in model.faces:
            triangle = projected_points[face]
            z_values = camera_points[face][:, 2]
            
            # Only count faces facing the camera
            if np.all(z_values > 0):  # all vertices in front of camera
                # Calculate triangle area in 2D
                area = cv2.contourArea(triangle)
                total_area += area
                
        return total_area

    def count_highly_visible_instances(self, models, poses, K, image_shape=(1080, 1440), visibility_threshold=0.9):
        """
        Count instances with visibility ratio > threshold
        
        Args:
            models: List of 3D models
            poses: List of poses, each containing R (3x3) and t (3x1)
            K: Camera intrinsic matrix (3x3)
            image_shape: Output image dimensions (height, width)
            visibility_threshold: Minimum visibility ratio (default 0.9 = 90%)
        
        Returns:
            count: Number of highly visible instances
            visibility_ratios: List of visibility ratios for all instances
        """
        z_buffer = np.full(image_shape, np.inf)
        instance_masks = np.zeros((len(models), *image_shape), dtype=np.uint8)
        visibility_ratios = []
        
        # First pass: Calculate total possible visible area for each instance
        total_areas = []
        for model, pose in zip(models, poses):
            R, t = pose
            total_area = self.calculate_total_area(model, K, R, t)
            total_areas.append(total_area)
        
        # Second pass: Render with occlusion handling
        for idx, (model, pose) in enumerate(zip(models, poses)):
            R, t = pose
            vertices = np.asarray(model.vertices)
            homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
            camera_points = np.dot(homogeneous_vertices, np.vstack((np.hstack((R, t)), [0, 0, 0, 1])).T)
            
            projected_points = np.dot(camera_points[:, :3], K.T)
            projected_points /= projected_points[:, 2][:, np.newaxis]
            projected_points = projected_points[:, :2].astype(int)
            
            visible_area = 0
            
            for face in model.faces:
                triangle = projected_points[face]
                z_values = camera_points[face][:, 2]
                
                if np.any(z_values <= 0):
                    continue
                    
                mask = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, triangle, 1)
                
                y_coords, x_coords = np.where(mask > 0)
                
                for y, x in zip(y_coords, x_coords):
                    w = triangle_area((x, y), triangle[1], triangle[2])
                    v = triangle_area((x, y), triangle[0], triangle[2])
                    u = triangle_area((x, y), triangle[0], triangle[1])
                    total_area = u + v + w
                    
                    if total_area > 0:
                        u /= total_area
                        v /= total_area
                        w /= total_area
                        z = u * z_values[0] + v * z_values[1] + w * z_values[2]
                        
                        if z < z_buffer[y, x]:
                            z_buffer[y, x] = z
                            instance_masks[idx, y, x] = 255
                            visible_area += 1
            
            # Calculate visibility ratio
            if total_areas[idx] > 0:
                visibility_ratio = visible_area / total_areas[idx]
            else:
                visibility_ratio = 0
                
            visibility_ratios.append(visibility_ratio)
        
        # Count instances above threshold
        highly_visible_count = sum(1 for ratio in visibility_ratios if ratio > visibility_threshold)
        
        return highly_visible_count, visibility_ratios

    

def main():
    xyz_dataset = XYZ(obj_mesh_path, dataset_path, image_shape)
    xyz_dataset.generate_all_data()


if __name__ == "__main__":
    image_shape = (1080, 1440)
    obj_mesh_path = "datasets/bop23_challenge/datasets/yuanguan/models/obj_000009.ply"
    dataset_path = "datasets/bop23_challenge/datasets/yuanguan/scene000001"
    main()
