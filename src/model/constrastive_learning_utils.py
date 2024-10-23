from PIL import Image
import numpy as np
import glob
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
import json
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm, trange
import pickle
import wandb
import cv2
from torchvision.ops.boxes import batched_nms, box_area

from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)

random.seed(10)

def compute_inplane(rot_query_openCV, rot_template_openCV):
    delta = rot_template_openCV.dot(rot_query_openCV.T)
    inp = extract_inplane_from_pose(delta)
    # double check to make sure that reconved rotation is correct
    R_inp = convert_inplane_to_rotation(inp)
    recovered_R1 = R_inp.dot(rot_template_openCV)
    err = geodesic_numpy(recovered_R1, rot_query_openCV)
    if err >= 15:
        print("WARINING, error of recovered pose is >=15, err=", err)
    return inp


def opencv2opengl(cam_matrix_world):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if len(cam_matrix_world.shape) == 2:
        return np.matmul(transform, cam_matrix_world)
    else:
        transform = np.tile(transform, (cam_matrix_world.shape[0], 1, 1))
        return np.matmul(transform, cam_matrix_world)


def extract_inplane_from_pose(pose):
    inp = Rotation.from_matrix(pose).as_euler("zyx", degrees=True)[0]
    return inp


def convert_inplane_to_rotation(inplane):
    R_inp = Rotation.from_euler("z", -inplane, degrees=True).as_matrix()
    return R_inp


def geodesic_numpy(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))

# def extract_positive_negative_pairs(all_pos_proposals, all_neg_proposals, templates):
#     pos_pairs = list()
#     for proposals_id in range(len(all_pos_proposals)):
#         obj_query_pose = all_pos_proposals[proposals_id]["pose"][None]
#         obj_template_poses = templates["poses"]

#         return_inplane = True

#         obj_query_openGL_pose = opencv2opengl(obj_query_pose)
#         obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3 # (translation components) -  It assumes that the 3D location is found in the third column of the pose matrices.
#         obj_template_openGL_poses = opencv2opengl(obj_template_poses)
#         obj_template_openGL_locations = obj_template_openGL_poses[:, 2, :3]  # Nx3 # (translation components)

#         # find the nearest template
#         # It computes the pairwise distances between each query pose location and each template pose location using cdist.
#         distances = cdist(obj_query_openGL_location, obj_template_openGL_locations)
#         best_index_in_pose_distribution = np.argmin(distances, axis=-1)  # M
#         if return_inplane:
#             nearest_poses = obj_template_poses[best_index_in_pose_distribution]
#             inplanes = np.zeros(len(obj_query_pose))
#             for idx in range(len(obj_query_pose)):
#                 rot_query_openCV = obj_query_pose[idx, :3, :3]
#                 rot_template_openCV = nearest_poses[idx, :3, :3]
#                 inplanes[idx] = compute_inplane(rot_query_openCV, rot_template_openCV)

#         pos_pair = {
#             "img1" : resize_and_pad_image(np.transpose(templates["rgb"][best_index_in_pose_distribution[0]]/255.0, (2,0,1)), target_max=224), # resize and pad images
#             "img2" : resize_and_pad_image(np.transpose(np.array(all_pos_proposals[proposals_id]["rgb"])/255.0, (2,0,1)), target_max=224), 
#             "label" : 1
#         }
#         pos_pairs.append(pos_pair)

#     return pos_pairs

def _augment_pos_pairs(pos_pairs):
    augmented_pos_pairs = []
    rotation_angles = [45, 90, 135, 180, 225] # [45, 90, 135] # 
    
    for pos_pair in pos_pairs:
        # Add the original pair
        augmented_pos_pairs.append(pos_pair)
        
        # Add rotated versions
        for angle in rotation_angles:
            rotated_pair = {
                "img1": transforms.functional.rotate(pos_pair["img1"], angle),
                "img2": pos_pair["img2"],
                "label": 1
            }
            augmented_pos_pairs.append(rotated_pair)
    
    return augmented_pos_pairs

def _augment_dataset(dataset):
    augmented_dataset = []
    rotation_angles = [45, 90, 135, 180, 225] # [45, 90, 135] # 
    
    for data in dataset:
        # Add the original pair
        augmented_dataset.append(data)
        
        # Add rotated versions
        for angle in rotation_angles:
            rotated_data = {
                "pos": transforms.functional.rotate(data["pos"], angle),
                "anchor": data["anchor"],
                "neg": data["neg"],
                "label": 1
            }
            augmented_dataset.append(rotated_data)
    
    return augmented_dataset

def extract_positive_pairs(all_pos_proposals, templates):
    
    transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    pos_pairs = list()
    for proposals_id in trange(len(all_pos_proposals)):
        if all_pos_proposals[proposals_id]["pose"] is not None:
            obj_query_pose = all_pos_proposals[proposals_id]["pose"][None]
            obj_template_poses = templates["poses"]

            return_inplane = True

            obj_query_openGL_pose = opencv2opengl(obj_query_pose)
            obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3 # (translation components) -  It assumes that the 3D location is found in the third column of the pose matrices.
            obj_template_openGL_poses = opencv2opengl(obj_template_poses)
            obj_template_openGL_locations = obj_template_openGL_poses[:, 2, :3]  # Nx3 # (translation components)

            # find the nearest template
            # It computes the pairwise distances between each query pose location and each template pose location using cdist.
            distances = cdist(obj_query_openGL_location, obj_template_openGL_locations)
            best_index_in_pose_distribution = np.argmin(distances, axis=-1)  # M
            if return_inplane:
                nearest_poses = obj_template_poses[best_index_in_pose_distribution]
                inplanes = np.zeros(len(obj_query_pose))
                for idx in range(len(obj_query_pose)):
                    rot_query_openCV = obj_query_pose[idx, :3, :3]
                    rot_template_openCV = nearest_poses[idx, :3, :3]
                    inplanes[idx] = compute_inplane(rot_query_openCV, rot_template_openCV)
            d, x, y = np.transpose(all_pos_proposals[proposals_id]['rgb'], (2,0,1)).shape
            log.info(f"Size of pos proposal: {d,x,y}")
            if d ==0 or x == 0 or y ==0:
                continue
            pos_pair = {
                "img1" : transform(resize_and_pad_image(np.transpose(all_pos_proposals[proposals_id]["rgb"], (2,0,1)), target_max=224)), 
                "img2" : transform(resize_and_pad_image(np.transpose(templates["rgb"][best_index_in_pose_distribution[0]], (2,0,1)), target_max=224)), # resize and pad images
                # "img1" : resize_and_pad_image(np.transpose(all_pos_proposals[proposals_id]["rgb"], (2,0,1)), target_max=224), 
                # "img2" : resize_and_pad_image(np.transpose(templates["rgb"][best_index_in_pose_distribution[0]], (2,0,1)), target_max=224), # resize and pad images
                "label" : 1
            }
            log.info(f"pos_pair['img1'].shape[-1], pos_pair['img2'].shape[-1]: {pos_pair['img1'].shape[-1]}, {pos_pair['img2'].shape[-1]}" )
            if pos_pair["img1"].shape[-1] != pos_pair["img2"].shape[-1]:
                continue
            pos_pairs.append(pos_pair)

    augmented_pos_pairs = pos_pairs # _augment_pos_pairs(pos_pairs)
    return augmented_pos_pairs


def extract_dataset_info_nce(all_pos_proposals, all_neg_proposals, templates):
    
    transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    selected_neg_proposals = random.sample(all_neg_proposals, len(all_pos_proposals))

    dataset = list()
    for proposals_id in trange(len(all_pos_proposals)):
        if all_pos_proposals[proposals_id]["pose"] is not None:
            obj_query_pose = all_pos_proposals[proposals_id]["pose"][None]
            obj_template_poses = templates["poses"]

            return_inplane = True

            obj_query_openGL_pose = opencv2opengl(obj_query_pose)
            obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3 # (translation components) -  It assumes that the 3D location is found in the third column of the pose matrices.
            obj_template_openGL_poses = opencv2opengl(obj_template_poses)
            obj_template_openGL_locations = obj_template_openGL_poses[:, 2, :3]  # Nx3 # (translation components)

            # find the nearest template
            # It computes the pairwise distances between each query pose location and each template pose location using cdist.
            distances = cdist(obj_query_openGL_location, obj_template_openGL_locations)
            best_index_in_pose_distribution = np.argmin(distances, axis=-1)  # M
            if return_inplane:
                nearest_poses = obj_template_poses[best_index_in_pose_distribution]
                inplanes = np.zeros(len(obj_query_pose))
                for idx in range(len(obj_query_pose)):
                    rot_query_openCV = obj_query_pose[idx, :3, :3]
                    rot_template_openCV = nearest_poses[idx, :3, :3]
                    inplanes[idx] = compute_inplane(rot_query_openCV, rot_template_openCV)
            d, x, y = np.transpose(all_pos_proposals[proposals_id]['rgb'], (2,0,1)).shape
            log.info(f"Size of pos proposal: {d,x,y}")
            if d ==0 or x == 0 or y ==0:
                continue
            data = {
                "pos" : transform(resize_and_pad_image(np.transpose(all_pos_proposals[proposals_id]["rgb"], (2,0,1)), target_max=224)), 
                "anchor" : transform(resize_and_pad_image(np.transpose(templates["rgb"][best_index_in_pose_distribution[0]], (2,0,1)), target_max=224)), # resize and pad images
                "neg" : transform(resize_and_pad_image(np.transpose(selected_neg_proposals[proposals_id], (2,0,1)), target_max=224)), 
            }
            # log.info(f"pos_pair['img1'].shape[-1], pos_pair['img2'].shape[-1]: {pos_pair['img1'].shape[-1]}, {pos_pair['img2'].shape[-1]}" )
            # if pos_pair["img1"].shape[-1] != pos_pair["img2"].shape[-1]:
            #     continue
            dataset.append(data)

    augmented_dataset= _augment_dataset(dataset)
    return augmented_dataset


def extract_positive_pairs_xyz(all_pos_proposals, templates):
    
    transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    pos_pairs = list()
    for proposals_id in trange(len(all_pos_proposals)):
        if all_pos_proposals[proposals_id]["pose"] is not None:
            obj_query_pose = all_pos_proposals[proposals_id]["pose"][None]
            obj_template_poses = templates["poses"]

            return_inplane = True

            obj_query_openGL_pose = opencv2opengl(obj_query_pose)
            obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3 # (translation components) -  It assumes that the 3D location is found in the third column of the pose matrices.
            obj_template_openGL_poses = opencv2opengl(obj_template_poses)
            obj_template_openGL_locations = obj_template_openGL_poses[:, 2, :3]  # Nx3 # (translation components)

            # find the nearest template
            # It computes the pairwise distances between each query pose location and each template pose location using cdist.
            distances = cdist(obj_query_openGL_location, obj_template_openGL_locations)
            best_index_in_pose_distribution = np.argmin(distances, axis=-1)  # M
            if return_inplane:
                nearest_poses = obj_template_poses[best_index_in_pose_distribution]
                inplanes = np.zeros(len(obj_query_pose))
                for idx in range(len(obj_query_pose)):
                    rot_query_openCV = obj_query_pose[idx, :3, :3]
                    rot_template_openCV = nearest_poses[idx, :3, :3]
                    inplanes[idx] = compute_inplane(rot_query_openCV, rot_template_openCV)
            d, x, y = np.transpose(all_pos_proposals[proposals_id]['rgb'], (2,0,1)).shape
            log.info(f"Size of pos proposal: {d,x,y}")
            if d ==0 or x == 0 or y ==0:
                continue
            pos_pair = {
                "img1" : transform(resize_and_pad_image(np.transpose(all_pos_proposals[proposals_id]["rgb"], (2,0,1)), target_max=224)), 
                "img2" : transform(resize_and_pad_image(np.transpose(templates["rgb"][best_index_in_pose_distribution[0]], (2,0,1)), target_max=224)), # resize and pad images
                "label" : 1
            }
            log.info(f"pos_pair['img1'].shape[-1], pos_pair['img2'].shape[-1]: {pos_pair['img1'].shape[-1]}, {pos_pair['img2'].shape[-1]}" )
            if pos_pair["img1"].shape[-1] != pos_pair["img2"].shape[-1]:
                continue
            pos_pairs.append(pos_pair)

    augmented_pos_pairs = [pos_pair for pos_pair in pos_pairs]
    return augmented_pos_pairs


def extract_negative_pairs(all_neg_proposals, all_pos_proposals, templates):
    '''
    only crop from train_pbr not from sam proposals
    '''
    selected_neg_proposals = random.sample(all_neg_proposals, len(all_pos_proposals)) # Num of negative = num of positive
    copied_templates = templates["rgb"].copy()

    neg_pairs = list()
    for neg_prop in selected_neg_proposals:
        selected_temp_index = random.randint(0, len(copied_templates) - 1)
        selected_temp = copied_templates[selected_temp_index]
        # del copied_templates[selected_temp_index]

        d, x, y = np.transpose(neg_prop['rgb'], (2,0,1)).shape
        log.info(f"Size of neg proposal: {d,x,y}")
        if d ==0 or x == 0 or y ==0:
            continue

        neg_pair = {
            "img1" : resize_and_pad_image(np.transpose(neg_prop['rgb'], (2,0,1)), target_max=224), 
            "img2" : resize_and_pad_image(np.transpose(selected_temp, (2,0,1)), target_max=224),
            "label" : 0
        }

        assert neg_pair["img1"].shape[-1] == neg_pair["img2"].shape[-1]
        neg_pairs.append(neg_pair)

    return neg_pairs

def extract_negative_pairs_2(all_neg_proposals, all_pos_proposals, templates):
    '''
    the one used for sam proposals
    '''
    selected_neg_proposals = random.sample(all_neg_proposals, len(all_pos_proposals)) # Num of negative = num of positive
    copied_templates = templates["rgb"].copy()

    neg_pairs = list()
    for neg_prop in tqdm(selected_neg_proposals):
        selected_temp_index = random.randint(0, len(copied_templates) - 1)
        selected_temp = copied_templates[selected_temp_index]
        # del copied_templates[selected_temp_index]

        d, x, y = np.transpose(neg_prop, (2,0,1)).shape
        log.info(f"Size of neg proposal: {d,x,y}")
        if d ==0 or x == 0 or y ==0:
            continue

        neg_pair = {
            "img1" : resize_and_pad_image(np.transpose(neg_prop, (2,0,1)), target_max=224), 
            "img2" : resize_and_pad_image(np.transpose(selected_temp, (2,0,1)), target_max=224),
            "label" : 0
        }

        assert neg_pair["img1"].shape[-1] == neg_pair["img2"].shape[-1]
        neg_pairs.append(neg_pair)

    return neg_pairs


def extract_negative_pairs_3(all_neg_proposals, templates):
    '''
    the one used for sam proposals
    use all negative proposals not jsut equal number of positive propsals- accept the imbalance
    '''
    
    copied_templates = templates["rgb"].copy()
    transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    neg_pairs = list()
    for neg_prop in tqdm(all_neg_proposals):
        selected_temp_index = random.randint(0, len(copied_templates) - 1)
        selected_temp = copied_templates[selected_temp_index]
        # del copied_templates[selected_temp_index]

        d, x, y = np.transpose(neg_prop, (2,0,1)).shape
        log.info(f"Size of neg proposal: {d,x,y}")
        if d ==0 or x == 0 or y ==0:
            continue

        neg_pair = {
            "img1" : transform(resize_and_pad_image(np.transpose(neg_prop, (2,0,1)), target_max=224)), 
            "img2" : transform(resize_and_pad_image(np.transpose(selected_temp, (2,0,1)), target_max=224)),
            # "img1" : resize_and_pad_image(np.transpose(neg_prop, (2,0,1)), target_max=224), 
            # "img2" : resize_and_pad_image(np.transpose(selected_temp, (2,0,1)), target_max=224),
            "label" : 0
        }

        assert neg_pair["img1"].shape[-1] == neg_pair["img2"].shape[-1]
        neg_pairs.append(neg_pair)

    return neg_pairs

def resize_and_pad_image(image, target_max=224):
    '''
    cnos target_max = 224
    foundpose target_max = 420
    '''

    # image = np.transpose(input_image.copy(), (2,0,1))
    # Scale image to 420
    scale_factor = target_max / torch.max(torch.tensor(image.shape)) # 420/max of x1,y1,x2,y2
    scaled_image = F.interpolate(torch.tensor(image).unsqueeze(0), scale_factor=scale_factor.item())[0] # unsqueeze at  0 - B,C, H, W
    
    # Padding 0 to 3, 420, 420
    original_h, original_w = scaled_image.shape[1:]
    original_ratio = original_w / original_h
    target_h, target_w = target_max, target_max
    target_ratio  = target_w/target_h 
    if  target_ratio != original_ratio: 
        padding_top = max((target_h - original_h) // 2, 0)
        padding_bottom = target_h - original_h - padding_top
        padding_left = max((target_w - original_w) // 2, 0)
        padding_right = target_w - original_w - padding_left
        scaled_padded_image = F.pad(
        scaled_image, (padding_left, padding_right, padding_top, padding_bottom)
        )
    else:
        scaled_padded_image = scaled_image
    
    if scaled_padded_image.shape[-1] == 223:
        scaled_padded_image = F.pad(scaled_padded_image, (0, 1, 0, 1), mode='constant', value=0)

    return scaled_padded_image

    
# def preprocess_images(input_images, contrastive_model):
#     '''
#     Return the features of input images
#     '''
#     rgb_normalize = T.Compose(
#         [
#             T.ToTensor(),
#             T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ]
#     )
#     normalized_images = [rgb_normalize(input_image/255.0).float() for img in input_images]
#     resized_images = [resize_and_pad_image(normalized_image, target_max=224) for normalized_image in normalized_images]

#     batch_size = 16
#     layers_list = list(range(24))
#     batches = [resized_images[i:i+batch_size] for i in range(0, len(resized_images), batch_size)]
#     patch_features= list()

#     for batch in batches:
#         batch = torch.stack(batch)
#         size = batch.shape[0]
#         torch.cuda.empty_cache()
#         with torch.no_grad(): 
#             batch_feature = contrastive_model(batch).reshape(size,-1,1024).cpu()
#         patch_features.append(batch_feature.to('cpu'))
#         del batch_feature
#     patch_features = torch.cat(patch_features)
#     del contrastive_model

#     return patch_features


def extract_object_by_mask(image, mask, width: int = 512):
    mask = Image.fromarray(mask)
    masked_image = Image.composite(
        image, Image.new("RGB", image.size, (0, 0, 0)), mask)
    cropped_image = masked_image.crop(masked_image.getbbox())
    # new_height = width * cropped_image.height // cropped_image.width
    return cropped_image


def calculate_iou(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def _is_mask1_inside_mask2(mask1, mask2, noise_threshold=250, area_threshold=0.5):
    # Ensure masks are binary (0 or 1)
    mask1 = (mask1 > 0).astype(int)
    mask2 = (mask2 > 0).astype(int)

    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    
    # Check if mask1 is entirely inside mask2
    intersection = np.bitwise_and(mask1, mask2)
    difference = np.sum(mask1) - np.sum(intersection)
    
    is_inside = difference <= noise_threshold

    is_area_too_small = area1 < (area_threshold * area2)

    return difference, is_inside, is_area_too_small
    # return difference <= noise_threshold


def _tighten_bboxes(sam_detections, device="cuda"):
    # Apply morphological opening - it does eorion then dilation to remove noise
    kernel = np.ones((5,5), np.uint8)

    filtered_masks = list()
    for mask in sam_detections["masks"].cpu():
        filtered_mask = cv2.morphologyEx(np.array(mask, dtype="uint8"), cv2.MORPH_OPEN, kernel)
        filtered_masks.append(torch.tensor(filtered_mask))
    sam_detections["masks"] = torch.stack(filtered_masks).to(device)
    return sam_detections


def _remove_very_small_detections(masks, boxes): # after this step only valid boxes, masks are saved, other are filtered out
    min_box_size = 0.05 # relative to image size 
    min_mask_size = 300/(640*480) # relative to image size assume the pixesl should be in range (300, 10000) need to remove them 
    max_mask_size = 10000/(640*480) 
    img_area = masks.shape[1] * masks.shape[2]
    box_areas = box_area(boxes) / img_area
    formatted_values = [f'{value.item():.6f}' for value in box_areas*img_area]
    mask_areas = masks.sum(dim=(1, 2)) / img_area
    keep_idxs = torch.logical_and(
        torch.logical_and(mask_areas > min_mask_size, mask_areas < max_mask_size),
        box_areas > min_box_size**2
    )
    indices = torch.nonzero(keep_idxs).squeeze().tolist()
    return indices


# def _remove_very_small_detections(masks, boxes): # after this step only valid boxes, masks are saved, other are filtered out
#         min_box_size = 0.05 # relative to image size 
#         min_mask_size = 4e-4 # relative to image size
#         img_area = masks.shape[1] * masks.shape[2]
#         box_areas = box_area(boxes) / img_area
#         mask_areas = masks.sum(dim=(1, 2)) / img_area
#         keep_idxs = torch.logical_and(
#             box_areas > min_box_size**2, mask_areas > min_mask_size
#         )
#         indices = torch.nonzero(keep_idxs).squeeze().tolist()
#         return indices