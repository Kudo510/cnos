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
from torchvision.ops.boxes import batched_nms, box_area
import cv2

from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam
from src.model.constrastive_learning_utils import (extract_object_by_mask, calculate_iou, 
_remove_very_small_detections, extract_object_by_mask, _tighten_bboxes,    
_is_mask1_inside_mask2, extract_positive_pairs, extract_positive_pairs_xyz,
 extract_negative_pairs_3, extract_dataset_info_nce)


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)

random.seed(10)


def extract_dataset(dataset="icbin",data_type="test", scene_id=1):  # data_type test or train 
    '''
    Extract positive and negative proposals from sam on all frame in test folder in dataset icbin - scene id is 1 here for the first object
    Use IoU to choose the positive proposals (the most similar masks as the gt)
    '''
    model_type = "vit_h"
    checkpoint_dir =  "datasets/bop23_challenge/pretrained/segment-anything"
    log.info("loading sam")
    sam_model = load_sam(model_type, checkpoint_dir)
    custom_sam_model = CustomSamAutomaticMaskGenerator(sam=sam_model)
    custom_sam_model.predictor.model.to("cuda")

    frame_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/*.png" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
    frame_paths = sorted(glob.glob(frame_paths)) # only 50 not 55 paths - some ids are missing s.t 10

    scene_gt_json = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/scene_gt.json"
    scene_gt = json.load(open(scene_gt_json, 'r'))
        
    all_pos_proposals = []
    all_neg_proposals = []
    for frame_path in frame_paths:
        rgb = Image.open(frame_path).convert("RGB") # rotate(180)
        detections = custom_sam_model.generate_masks(np.array(rgb)) # Include masks and bboxes
        
        masked_images = []
        for mask in detections["masks"].cpu():
            binary_mask = np.array(mask) * 255
            binary_mask = binary_mask.astype(np.uint8)
            masked_image = extract_object_by_mask(rgb, binary_mask)
            masked_images.append(masked_image)

        frame_id = frame_path.split("/")[-1].split(".")[0]
        visib_mask_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/mask_visib/{frame_id}_*.png" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
        mask_paths = sorted(glob.glob(visib_mask_paths))

        poses = list()
        for mask_path in mask_paths:
            mask_scene_id_ = int(mask_path.split('/')[-1].split('.')[0].split('_')[0])
            mask_frame_id = int(mask_path.split('/')[-1].split('.')[0].split('_')[1])

            # Extracting rotation (R) and translation (t)
            R = np.array(scene_gt[str(mask_scene_id_)][mask_frame_id]["cam_R_m2c"]).reshape(3,3)
            t = np.array(scene_gt[str(mask_scene_id_)][mask_frame_id]["cam_t_m2c"])

            # Construct the 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T)

        masks_gt = {
            "masks" : [(np.array(Image.open(mask_path).convert("L"))>0).astype(int) for mask_path in mask_paths],
            "poses" : poses
        }
        masks_pred = {
            "masks" : [np.array(mask.cpu()).astype(int) for mask in detections["masks"]],
            "rgb" : [rgb for rgb in masked_images]
        }

        best_mask_indices = []
        pos_proposals = []
        for i_gt, mask_gt in enumerate(masks_gt["masks"]):

            best_iou = 0
            best_mask_index = -1

            for i, mask_pred in enumerate(masks_pred["masks"]):
                iou = calculate_iou(mask_gt, mask_pred)
                if iou > best_iou:
                    best_iou = iou
                    best_mask_index = i

            if best_iou >0.5:
                best_mask_indices.append(best_mask_index)
                # pos_proposal rgb from prediction and pose from gt
                pos_proposal = {
                    "rgb": np.array(masks_pred["rgb"][best_mask_index])/255.0,
                    "pose": masks_gt["poses"][i_gt]
                }
                pos_proposals.append(pos_proposal)
                
            log.info(f"For frame {frame_path.split('/')[-1]}, the best for {i_gt}th mask is at index {best_mask_index} with an IoU of {best_iou}")      
        
        del detections
    
        all_pos_proposals.append(pos_proposals)
        all_neg_proposals.append([np.array(masked_images[j])/255.0 for j in range(len(masked_images)) if j not in best_mask_indices])
        log.info(f"Number of prediction masks: {len(masks_pred['masks'])}, positive proposals: {len(pos_proposals)}, negative proposals: {len(all_neg_proposals[-1])}")

    return all_pos_proposals, all_neg_proposals

def extract_object_by_mask(image, mask, width: int = 512):
    mask = Image.fromarray(mask)
    masked_image = Image.composite(
        image, Image.new("RGB", image.size, (0, 0, 0)), mask)
    cropped_image = masked_image.crop(masked_image.getbbox())
    # new_height = width * cropped_image.height // cropped_image.width
    
    return cropped_image, masked_image.getbbox()


def extract_dataset_train_pbr(dataset="icbin",data_type="test", scene_id=1, target_obj_id=1):  # data_type test or train 
    '''
    For train_pbr folder not test as in extract_dataset
    This is the one we choose
    so we extract the first scene nly for the targetobject 1 only
    '''
    model_type = "vit_h"
    checkpoint_dir =  "datasets/bop23_challenge/pretrained/segment-anything"
    log.info("loading sam")
    sam_model = load_sam(model_type, checkpoint_dir)
    custom_sam_model = CustomSamAutomaticMaskGenerator(sam=sam_model)
    custom_sam_model.predictor.model.to("cuda")

    frame_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/*.jpg" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
    frame_paths = sorted(glob.glob(frame_paths)) # only 50 not 55 paths - some ids are missing s.t 10

    scene_gt_json = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/scene_gt.json"
    scene_gt = json.load(open(scene_gt_json, 'r'))
    
    obj_dicts = []
    # for frame_id in range(len(scene_gt)):
    for frame_id in range(len(scene_gt)): ## Only for the first 200 scenes - not all of it - too much
        for i, obj in enumerate(scene_gt[str(frame_id)]):
            obj_id = obj["obj_id"] # real object id in the frame

            R = np.array(obj["cam_R_m2c"]).reshape(3,3)
            t = np.array(obj["cam_t_m2c"])
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t

            mask_visib_id = f"{frame_id:06d}_{i:06d}"
            mask_visib_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/mask_visib/{mask_visib_id}.png"

            frame_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/{frame_id:06d}.jpg"
            obj_dict = {
                "obj_id" : obj_id,
                "scene_id" : f"{scene_id:06d}", 
                "mask_visib_path" : mask_visib_path,
                "frame_path" : frame_path,
                "pose" : pose
            }
            obj_dicts.append(obj_dict)

    def find_visib_mask_path(data_list, frame_path, target_obj_id=1):
        return [item for item in data_list if item['obj_id'] == 
                     target_obj_id and item['frame_path'] == frame_path]

    all_pos_proposals = []
    all_neg_proposals = []
    for frame_path in tqdm(frame_paths[0:200]): # only take 200 out of 1000 frames
        rgb = Image.open(frame_path).convert("RGB") # rotate(180)
        detections = custom_sam_model.generate_masks(np.array(rgb)) # Include masks and bboxes
        keep_ids = _remove_very_small_detections(detections["masks"], detections["boxes"])

        selected_masks = [detections["masks"][i].cpu() for i in keep_ids]
        log.info(f"Keeping only {len(selected_masks)} from {detections['masks'].shape[0]} masks")
        masked_images = []
        for mask in selected_masks:
            binary_mask = np.array(mask) * 255
            binary_mask = binary_mask.astype(np.uint8)
            masked_image = extract_object_by_mask(rgb, binary_mask)
            masked_images.append(masked_image[0])
        # noise_remove_sam_detections = _tighten_bboxes(sam_detections)

        # keep_ids = _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"])

        # selected_masks = [noise_remove_sam_detections["masks"][i] for i in range(len(keep_ids)) if keep_ids[i]]
        # selected_bboxes = [noise_remove_sam_detections["boxes"][i] for i in range(len(keep_ids)) if keep_ids[i]]

        # selected_sam_detections = {
        #         "masks" : torch.stack(selected_masks),
        #         "boxes" : torch.stack(selected_bboxes)
        # }
        # log.info(f"Keeping only {len(selected_masks)} from {selected_sam_detections['masks'].shape[0]} masks")

        # masked_images = []
        # for mask in selected_sam_detections["masks"].cpu():
        #     binary_mask = np.array(mask) * 255
        #     binary_mask = binary_mask.astype(np.uint8)
        #     masked_image = extract_object_by_mask(rgb, binary_mask)
        #     masked_images.append(masked_image)
        # noise_remove_sam_detections = _tighten_bboxes(detections)
        # keep_ids = _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"])

        # selected_masks = [noise_remove_sam_detections["masks"][i].cpu() for i in keep_ids]
        # log.info(f"Keeping only {len(selected_masks)} from {noise_remove_sam_detections['masks'].shape[0]} masks")
        # masked_images = []
        # for mask in selected_masks:
        #     binary_mask = np.array(mask) * 255
        #     binary_mask = binary_mask.astype(np.uint8)
        #     masked_image = extract_object_by_mask(rgb, binary_mask)
        #     masked_images.append(masked_image)
        # Find visib_mask path based on obj_dicts
        # obj_id = 1
        selected_obj_list = find_visib_mask_path(obj_dicts, frame_path, target_obj_id) # the contains the mask of object id 1 and frame_path
        # mask_paths = sorted(glob.glob(visib_mask_paths))

        masks_pred = {
            "masks" : [np.array(mask.cpu()).astype(int) for mask in selected_masks],
            "rgb" : [rgb for rgb in masked_images]
        }

        best_mask_indices = []
        pos_proposals = []
        pred_is_inside_indices = []
        too_small_indices = []
        for selected_obj in selected_obj_list:
            for i, mask_pred in enumerate(masks_pred["masks"]):
                mask_gt = (np.array(Image.open(selected_obj["mask_visib_path"]).convert("L"))>0).astype(int)

                pred_diff, pred_is_inside, is_area_too_small = _is_mask1_inside_mask2(mask_pred, mask_gt)
                gt_diff, gt_is_inside, _ = _is_mask1_inside_mask2(mask_gt, mask_pred, noise_threshold=20)
                # log.info(f"Difference between mask {selected_obj['mask_visib_path'].split('/')[-1]} and proposal index {i} is {pred_diff, gt_diff}")
                if (pred_is_inside or gt_is_inside):
                    best_mask_indices.append(i)
                    # pos_proposal rgb from prediction and pose from gt
                    pos_proposal = {
                        "idx" : i,
                        "rgb": np.array(masks_pred["rgb"][i])/255.0,
                        "pose": selected_obj["pose"]
                    }
                    pos_proposals.append(pos_proposal)
                if is_area_too_small:
                    too_small_indices.append(i)
                # if pred_diff <= 2000 and not is_area_too_small:
                if pred_is_inside and not is_area_too_small: # just include also gt inside if u want
                    pred_is_inside_indices.append(i)
                
            # log.info(f"For frame {frame_path.split('/')[-1]}, the best for mask {selected_obj['mask_visib_path'].split('/')[-1]} is at index {best_mask_index} ")      
        
        best_mask_indices = list(set(best_mask_indices))
        pred_is_inside_indices = list(set(pred_is_inside_indices))
        final_pos_proposals = [{
            "idx": i,
            "rgb": next((pos["rgb"] for pos in pos_proposals if pos["idx"] == i), None),
            "pose": next((pos["pose"] for pos in pos_proposals if pos["idx"] == i), None)
        } for i in pred_is_inside_indices] # change to best_mask_indices for threhold 100
       
        del detections
    
        all_pos_proposals.append(final_pos_proposals)
        all_neg_proposals.append([np.array(masked_images[j])/255.0 for j in range(len(masked_images)) if j not in (pred_is_inside_indices + too_small_indices)])
        log.info(f"Number of prediction masks: {len(masks_pred['masks'])}, positive proposals: {len(final_pos_proposals)}, negative proposals: {len(all_neg_proposals[-1])}")

    return all_pos_proposals, all_neg_proposals, best_mask_indices


def extract_dataset_train_pbr_icbin_obj2(dataset="icbin",data_type="train_pbr", scene_id=1, target_obj_id=2):  # data_type test or train 
    '''
    sam as extract_dataset_train_pbr + the positive only contains object 2 # nagative aonly background not the obj1 (need to include neg not in obj_1_pred_is_inside_indices); also ignore all small tiny sam proposals+ all the proposals that contain the gt mask s.t 2 2 boxes shouldnot be in postive and also negative
    '''
    model_type = "vit_h"
    checkpoint_dir =  "datasets/bop23_challenge/pretrained/segment-anything"
    log.info("loading sam")
    sam_model = load_sam(model_type, checkpoint_dir)
    custom_sam_model = CustomSamAutomaticMaskGenerator(sam=sam_model)
    custom_sam_model.predictor.model.to("cuda")

    frame_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/*.jpg" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
    frame_paths = sorted(glob.glob(frame_paths)) # only 50 not 55 paths - some ids are missing s.t 10

    scene_gt_json = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/scene_gt.json"
    scene_gt = json.load(open(scene_gt_json, 'r'))
    
    obj_dicts = []
    # for frame_id in range(len(scene_gt)):
    for frame_id in range(len(scene_gt)): ## Only for the first 200 scenes - not all of it - too much
        for i, obj in enumerate(scene_gt[str(frame_id)]):
            obj_id = obj["obj_id"] # real object id in the frame

            R = np.array(obj["cam_R_m2c"]).reshape(3,3)
            t = np.array(obj["cam_t_m2c"])
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t

            mask_visib_id = f"{frame_id:06d}_{i:06d}"
            mask_visib_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/mask_visib/{mask_visib_id}.png"

            frame_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/{frame_id:06d}.jpg"
            obj_dict = {
                "obj_id" : obj_id,
                "scene_id" : f"{scene_id:06d}", 
                "mask_visib_path" : mask_visib_path,
                "frame_path" : frame_path,
                "pose" : pose
            }
            obj_dicts.append(obj_dict)

    def find_visib_mask_path(data_list, frame_path, target_obj_id):
        return [item for item in data_list if item['obj_id'] == 
                     target_obj_id and item['frame_path'] == frame_path]

    all_pos_proposals = []
    all_neg_proposals = []
    for frame_path in tqdm(frame_paths[0:195]): # only take 200 out of 1000 frames
        rgb = Image.open(frame_path).convert("RGB") # rotate(180)
        detections = custom_sam_model.generate_masks(np.array(rgb)) # Include masks and bboxes
        keep_ids = _remove_very_small_detections(detections["masks"], detections["boxes"])

        selected_masks = [detections["masks"][i].cpu() for i in keep_ids]
        log.info(f"Keeping only {len(selected_masks)} from {detections['masks'].shape[0]} masks")
        masked_images = []
        for mask in selected_masks:
            binary_mask = np.array(mask) * 255
            binary_mask = binary_mask.astype(np.uint8)
            masked_image = extract_object_by_mask(rgb, binary_mask)
            masked_images.append(masked_image[0])

        # Find visib_mask path based on obj_dicts
        # obj_id = 1
        selected_obj_list = find_visib_mask_path(obj_dicts, frame_path, target_obj_id = 2) # the contains the mask of object id 1 and frame_path
        # mask_paths = sorted(glob.glob(visib_mask_paths))
        obj_1_selected_obj_list = find_visib_mask_path(obj_dicts, frame_path, target_obj_id = 1) 

        masks_pred = {
            "masks" : [np.array(mask.cpu()).astype(int) for mask in selected_masks],
            "rgb" : [rgb for rgb in masked_images]
        }

        best_mask_indices = []
        pos_proposals = []
        pred_is_inside_indices = []
        too_small_indices = []
        for selected_obj in selected_obj_list:
            for i, mask_pred in enumerate(masks_pred["masks"]):
                mask_gt = (np.array(Image.open(selected_obj["mask_visib_path"]).convert("L"))>0).astype(int)

                pred_diff, pred_is_inside, is_area_too_small = _is_mask1_inside_mask2(mask_pred, mask_gt)
                gt_diff, gt_is_inside, _ = _is_mask1_inside_mask2(mask_gt, mask_pred, noise_threshold=100)
                # log.info(f"Difference between mask {selected_obj['mask_visib_path'].split('/')[-1]} and proposal index {i} is {pred_diff, gt_diff}")
                if (pred_is_inside or gt_is_inside):
                    best_mask_indices.append(i)
                    # pos_proposal rgb from prediction and pose from gt
                    pos_proposal = {
                        "idx" : i,
                        "rgb": np.array(masks_pred["rgb"][i])/255.0,
                        # "rgb": np.array(masks_pred["rgb"][i]),
                        "pose": selected_obj["pose"]
                    }
                    pos_proposals.append(pos_proposal)
                if is_area_too_small or gt_is_inside:
                    too_small_indices.append(i)
                # if pred_diff <= 2000 and not is_area_too_small:
                if pred_is_inside and not is_area_too_small: # just include also gt inside if u want
                    pred_is_inside_indices.append(i)
            

            # For object 1
        obj_1_pred_is_inside_indices = []
        for obj_1_selected_obj in obj_1_selected_obj_list:
            for i, obj_1_mask_pred in enumerate(masks_pred["masks"]):
                obj_1_mask_gt = (np.array(Image.open(obj_1_selected_obj["mask_visib_path"]).convert("L")) > 0).astype(int)

                obj_1_pred_diff, obj_1_pred_is_inside, obj_1_is_area_too_small = _is_mask1_inside_mask2(obj_1_mask_pred, obj_1_mask_gt, noise_threshold=300)
                obj_1_gt_diff, obj_1_gt_is_inside, _ = _is_mask1_inside_mask2(obj_1_mask_gt, obj_1_mask_pred, noise_threshold=150)
                # log.info(f"Difference between mask {obj_1_selected_obj['mask_visib_path'].split('/')[-1]} and proposal index {i} is {obj_1_pred_diff, obj_1_gt_diff}")
                
                # if pred_diff <= 2000 and not is_area_too_small:
                if obj_1_pred_is_inside or obj_1_gt_is_inside:  # just include also gt inside if you want
                    obj_1_pred_is_inside_indices.append(i)

                
            # log.info(f"For frame {frame_path.split('/')[-1]}, the best for mask {selected_obj['mask_visib_path'].split('/')[-1]} is at index {best_mask_index} ")      
        
        best_mask_indices = list(set(best_mask_indices))
        pred_is_inside_indices = list(set(pred_is_inside_indices))
        final_pos_proposals = [{
            "idx": i,
            "rgb": next((pos["rgb"] for pos in pos_proposals if pos["idx"] == i), None),
            "pose": next((pos["pose"] for pos in pos_proposals if pos["idx"] == i), None)
        } for i in pred_is_inside_indices] # change to best_mask_indices for threhold 100
       
        del detections
    
        all_pos_proposals.append(final_pos_proposals)
        all_neg_proposals.append([np.array(masked_images[j])/255.0 for j in range(len(masked_images)) if j not in (pred_is_inside_indices + too_small_indices+obj_1_pred_is_inside_indices)])
        # all_neg_proposals.append([np.array(masked_images[j]) for j in range(len(masked_images)) if j not in (pred_is_inside_indices + too_small_indices+obj_1_pred_is_inside_indices)])
        log.info(f"Number of prediction masks: {len(masks_pred['masks'])}, positive proposals: {len(final_pos_proposals)}, negative proposals: {len(all_neg_proposals[-1])}")

    return all_pos_proposals, all_neg_proposals, best_mask_indices


def extract_dataset_train_pbr_xyz(dataset="icbin",data_type="test", scene_id=1, target_obj_id=1):  # data_type test or train 
    '''
    For train_pbr folder not test as in extract_dataset
    This is the one we choose
    so we extract the first scene nly for the targetobject 1 only
    '''
    model_type = "vit_h"
    checkpoint_dir =  "datasets/bop23_challenge/pretrained/segment-anything"
    log.info("loading sam")
    sam_model = load_sam(model_type, checkpoint_dir)
    custom_sam_model = CustomSamAutomaticMaskGenerator(sam=sam_model)
    custom_sam_model.predictor.model.to("cuda")

    frame_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/*.jpg" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
    frame_paths = sorted(glob.glob(frame_paths)) # only 50 not 55 paths - some ids are missing s.t 10

    scene_gt_json = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/scene_gt.json"
    scene_gt = json.load(open(scene_gt_json, 'r'))
    
    obj_dicts = []
    # for frame_id in range(len(scene_gt)):
    for frame_id in range(len(scene_gt)): ## Only for the first 200 scenes - not all of it - too much
        for i, obj in enumerate(scene_gt[str(frame_id)]):
            obj_id = obj["obj_id"] # real object id in the frame

            R = np.array(obj["cam_R_m2c"]).reshape(3,3)
            t = np.array(obj["cam_t_m2c"])
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t

            mask_visib_id = f"{frame_id:06d}_{i:06d}"
            mask_visib_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/mask_visib/{mask_visib_id}.png"

            frame_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/{frame_id:06d}.jpg"
            obj_dict = {
                "obj_id" : obj_id,
                "scene_id" : f"{scene_id:06d}", 
                "mask_visib_path" : mask_visib_path,
                "frame_path" : frame_path,
                "pose" : pose
            }
            obj_dicts.append(obj_dict)

    def find_visib_mask_path(data_list, frame_path, target_obj_id=1):
        return [item for item in data_list if item['obj_id'] == 
                     target_obj_id and item['frame_path'] == frame_path]

    all_pos_proposals = []
    all_neg_proposals = []
    for frame_path in tqdm(frame_paths[0:150]): # only take 200 out of 1000 frames
        rgb = Image.open(frame_path).convert("RGB") # rotate(180)
        detections = custom_sam_model.generate_masks(np.array(rgb)) # Include masks and bboxes
        keep_ids = _remove_very_small_detections(detections["masks"], detections["boxes"])

        selected_masks = [detections["masks"][i].cpu() for i in keep_ids]
        log.info(f"Keeping only {len(selected_masks)} from {detections['masks'].shape[0]} masks")
        masked_images = []
        for mask in selected_masks:
            binary_mask = np.array(mask) * 255
            binary_mask = binary_mask.astype(np.uint8)
            masked_image = extract_object_by_mask(rgb, binary_mask)
            masked_images.append(masked_image[0])
        # noise_remove_sam_detections = _tighten_bboxes(sam_detections)

        # keep_ids = _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"])

        # selected_masks = [noise_remove_sam_detections["masks"][i] for i in range(len(keep_ids)) if keep_ids[i]]
        # selected_bboxes = [noise_remove_sam_detections["boxes"][i] for i in range(len(keep_ids)) if keep_ids[i]]

        # selected_sam_detections = {
        #         "masks" : torch.stack(selected_masks),
        #         "boxes" : torch.stack(selected_bboxes)
        # }
        # log.info(f"Keeping only {len(selected_masks)} from {selected_sam_detections['masks'].shape[0]} masks")

        # masked_images = []
        # for mask in selected_sam_detections["masks"].cpu():
        #     binary_mask = np.array(mask) * 255
        #     binary_mask = binary_mask.astype(np.uint8)
        #     masked_image = extract_object_by_mask(rgb, binary_mask)
        #     masked_images.append(masked_image)
        # noise_remove_sam_detections = _tighten_bboxes(detections)
        # keep_ids = _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"])

        # selected_masks = [noise_remove_sam_detections["masks"][i].cpu() for i in keep_ids]
        # log.info(f"Keeping only {len(selected_masks)} from {noise_remove_sam_detections['masks'].shape[0]} masks")
        # masked_images = []
        # for mask in selected_masks:
        #     binary_mask = np.array(mask) * 255
        #     binary_mask = binary_mask.astype(np.uint8)
        #     masked_image = extract_object_by_mask(rgb, binary_mask)
        #     masked_images.append(masked_image)
        # Find visib_mask path based on obj_dicts
        # obj_id = 1
        selected_obj_list = find_visib_mask_path(obj_dicts, frame_path, target_obj_id) # the contains the mask of object id 1 and frame_path
        # mask_paths = sorted(glob.glob(visib_mask_paths))

        masks_pred = {
            "masks" : [np.array(mask.cpu()).astype(int) for mask in selected_masks],
            "rgb" : [rgb for rgb in masked_images]
        }

        best_mask_indices = []
        pos_proposals = []
        pred_is_inside_indices = []
        for selected_obj in selected_obj_list:
            for i, mask_pred in enumerate(masks_pred["masks"]):
                mask_gt = (np.array(Image.open(selected_obj["mask_visib_path"]).convert("L"))>0).astype(int)

                pred_diff, pred_is_inside = _is_mask1_inside_mask2(mask_pred, mask_gt)
                gt_diff, gt_is_inside = _is_mask1_inside_mask2(mask_gt, mask_pred)
                # log.info(f"Difference between mask {selected_obj['mask_visib_path'].split('/')[-1]} and proposal index {i} is {pred_diff, gt_diff}")
                if pred_is_inside or gt_is_inside:
                    best_mask_indices.append(i)
                    # pos_proposal rgb from prediction and pose from gt
                    pos_proposal = {
                        "idx" : i,
                        "rgb": np.array(masks_pred["rgb"][i])/255.0,
                        "pose": selected_obj["pose"]
                    }
                    pos_proposals.append(pos_proposal)

                if pred_diff <= 2000:
                    pred_is_inside_indices.append(i)
                
            # log.info(f"For frame {frame_path.split('/')[-1]}, the best for mask {selected_obj['mask_visib_path'].split('/')[-1]} is at index {best_mask_index} ")      
        
        best_mask_indices = list(set(best_mask_indices))
        pred_is_inside_indices = list(set(pred_is_inside_indices))
        final_pos_proposals = [{
            "idx": i,
            "rgb": next((pos["rgb"] for pos in pos_proposals if pos["idx"] == i), None),
            "pose": next((pos["pose"] for pos in pos_proposals if pos["idx"] == i), None)
        } for i in pred_is_inside_indices] # change to best_mask_indices for threhold 100
       
        del detections
    
        all_pos_proposals.append(final_pos_proposals)
        all_neg_proposals.append([np.array(masked_images[j])/255.0 for j in range(len(masked_images)) if j not in pred_is_inside_indices])
        log.info(f"Number of prediction masks: {len(masks_pred['masks'])}, positive proposals: {len(final_pos_proposals)}, negative proposals: {len(all_neg_proposals[-1])}")

    return all_pos_proposals, all_neg_proposals, best_mask_indices


def extract_dataset_train_pbr_2(dataset="icbin",data_type="test", scene_id=1):  # data_type test or train 
    '''
    For train_pbr folder not test as in extract_dataset
    Only use the crop from gt- not use any sam proposals- just the gt
    positive is the object id 1 and negative is object id 2 - Want to get a proper dataset to test the model to see if it works first 
    '''

    frame_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/*.jpg" 
    frame_paths = sorted(glob.glob(frame_paths)) # only 50 not 55 paths - some ids are missing s.t 10

    scene_gt_json = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/scene_gt.json"
    scene_gt = json.load(open(scene_gt_json, 'r'))

    scene_gt_info_json = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/scene_gt_info.json"
    scene_gt_info = json.load(open(scene_gt_info_json, 'r'))


    obj_dicts = []
    # for frame_id in range(len(scene_gt)):
    for frame_id in range(0, 200): ## Only for the first 200 scenes - not all of it - too much
        assert len(scene_gt[str(frame_id)]) == len(scene_gt_info[str(frame_id)])
        for i, obj in enumerate(scene_gt[str(frame_id)]):
            obj_id = obj["obj_id"] # real object id in the frame

            R = np.array(obj["cam_R_m2c"]).reshape(3,3)
            t = np.array(obj["cam_t_m2c"])
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t

            mask_visib_id = f"{frame_id:06d}_{i:06d}"
            mask_visib_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/mask_visib/{mask_visib_id}.png"

    
            x_min, y_min, w, h = scene_gt_info[str(frame_id)][i]["bbox_visib"]
            # 10* bbox can be -1, -1, -1, -1 , whcih means it doesn't exist
            # Convert to (x_min, y_min, x_max, y_max)
            if x_min < 1 or y_min < 1:
                continue
            bbox_visib = (x_min, y_min, x_min + w, y_min + h)     
            frame_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/{frame_id:06d}.jpg"
            obj_dict = {
                "obj_id" : obj_id,
                "scene_id" : f"{scene_id:06d}", 
                "mask_visib_path" : mask_visib_path,
                "frame_path" : frame_path,
                "bbox_visib": bbox_visib,
                "pose" : pose
            }
            obj_dicts.append(obj_dict)

    all_pos_proposals = []
    all_neg_proposals = []
    
    for obj_dict in obj_dicts:
        img = Image.open(obj_dict["frame_path"])
        mask = Image.open(obj_dict["mask_visib_path"])
        masked_image = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)
        log.info(f"bbox_visib: {obj_dict['bbox_visib']}")
        crop = masked_image.crop(obj_dict["bbox_visib"])

        if obj_dict["obj_id"] == 1:
            pos = {
                "rgb" : np.array(crop.convert("RGB"))/255.0,
                "pose" : obj_dict["pose"]
            }
            all_pos_proposals.append(pos)
        elif obj_dict["obj_id"] == 2:
            neg = {
                "rgb" : np.array(crop.convert("RGB"))/255.0,
                "pose" : obj_dict["pose"]
            }
            all_neg_proposals.append(neg)

    return all_pos_proposals, all_neg_proposals


def extract_dataset_test(dataset="icbin",data_type="test", scene_id=2):  # data_type test or train 
    '''
    For test as in extract_dataset
    Positive crops are the one with 
    use scene id 2 and 3 to get the dataset and test on scene 1
    '''
    model_type = "vit_h"
    checkpoint_dir =  "datasets/bop23_challenge/pretrained/segment-anything"
    log.info("loading sam")
    sam_model = load_sam(model_type, checkpoint_dir)
    custom_sam_model = CustomSamAutomaticMaskGenerator(sam=sam_model)
    custom_sam_model.predictor.model.to("cuda")

    frame_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/*.png" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
    frame_paths = sorted(glob.glob(frame_paths)) # only 50 not 55 paths - some ids are missing s.t 10

    scene_gt_json = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/scene_gt.json"
    scene_gt = json.load(open(scene_gt_json, 'r'))
    
    obj_dicts = []
    # for frame_id in range(len(scene_gt)):
    for frame_id in range(len(scene_gt)): ## Only for the first 200 scenes - not all of it - too much
        for i, obj in enumerate(scene_gt[str(frame_id)]):
            obj_id = obj["obj_id"] # real object id in the frame

            R = np.array(obj["cam_R_m2c"]).reshape(3,3)
            t = np.array(obj["cam_t_m2c"])
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t

            mask_visib_id = f"{frame_id:06d}_{i:06d}"
            mask_visib_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/mask_visib/{mask_visib_id}.png"

            frame_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/{frame_id:06d}.png"
            obj_dict = {
                "obj_id" : obj_id,
                "scene_id" : f"{scene_id:06d}", 
                "mask_visib_path" : mask_visib_path,
                "frame_path" : frame_path,
                "pose" : pose
            }
            obj_dicts.append(obj_dict)

    def find_visib_mask_path(data_list, frame_path, target_obj_id=1):
        return [item for item in data_list if item['obj_id'] == 
                     target_obj_id and item['frame_path'] == frame_path]

    all_pos_proposals = []
    all_neg_proposals = []
    for frame_path in tqdm(frame_paths): # only take 200 out of 1000 frames
        rgb = Image.open(frame_path).convert("RGB") # rotate(180)
        detections = custom_sam_model.generate_masks(np.array(rgb)) # Include masks and bboxes
        keep_ids = _remove_very_small_detections(detections["masks"], detections["boxes"])

        selected_masks = [detections["masks"][i].cpu() for i in keep_ids]
        log.info(f"Keeping only {len(selected_masks)} from {detections['masks'].shape[0]} masks")
        masked_images = []
        for mask in selected_masks:
            binary_mask = np.array(mask) * 255
            binary_mask = binary_mask.astype(np.uint8)
            masked_image = extract_object_by_mask(rgb, binary_mask)
            masked_images.append(masked_image)

        # Find visib_mask path based on obj_dicts
        obj_id = 1
        selected_obj_list = find_visib_mask_path(obj_dicts, frame_path) # the contains the mask of object id 1 and frame_path
        # mask_paths = sorted(glob.glob(visib_mask_paths))

        masks_pred = {
            "masks" : [np.array(mask.cpu()).astype(int) for mask in selected_masks],
            "rgb" : [rgb for rgb in masked_images]
        }

        best_mask_indices = []
        pos_proposals = []
        pred_is_inside_indices = []
        for selected_obj in selected_obj_list:
            for i, mask_pred in enumerate(masks_pred["masks"]):
                mask_gt = (np.array(Image.open(selected_obj["mask_visib_path"]).convert("L"))>0).astype(int)

                pred_diff, pred_is_inside = _is_mask1_inside_mask2(mask_pred, mask_gt)
                gt_diff, gt_is_inside = _is_mask1_inside_mask2(mask_gt, mask_pred)
                # log.info(f"Difference between mask {selected_obj['mask_visib_path'].split('/')[-1]} and proposal index {i} is {pred_diff, gt_diff}")
                if pred_is_inside or gt_is_inside:
                    best_mask_indices.append(i)
                    # pos_proposal rgb from prediction and pose from gt
                    pos_proposal = {
                        "idx" : i,
                        "rgb": np.array(masks_pred["rgb"][i])/255.0,
                        "pose": selected_obj["pose"]
                    }
                    pos_proposals.append(pos_proposal)

                if pred_diff <= 100:
                    pred_is_inside_indices.append(i)
                
            # log.info(f"For frame {frame_path.split('/')[-1]}, the best for mask {selected_obj['mask_visib_path'].split('/')[-1]} is at index {best_mask_index} ")      
        
        best_mask_indices = list(set(best_mask_indices))
        pred_is_inside_indices = list(set(pred_is_inside_indices))
        final_pos_proposals = [{
            "idx": i,
            "rgb": next((pos["rgb"] for pos in pos_proposals if pos["idx"] == i), None),
            "pose": next((pos["pose"] for pos in pos_proposals if pos["idx"] == i), None)
        } for i in pred_is_inside_indices] # change to best_mask_indices for threhold 100
       
        del detections
    
        all_pos_proposals.append(final_pos_proposals)
        all_neg_proposals.append([np.array(masked_images[j])/255.0 for j in range(len(masked_images)) if j not in pred_is_inside_indices])
        log.info(f"Number of prediction masks: {len(masks_pred['masks'])}, positive proposals: {len(final_pos_proposals)}, negative proposals: {len(all_neg_proposals[-1])}")

    return all_pos_proposals, all_neg_proposals, best_mask_indices


# Custom dataset for paired images
class DatasetInfoNCE():
    def __init__(self, dataset, transform, train_transform = True):
        self.dataset = dataset
        self.transform = transform
        self.train_transform = train_transform

    def __getitem__(self, index):
        if self.transform and self.train_transform:
            pos = self.transform(self.dataset[index]["pos"]).permute(1,2,0)
            anchor = self.transform(self.dataset[index]["anchor"]).permute(1,2,0)
            neg = self.transform(self.dataset[index]["neg"]).permute(1,2,0)
        elif self.transform and not self.train_transform:
            pos = self.transform(self.dataset[index]["pos"])
            anchor = self.transform(self.dataset[index]["anchor"])
            neg = self.transform(self.dataset[index]["neg"])
        else:
            pos = self.dataset[index]["pos"]
            anchor = self.dataset[index]["anchor"]
            neg = self.dataset[index]["neg"]

        return pos.float(), anchor.float(), neg.float()

    def __len__(self):
        return len(self.dataset)


class ContrastiveLearningModelInfoNCE(nn.Module):
    def __init__(self, device):
        super().__init__()  # Initialize the nn.Module superclass
        self.device = device
        log.info(f"model used is dinov2_vitl14")
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dinov2_vitl14.patch_size = 14
        if torch.cuda.is_available():
            dinov2_vitl14 = torch.nn.DataParallel(dinov2_vitl14).to(self.device)
        self.dinov2_vitl14 = dinov2_vitl14

        # 2 classes: similar or dissimilar
        # self.fc1 = nn.Linear(2048, 64)
        # self.fc2 = nn.Linear(64, 1)

    def forward_one(self, x):
        x = self.dinov2_vitl14(x)
        # log.info(f"Shape after dinov2 classfication: {x.shape}")
        return x
    
    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)
        return output1, output2, output3

class ContrastiveLearningModel_2(nn.Module):
    '''
    Obtain 30*30*1024 from dinov2 then convert to 1024
    '''
    def __init__(self, device):
        super().__init__()
        self.device = device
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dinov2_vitl14.patch_size = 14
        if torch.cuda.is_available():
            dinov2_vitl14 = torch.nn.DataParallel(dinov2_vitl14).to(self.device)
        self.dinov2_vitl14 = dinov2_vitl14

        self.conv1 = nn.Conv1d(900, 450, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(450, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward_one(self, x):
        layers_list = list(range(24))
        batch_size = x.shape[0]
        x = self.dinov2_vitl14.module.get_intermediate_layers(
                x, n=layers_list, return_class_token=True
                )[23][0]# .reshape(-1,900,1024)
        print(x.shape)
        # x = x.view(batch_size, 900, 1024)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = x.squeeze(1)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07): # lower temperature - more contrast # # Usually between 0.07 and 0.1
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        # Normalize feature vectors
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        # Compute similarity scores
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature

        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)

        # Create labels (positive pair should be selected)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        # Compute cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss


class MultiRotate(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
    

def prepare_dataset(template_paths, template_poses_path, all_pos_proposals, all_neg_proposals):

    # Data loading and preprocessing
    # sigma_range = (0.5, 2.0)
    # rotation_angles = [30, 60, 90, 120, 150]
    # transform = transforms.Compose(
    #     [
    #         # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         # transforms.GaussianBlur(kernel_size=(5, 5), sigma=sigma_range), 
    #         # transforms.ColorJitter(brightness=(1.1, 1.2)),
    #         # transforms.ColorJitter(brightness=(0.7, 0.9)),
    #         # transforms.RandomAutocontrast(p=1)
    #         transforms.RandomHorizontalFlip(),  # Random horizontal flip
    #         transforms.RandomVerticalFlip(),  # Random horizontal flip
    #         # MultiRotate(rotation_angles),  # Custom MultiRotate transformation

    #     ]
    # )

    jitter_strength =1
    color_jitter = transforms.ColorJitter(
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.2 * jitter_strength
    )

    input_height = 224 # shape of input image
    data_transforms  = [
        # transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        # transforms.RandomAutocontrast(p=0.5),
        # transforms.RandomEqualize(p=0.5),
        # transforms.RandomInvert(p=0.3),
        # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        # transforms.RandomSolarize(threshold=128, p=0.5),
        # transforms.RandomPosterize(bits=2, p=0.3),
        # transforms.Lambda(lambda x: x / 255.0),  # Division by 255.0
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomApply([color_jitter], p=0.8),
        GaussianBlur(kernel_size=int(0.1 * input_height)-1, p=0.5), # kernelsize must be odd number
        transforms.ToTensor(),
    ]

    # tensor_transforms = transforms.Compose([
    #     NumpyToTensor(),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
    #     transforms.RandomAutocontrast(p=0.5),
    #     transforms.RandomEqualize(p=0.5),
    #     transforms.RandomInvert(p=0.3),
    #     transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    #     transforms.RandomSolarize(threshold=128, p=0.5),
    #     transforms.RandomPosterize(bits=2, p=0.3),
    # ])

    # train_transform = transforms.Compose([
    #     numpy_transforms,
    #     tensor_transforms
    # ])

    train_transform = transforms.Compose(data_transforms)

    # val_transform = transforms.Compose([
    #     transforms.Lambda(lambda x: x / 255.0),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    template_paths = sorted(glob.glob(template_paths))
    # For train_pbr templates
    with open(template_poses_path, 'rb') as file:
        template_poses = np.array(pickle.load(file))

    # # For pyrender templates
    # template_poses = np.load(template_poses_path)

    templates = {
        "rgb" : [np.array(Image.open(template_path).convert("RGB"))/255.0 for template_path in template_paths],
        "poses" : template_poses
    }


    # postive_pairs = extract_positive_pairs(all_pos_proposals, templates)    
    # negative_pairs = extract_negative_pairs_3(all_neg_proposals, templates)
    dataset = extract_dataset_info_nce(all_pos_proposals, all_neg_proposals, templates)
    train_set, remaining_set = train_test_split(dataset, test_size=0.2, random_state=0)
    val_set, test_set = train_test_split(remaining_set, test_size=0.5, random_state=0)

    train_dataset = DatasetInfoNCE(train_set, transform=train_transform)
    val_dataset = DatasetInfoNCE(val_set, transform=None) # No transform for val set
    test_dataset = DatasetInfoNCE(test_set, transform=None)

    return train_dataset, val_dataset, test_dataset, val_set

def prepare_dataset_xyz(template_paths, template_poses_path, all_pos_proposals, all_neg_proposals):

    # Data loading and preprocessing
    # sigma_range = (0.5, 2.0)
    # rotation_angles = [30, 60, 90, 120, 150]
    # transform = transforms.Compose(
    #     [
    #         # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         # transforms.GaussianBlur(kernel_size=(5, 5), sigma=sigma_range), 
    #         # transforms.ColorJitter(brightness=(1.1, 1.2)),
    #         # transforms.ColorJitter(brightness=(0.7, 0.9)),
    #         # transforms.RandomAutocontrast(p=1)
    #         transforms.RandomHorizontalFlip(),  # Random horizontal flip
    #         transforms.RandomVerticalFlip(),  # Random horizontal flip
    #         # MultiRotate(rotation_angles),  # Custom MultiRotate transformation

    #     ]
    # )

    jitter_strength =1
    color_jitter = transforms.ColorJitter(
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.8 * jitter_strength,
        0.2 * jitter_strength
    )

    input_height = 224 # shape of input image
    data_transforms = [
        transforms.RandomApply([color_jitter], p=0.8),
        GaussianBlur(kernel_size=int(0.1 * input_height)-1, p=0.5), # kernelsize must be odd number
        transforms.ToTensor()
    ]
    train_transform = transforms.Compose(data_transforms)

    template_paths = sorted(glob.glob(template_paths))
    # For train_pbr templates
    with open(template_poses_path, 'rb') as file:
        template_poses = np.array(pickle.load(file))

    # # For pyrender templates
    # template_poses = np.load(template_poses_path)

    templates = {
        "rgb" : [np.array(Image.open(template_path).convert("RGB"))/255.0 for template_path in template_paths],
        "poses" : template_poses
    }

    postive_pairs = extract_positive_pairs_xyz(all_pos_proposals, templates)    
    negative_pairs = extract_negative_pairs_3(all_neg_proposals, templates)

    train_postive_pairs, remaining_postive_pairs = train_test_split(postive_pairs, test_size=0.2, random_state=0)
    val_postive_pairs, test_postive_pairs = train_test_split(remaining_postive_pairs, test_size=0.5, random_state=0)

    train_negative_pairs, remaining_negative_pairs = train_test_split(negative_pairs, test_size=0.2, random_state=0)
    val_negative_pairs, test_negative_pairs = train_test_split(remaining_negative_pairs, test_size=0.5, random_state=0)

    train_dataset = PairedDataset(train_postive_pairs + train_negative_pairs, transform=train_transform)
    val_dataset = PairedDataset(val_postive_pairs + val_negative_pairs, transform=None) # No transform for val set
    test_dataset = PairedDataset(test_postive_pairs + test_negative_pairs, transform=None)

    return train_dataset, val_dataset, test_dataset, train_negative_pairs, train_postive_pairs


def train_info_nce_loss(device, model, train_loader, val_loader, num_epochs, dataset):

    wandb_run = wandb.init(project="cnos_info_nce_loss")
    model = model.to(device)
    criterion = InfoNCELoss() # use a Classification Cross-Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=1)
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)

    # return train_dataset, train_negative_pairs, train_postive_pairs
    # Training loop
    best_val_loss = float('inf')
    for epoch in trange(int(num_epochs), desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            pos, anchor, neg = batch
            pos, anchor, neg = pos.to(device), anchor.to(device), neg.to(device)
            pos_output, anchor_output, neg_output = model(pos, anchor, neg)
            optimizer.zero_grad()
            loss = criterion(pos_output, anchor_output, neg_output)
            train_loss += loss.detach().cpu().item() 
            # print(f"train_loss: {train_loss}")
            loss.backward()
            optimizer.step()
            del pos, anchor, neg, batch
        log.info(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss/len(train_loader):.5f}")
        if epoch %5 == 0:
            torch.save(model.state_dict(), f'contrastive_learning/saved_checkpoints/info_nce_{dataset}_model_checkpoint'+str(epoch)+'.pth')

        if epoch %3 == 0:
            # Validation
            val_loss = 0.0
            model.eval()  #
            with torch.no_grad():
                for batch_val in tqdm(val_loader, desc=f"Epoch {epoch + 1} in validation", leave=False):
                    pos_val, anchor_val, neg_val = batch_val
                    pos_val, anchor_val, neg_val = pos_val.to(device), anchor_val.to(device), neg_val.to(device)
                    pos_output_val, anchor_output_val, neg_output_val = model(pos_val, anchor_val, neg_val)
                    loss = criterion(pos_output_val, anchor_output_val, neg_output_val)
                    val_loss += loss.detach().cpu().item()
                    del pos_output_val, anchor_output_val, neg_output_val, batch_val
            log.info(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss/len(val_loader):.5f}")
            # Check if the validation loss is better than the best validation loss so far
            if val_loss < best_val_loss:
                # Update the best validation loss and save the model state
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                print("best_val_loss: ", best_val_loss/len(val_loader))
                print("saving best model at epoch: ", epoch)
                torch.save(best_model_state, f'contrastive_learning/saved_checkpoints/info_nce_{dataset}_best_model_checkpoint.pth')

        wandb_run.log({
            'train/epoch': epoch + 1,
            'train/loss': train_loss/len(train_loader),
            'eval/loss' : val_loss/len(val_loader),
            'best_val_loss': best_val_loss/len(val_loader)
        })


def train_cosine_loss(device, model, train_loader, val_loader, num_epochs):

    wandb_run = wandb.init(project="cnos_contrastive_learning")
    model = model.to(device)
    criterion = nn.CosineEmbeddingLoss() # use a Classification Cross-Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=1)
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)

    # return train_dataset, train_negative_pairs, train_postive_pairs
    # Training loop
    best_val_loss = float('inf')
    for epoch in trange(int(num_epochs), desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            img1, img2, origin_labels = batch
            labels = torch.where(origin_labels==0, -1 , origin_labels)
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            output1, output2 = model(img1, img2)
            optimizer.zero_grad()
            loss = criterion(output1, output2, labels)
            train_loss += loss.detach().cpu().item() 
            # print(f"train_loss: {train_loss}")
            loss.backward()
            optimizer.step()
            del img1, img2, labels, batch
        log.info(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss/len(train_loader):.5f}")
        if epoch %5 == 0:
            torch.save(model.state_dict(), f'contrastive_learning/saved_checkpoints/model_checkpoint'+str(epoch)+'.pth')

        if epoch %3 == 0:
            # Validation
            val_loss = 0.0
            model.eval()  #
            with torch.no_grad():
                for batch_val in tqdm(val_loader, desc=f"Epoch {epoch + 1} in validation", leave=False):
                    img1_val, img2_val, label_val_original = batch_val
                    label_val = torch.where(label_val_original==0, -1 , label_val_original)
                    img1_val, img2_val, label_val = img1_val.to(device), img2_val.to(device), label_val.to(device)
                    output1_val, output2_val = model(img1_val, img2_val)
                    loss = criterion(output1_val, output2_val, label_val)
                    val_loss += loss.detach().cpu().item()
                    del img1_val, img2_val, label_val, batch_val
            log.info(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {val_loss/len(val_loader):.5f}")
            # Check if the validation loss is better than the best validation loss so far
            if val_loss < best_val_loss:
                # Update the best validation loss and save the model state
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                print("best_val_loss: ", best_val_loss/len(val_loader))
                print("saving best model at epoch: ", epoch)
                torch.save(best_model_state, 'contrastive_learning/saved_checkpoints/best_model_checkpoint.pth')

        wandb_run.log({
            'train/epoch': epoch + 1,
            'train/loss': train_loss/len(train_loader),
            'eval/loss' : val_loss/len(val_loader),
            'best_val_loss': best_val_loss/len(val_loader)
        })

# def test(model, test_loader, device):
    
#     model = model.to(device)
#     criterion = ContrastiveLoss()
#     model.eval()
#     with torch.no_grad():
#         correct, total = 0, 0
#         test_loss = 0.0
#         for batch_test in tqdm(test_loader, desc="Testing"):
#             img1_test, img2_test, label_test = batch_test
#             img1_test, img2_test, label_test = img1_test.to(device), img2_test.to(device), label_test.to(device)
#             output1_test, output2_test = model(img1_test), model(img2_test)
#             loss = criterion(output1_test, output2_test, label_test)
#             test_loss += loss.detach().cpu().item() / len(test_loader)
#             # correct += torch.sum(torch.argmax(y_hat_test, dim=1) == y_test).detach().cpu().item()
#             # total += len(x_test)
#         print(f"Test loss: {test_loss:.2f}")
#         # print(f"Test accuracy: {correct / total * 100:.2f}%")
