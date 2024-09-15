from PIL import Image
import torch
from tqdm import trange
import numpy as np
from src.model.utils import Detections
import glob
import os
from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam
from segment_anything.modeling.sam import Sam
from PIL import Image
import numpy as np
import pickle
from src.model.custom_cnos import cnos_templates_feature_extraction
from src.model.custom_cnos import cnos_crop_feature_extraction
from src.model.custom_cnos import calculate_similarity 
from sklearn.decomposition import PCA

from src.model.custom_cnos import custom_detections, custom_visualize, custom_detections_2, custom_visualize_2
from src.model.cnos_utils import _save_final_results, _bow_retrieval, _move_to_device, _extract_object_by_mask

from torchvision.ops.boxes import batched_nms, box_area
import cv2
import numpy as np


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

    return keep_idxs


def cnos_full_pipeline(rgb_paths, custom_sam_model, syn_template_path_1, template_type, obj_id=1, dataset="icbin"):
    # Load dinov2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    if torch.cuda.is_available():
        dinov2_vitl14 = torch.nn.DataParallel(dinov2_vitl14).to(device)  # Use DataParallel for multiple GPUs

    # Extract features for templates
    syn_data_type = "train_pbr" # test
    out_folder = f"foundpose_analysis/{dataset}/templates"

    # syn_template_path_1 = f"{out_folder}/{syn_data_type}_images_templates/obj_{obj_id:06d}_original" 
    syn_template_files_1 = sorted(glob.glob(os.path.join(syn_template_path_1, "*.png")), key=os.path.getmtime)
    syn_template_files = syn_template_files_1 
    syn_num_templates = len(syn_template_files)
    syn_templates = [np.array(Image.open(template_file).convert("RGB"))[:,:,:3] for template_file in syn_template_files] # This image has 4 channels- the last one is not crucial - maybe about opacity
    syn_ref_features = cnos_templates_feature_extraction(
        templates = syn_templates, num_templates = syn_num_templates, dino_model = dinov2_vitl14, device = device
        )

    for rgb_path in rgb_paths:
        scene_id = int(rgb_path.split("/")[-3])
        frame_id = int(rgb_path.split("/")[-1].split(".")[0])
        rgb = Image.open(rgb_path).convert("RGB")
        sam_detections = custom_sam_model.generate_masks(np.array(rgb))

        noise_remove_sam_detections = _tighten_bboxes(sam_detections)
        keep_ids = _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"]) # torch.arange(len(noise_remove_sam_detections["masks"])) # 
        selected_masks = [noise_remove_sam_detections["masks"][i] for i in range(len(keep_ids)) if keep_ids[i]]
        selected_bboxes = [noise_remove_sam_detections["boxes"][i] for i in range(len(keep_ids)) if keep_ids[i]]

        selected_sam_detections = {
            "masks" : torch.stack(selected_masks),
            "boxes" : torch.stack(selected_bboxes)
        }
        masked_images = []
        for mask in selected_sam_detections["masks"].cpu():
            binary_mask = np.array(mask) * 255
            binary_mask = binary_mask.astype(np.uint8)
            masked_image = _extract_object_by_mask(rgb, binary_mask)
            masked_images.append(masked_image)

        # Cnos Results
        cnos_avg_scores = list()
        cnos_top_5_scores = list()
        # normal crop
        for i in trange(len(masked_images)):
            crop_rgb = np.array(masked_images[i]) # (124, 157, 3)
            crop_feature = cnos_crop_feature_extraction(crop_rgb, dinov2_vitl14, device)

            rounded_avg_score, rounded_scores = calculate_similarity(crop_rgb, crop_feature, syn_ref_features, syn_templates)
            cnos_avg_scores.append(rounded_avg_score)
            cnos_top_5_scores.append(rounded_scores)

        cnos_selected_proposals_indices_05 = [i for i, a_s in enumerate(cnos_avg_scores) if a_s >0.4]
        # cnos_selected_proposals_scores_05 = [a_s for i, a_s in enumerate(cnos_avg_scores) if a_s >0.5]

        template_type
        _save_final_results(selected_proposals_indices=cnos_selected_proposals_indices_05, scene_id=scene_id, frame_id=frame_id, sam_detections=selected_sam_detections, dataset=dataset, rgb_path=rgb_path, type = f"cnos_{template_type}_04")

    return 0