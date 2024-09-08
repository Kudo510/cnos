from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from torchvision.ops.boxes import batched_nms, box_area
import cv2
import numpy as np
import logging
import glob
import os

from src.model.utils import Detections
from cnos_clustering_utils import (move_to_device, tighten_bboxes, extract_object_by_mask, _remove_very_small_detections, 
                                    plot_images, extract_sam_crops_features, hierarchical_clustering)
from src.model.custom_cnos import cnos_templates_feature_extraction
from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam
from segment_anything.modeling.sam import Sam

# Create a logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)


def main(dataset, rgb_path, obj_id, device): 
    model_type = "vit_h"
    checkpoint_dir =  "datasets/bop23_challenge/pretrained/segment-anything"
    sam_model = load_sam(model_type, checkpoint_dir)
    custom_sam_model = CustomSamAutomaticMaskGenerator(sam=sam_model) #, crop_overlap_ratio = 750 / 1500) # , box_nms_thresh = 0.6 )
    move_to_device(custom_sam_model)

    rgb = Image.open(rgb_path).convert("RGB")
    sam_detections = custom_sam_model.generate_masks(np.array(rgb))

    # Remove noise and filter too small and too big crops
    noise_remove_sam_detections = tighten_bboxes(sam_detections)
    keep_ids = _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"]) # torch.arange(len(noise_remove_sam_detections["masks"])) # 
    selected_masks = [noise_remove_sam_detections["masks"][i] for i in range(len(keep_ids)) if keep_ids[i]]
    selected_bboxes = [noise_remove_sam_detections["boxes"][i] for i in range(len(keep_ids)) if keep_ids[i]]
    # noise_remove_sam_detections = sam_detections
    # keep_ids = torch.arange(0, len(noise_remove_sam_detections["masks"]))#  _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"]) # torch.arange(len(noise_remove_sam_detections["masks"])) # 
    # selected_masks = [noise_remove_sam_detections["masks"][i] for i in range(len(keep_ids))]
    # selected_bboxes = [noise_remove_sam_detections["boxes"][i] for i in range(len(keep_ids))]
    selected_sam_detections = {
        "masks" : torch.stack(selected_masks),
        "boxes" : torch.stack(selected_bboxes)
    }
    log.info(len(selected_sam_detections["masks"]))
    log.info(len(sam_detections["masks"]))

    masked_images = []
    for mask in selected_sam_detections["masks"].cpu():
        binary_mask = np.array(mask) * 255
        binary_mask = binary_mask.astype(np.uint8)
        masked_image = extract_object_by_mask(rgb, binary_mask)
        masked_images.append(masked_image)
        
    # # Plot sam crops
    # rows = ceil(len(masked_images) / 6)
    # cols = 6
    # plot_images(masked_images, rows, cols) 

    # Load synthetic train_pbr templates 
    out_folder = f"foundpose_analysis/{dataset}/templates"
    obj_id_2 = 2
    syn_data_type = "train_pbr" # test
    syn_template_path_1 = f"{out_folder}/{syn_data_type}_images_templates/obj_{obj_id:06d}_original" 
    syn_template_path_2 = f"{out_folder}/{syn_data_type}_images_templates/obj_{obj_id_2:06d}_original" 
    syn_template_files_1 = sorted(glob.glob(os.path.join(syn_template_path_1, "*.png")), key=os.path.getmtime)
    syn_template_files_2 = sorted(glob.glob(os.path.join(syn_template_path_2, "*.png")), key=os.path.getmtime)
    syn_template_files = syn_template_files_1 # + syn_template_files_2
    syn_num_templates = len(syn_template_files)
    syn_templates = [np.array(Image.open(template_file).convert("RGB"))[:,:,:3] for template_file in syn_template_files] # This image has 4 channels- the last one is not crucial - maybe about opacity

    # Load Dinov2
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2_vitl14.patch_size = 14
    if torch.cuda.is_available():
        dinov2_vitl14 = torch.nn.DataParallel(dinov2_vitl14).to(device)

    # Extract features of templates
    syn_ref_features = cnos_templates_feature_extraction(
        templates = syn_templates, num_templates = syn_num_templates, dino_model = dinov2_vitl14, device = device
        )

    # Extract feature for all crops
    crop_features = extract_sam_crops_features(masked_images, dinov2_vitl14, device)

    input_features = torch.cat((syn_ref_features, crop_features), dim=0)
    log.info(input_features.shape)

    np.random.seed(42)
    hierarchical_clustering(input_features, n_clusters=2)
if __name__ == "__main__":
    dataset = "icbin"
    rgb_path = "datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000000.png" # f"datasets/bop23_challenge/datasets/{dataset}/test/000048/rgb/000001.png"
    obj_id = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(dataset, rgb_path, obj_id, device)