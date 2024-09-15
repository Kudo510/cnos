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
from src.model.custom_cnos import calculate_similarity as cnos_calculate_similarity
from sklearn.decomposition import PCA
from src.model.foundpose import crop_feature_extraction_3
from src.model.foundpose import (
    crop_feature_extraction, 
    kmeans_clustering, 
    calculate_templates_labels, 
    calculate_templates_vector,
    calculate_crop_vector,
    calculate_similarity,
    templates_feature_extraction,
    templates_feature_extraction_3
)
from src.model.custom_cnos import custom_detections, custom_visualize, custom_detections_2, custom_visualize_2


def _save_final_results(selected_proposals_indices, scene_id, frame_id, sam_detections, dataset, rgb_path, type = "cnos"):
    # Cnos final results
    file_path = f"cnos_analysis/output_npz/{scene_id:06d}_{frame_id:06d}_{type}"
    custom_detections_2(sam_detections, selected_proposals_indices, file_path=file_path, scene_id=scene_id, frame_id=frame_id)
    results = np.load(file_path+".npz")
    dets = []
    for i in range(results["segmentation"].shape[0]):
        det = {
        "scene_id": results["scene_id"],
        "image_id": results["image_id"],
        "category_id": results["category_id"][i],
        "bbox": results["bbox"][i],
        "segmentation": results["segmentation"][i].astype(bool),
        }
        dets.append(det)
    if len(dets) > 0:
        final_result = custom_visualize_2(dataset, rgb_path, dets)
        # Save image
        saved_path = f"cnos_analysis/outputs/extended_templates/{scene_id:06d}_{frame_id:06d}_{type}.png"
        final_result.save(saved_path)
    return 0


def _bow_retrieval(crop_rgb, templates, valid_patch_features, num_valid_patches, dino_model, device):
    crop_num_valid_patches, valid_crop_feature_patches = crop_feature_extraction_3(crop_rgb = crop_rgb, dino_model=dino_model, device = device)
    all_valid_patch_features = torch.cat((valid_crop_feature_patches.to('cpu'), valid_patch_features), dim=0)
    # PCA
    pca = PCA(n_components=256, random_state=5)
    pca_crop_patches_descriptors = pca.fit_transform(np.array(all_valid_patch_features.cpu()))
    
    pca_crop = pca_crop_patches_descriptors[:valid_crop_feature_patches.shape[0]]
    pca_templates = pca_crop_patches_descriptors[valid_crop_feature_patches.shape[0]:]

    kmeans = kmeans_clustering(pca_templates, ncentroids = 2048, niter = 20, verbose = True)
    templates_labels = calculate_templates_labels(num_valid_patches, kmeans, pca_templates)
    templates_vector = calculate_templates_vector(templates_labels = templates_labels, num_clusters = 2048)

    # Assign labels to the data points
    crop_labels = kmeans.index.search(pca_crop, 1)[1].reshape(-1)
    
    crop_vector = calculate_crop_vector(crop_labels = crop_labels, templates_labels = templates_labels, num_clusters = 2048)
    concat_templates_vector = torch.cat([torch.tensor(vector).view(1,-1) for vector in templates_vector]) # Goal torch.Size([642, 2048])

    # Compare crop to templates
    rounded_avg_score, rounded_scores, similar_templates = calculate_similarity(crop_rgb, crop_vector, concat_templates_vector, templates, synthetic=True)
    return rounded_avg_score, rounded_scores, similar_templates


def _move_to_device(segmentor_model, device="cuda"):
    # if there is predictor in the model, move it to device
    if hasattr(segmentor_model, "predictor"):
        segmentor_model.predictor.model = (
            segmentor_model.predictor.model.to(device)
        )
    else:
        segmentor_model.model.setup_model(device=device, verbose=True)


def _extract_object_by_mask(image, mask, width: int = 512):
    mask = Image.fromarray(mask)
    masked_image = Image.composite(
        image, Image.new("RGB", image.size, (0, 0, 0)), mask)
    cropped_image = masked_image.crop(masked_image.getbbox())
    # new_height = width * cropped_image.height // cropped_image.width
    return cropped_image




