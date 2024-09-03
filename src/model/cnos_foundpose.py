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
from src.model.foundpose import calculate_similarity, crop_feature_extraction_3
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
    file_path = f"cnos_foundpose_analysis/output_npz/{scene_id:06d}_{frame_id:06d}_{type}"
    custom_detections_2(sam_detections, selected_proposals_indices, file_path=file_path, scene_id=scene_id, frame_id=frame_id)
    results = np.load(file_path+".npz")
    dets = []
    for i in range(results["segmentation"].shape[0]):
        det = {
        "scene_id": results["scene_id"],
        "image_id": results["image_id"],
        "category_id": results["category_id"][i],
        "bbox": results["bbox"][i],
        "segmentation": results["segmentation"][i],
        }
        dets.append(det)
    if len(dets) > 0:
        final_result = custom_visualize_2(dataset, rgb_path, dets)
        # Save image
        saved_path = f"cnos_foundpose_analysis/output_images_different_thresholds/{scene_id:06d}_{frame_id:06d}_{type}.png"
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
    rounded_avg_score, rounded_scores = calculate_similarity(crop_rgb, crop_vector, concat_templates_vector, templates, synthetic=True)
    return rounded_avg_score, rounded_scores

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
    
def cnos_foundpose(rgb_path, scene_id, frame_id, obj_id=1, dataset="icbin"):
    model_type = "vit_h"
    checkpoint_dir =  "datasets/bop23_challenge/pretrained/segment-anything"
    sam_model = load_sam(model_type, checkpoint_dir)
    custom_sam_model = CustomSamAutomaticMaskGenerator(sam=sam_model) 

    _move_to_device(custom_sam_model)
    rgb = Image.open(rgb_path).convert("RGB")
    sam_detections = custom_sam_model.generate_masks(np.array(rgb))

    masked_images = []
    for mask in sam_detections["masks"].cpu():
        binary_mask = np.array(mask) * 255
        binary_mask = binary_mask.astype(np.uint8)
        masked_image = _extract_object_by_mask(rgb, binary_mask)
        masked_images.append(masked_image)
    
    # Load dinov2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    if torch.cuda.is_available():
        dinov2_vitl14 = torch.nn.DataParallel(dinov2_vitl14).to(device)  # Use DataParallel for multiple GPUs

    # Extract features for templates
    syn_data_type = "train_pbr" # test
    out_folder = f"foundpose_analysis/{dataset}/templates"

    syn_template_path_1 = f"{out_folder}/{syn_data_type}_images_templates/obj_{obj_id:06d}_original" 
    syn_template_files_1 = sorted(glob.glob(os.path.join(syn_template_path_1, "*.png")), key=os.path.getmtime)
    syn_template_files = syn_template_files_1 
    syn_num_templates = len(syn_template_files)
    syn_templates = [np.array(Image.open(template_file).convert("RGB"))[:,:,:3] for template_file in syn_template_files] # This image has 4 channels- the last one is not crucial - maybe about opacity
    syn_ref_features = cnos_templates_feature_extraction(
        templates = syn_templates, num_templates = syn_num_templates, dino_model = dinov2_vitl14, device = device
        )

    # Cnos Results
    cnos_avg_scores = list()
    cnos_top_5_scores = list()
    # normal crop
    for i in trange(len(masked_images)):
        crop_rgb = np.array(masked_images[i]) # (124, 157, 3)
        normal_features = cnos_crop_feature_extraction(crop_rgb, dinov2_vitl14, device)

        rounded_avg_score, rounded_scores = cnos_calculate_similarity(crop_rgb, normal_features, syn_ref_features, syn_templates)
        cnos_avg_scores.append(rounded_avg_score)
        cnos_top_5_scores.append(rounded_scores)

    # Foundpose results
    # Load original templates when before putting through dinov2 we also apply transformation.
    def _create_mask(image, threshold=10):
        return (np.sum(np.array(image), axis=2) > threshold).astype(np.uint8)
    mask_syn_templates = [_create_mask(syn_temp) for syn_temp in syn_templates] # This image has 4 channels- the last one is not crucial - maybe about opacity

    syn_num_valid_patches, syn_valid_patch_features = templates_feature_extraction_3(
        templates = syn_templates, 
        template_masks = mask_syn_templates, 
        num_templates = syn_num_templates, 
        dino_model = dinov2_vitl14, 
        device = device
    )
    
    foundpose_average_scores = list()
    foundpose_top_5_scores = list()
    for i in trange(len(masked_images)):
    # for i in range(0,2):
        crop_rgb = np.array( masked_images[i]) # (124, 157, 3)
        rounded_avg_score, rounded_scores = _bow_retrieval(crop_rgb, syn_templates, syn_valid_patch_features, syn_num_valid_patches, dino_model=dinov2_vitl14, device=device)
        foundpose_average_scores.append(rounded_avg_score)
        foundpose_top_5_scores.append(rounded_scores)

    # save the score dict
    score_dict = {
        "cnos_avg_scores" : cnos_avg_scores,
        "cnos_top_5_scores" : cnos_top_5_scores,
        "foundpose_avg_scores" : foundpose_average_scores,
        "foundpose_top_5_scores" : foundpose_top_5_scores
    }

    with open(f'cnos_foundpose_analysis/score_dicts/score_dict_{scene_id:06d}_{frame_id:06d}.pkl', 'wb') as file:
        pickle.dump(score_dict, file)

    combined_avg_scores = [(cnos_avg_scores[i] + foundpose_average_scores[i])/2 for i in range(len(foundpose_average_scores))]

    selected_proposals_indices = [i for i, a_s in enumerate(combined_avg_scores) if a_s >0.35]
    selected_proposals_scores = [a_s for i, a_s in enumerate(combined_avg_scores) if a_s >0.35]
    cnos_selected_proposals_indices = [i for i, a_s in enumerate(cnos_avg_scores) if a_s >0.5]
    cnos_selected_proposals_scores = [a_s for i, a_s in enumerate(cnos_avg_scores) if a_s >0.5]
    foundpose_selected_proposals_indices = [i for i, a_s in enumerate(foundpose_average_scores) if a_s >0.2]
    foundpose_selected_proposals_scores = [a_s for i, a_s in enumerate(foundpose_average_scores) if a_s >0.2]

    # Cnos
    _save_final_results(selected_proposals_indices=cnos_selected_proposals_indices, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "cnos")
    # Foundpose
    _save_final_results(selected_proposals_indices=foundpose_selected_proposals_indices, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "foundpose")
    # Cnos_foundpose
    _save_final_results(selected_proposals_indices=selected_proposals_indices, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "cnos_foundpose")

    return 0
    # final_result


def cnos_different_thresholds(rgb_path, custom_sam_model, scene_id, frame_id, obj_id=1, dataset="icbin"):

    rgb = Image.open(rgb_path).convert("RGB")
    sam_detections = custom_sam_model.generate_masks(np.array(rgb))

    masked_images = []
    for mask in sam_detections["masks"].cpu():
        binary_mask = np.array(mask) * 255
        binary_mask = binary_mask.astype(np.uint8)
        masked_image = _extract_object_by_mask(rgb, binary_mask)
        masked_images.append(masked_image)
    
    # Load dinov2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    if torch.cuda.is_available():
        dinov2_vitl14 = torch.nn.DataParallel(dinov2_vitl14).to(device)  # Use DataParallel for multiple GPUs

    # Extract features for templates
    syn_data_type = "train_pbr" # test
    out_folder = f"foundpose_analysis/{dataset}/templates"

    syn_template_path_1 = f"{out_folder}/{syn_data_type}_images_templates/obj_{obj_id:06d}_original" 
    syn_template_files_1 = sorted(glob.glob(os.path.join(syn_template_path_1, "*.png")), key=os.path.getmtime)
    syn_template_files = syn_template_files_1 
    syn_num_templates = len(syn_template_files)
    syn_templates = [np.array(Image.open(template_file).convert("RGB"))[:,:,:3] for template_file in syn_template_files] # This image has 4 channels- the last one is not crucial - maybe about opacity
    syn_ref_features = cnos_templates_feature_extraction(
        templates = syn_templates, num_templates = syn_num_templates, dino_model = dinov2_vitl14, device = device
        )

    # Cnos Results
    cnos_avg_scores = list()
    cnos_top_5_scores = list()
    # normal crop
    for i in trange(len(masked_images)):
        crop_rgb = np.array(masked_images[i]) # (124, 157, 3)
        normal_features = cnos_crop_feature_extraction(crop_rgb, dinov2_vitl14, device)

        rounded_avg_score, rounded_scores = cnos_calculate_similarity(crop_rgb, normal_features, syn_ref_features, syn_templates)
        cnos_avg_scores.append(rounded_avg_score)
        cnos_top_5_scores.append(rounded_scores)

    
    cnos_selected_proposals_indices_05 = [i for i, a_s in enumerate(cnos_avg_scores) if a_s >0.5]
    cnos_selected_proposals_scores_05 = [a_s for i, a_s in enumerate(cnos_avg_scores) if a_s >0.5]

    cnos_selected_proposals_indices_04 = [i for i, a_s in enumerate(cnos_avg_scores) if a_s >0.4]
    cnos_selected_proposals_scores_04 = [a_s for i, a_s in enumerate(cnos_avg_scores) if a_s >0.4]

    cnos_selected_proposals_indices_03 = [i for i, a_s in enumerate(cnos_avg_scores) if a_s >0.3]
    cnos_selected_proposals_scores_03 = [a_s for i, a_s in enumerate(cnos_avg_scores) if a_s >0.3]

    cnos_selected_proposals_indices_02 = [i for i, a_s in enumerate(cnos_avg_scores) if a_s >0.2]
    cnos_selected_proposals_scores_02 = [a_s for i, a_s in enumerate(cnos_avg_scores) if a_s >0.2]


    # Cnos
    # _save_final_results(selected_proposals_indices=cnos_selected_proposals_indices_05, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "cnos_05")
    # # Foundpose
    _save_final_results(selected_proposals_indices=cnos_selected_proposals_indices_04, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "cnos_04")
    # Cnos_foundpose
    # _save_final_results(selected_proposals_indices=cnos_selected_proposals_indices_03, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "cnos_03")
    # _save_final_results(selected_proposals_indices=cnos_selected_proposals_indices_02, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "cnos_02")

    return 0
    # final_result












    
























