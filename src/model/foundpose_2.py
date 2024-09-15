from PIL import Image
import torch
from tqdm import trange
import faiss
import numpy as np
from src.model.utils import Detections
import glob
import os
from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam
from segment_anything.modeling.sam import Sam
from PIL import Image
import numpy as np
import pickle
import argparse
import glob
import logging
import math
import multiprocessing
import os
import random
import shutil
import sys
import time
from functools import partial
from types import SimpleNamespace

import cv2
import distinctipy
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.utils
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pycocotools import mask as mutils
from pycocotools import _mask as coco_mask
from skimage.feature import canny
from skimage.morphology import binary_dilation
from sklearn.decomposition import PCA
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision.ops.boxes import batched_nms, box_area

from segment_anything.modeling.sam import Sam
from segment_anything.utils.amg import rle_to_mask


def custom_detections_2(sam_detections, idx_selected_proposals, file_path, scene_id=1, frame_id=1):
    '''
    For classicication model (constrastive loss)
    '''

    # rgb = Image.open(rgb_path).convert("RGB")
    detections = Detections(sam_detections)
    # mask_post_processing = SimpleNamespace(
    #     min_box_size=0.05,  # relative to image size
    #     min_mask_size=3e-4  # relative to image size
    # )

    # get scores per proposal
    # cosine_metric = PairwiseSimilarity()
    # scores = cosine_metric(proposals_features, real_ref_features.unsqueeze(dim=0))
    # score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
    # score_per_proposal_and_object = torch.mean(
    #     score_per_proposal_and_object, dim=-1
    # )
    # assign each proposal to the object with highest scores
    # score_per_proposal, assigned_idx_object = torch.max(
    #     score_per_proposal_and_object, dim=-1
    # )  # N_query

    # idx_selected_proposals = torch.arange(
    #     len(score_per_proposal), device=score_per_proposal.device
    # )[score_per_proposal > 0.5]
    # pred_idx_objects = assigned_idx_object[idx_selected_proposals]
    # pred_scores = score_per_proposal[idx_selected_proposals]

    
    pred_idx_objects = torch.tensor([1]).repeat(len(idx_selected_proposals)) # temperary class 1 for object 1 only

    # keep only detections with score > conf_threshold
    detections.filter(idx_selected_proposals)
    # detections.add_attribute("scores", pred_scores)
    detections.add_attribute("object_ids", pred_idx_objects)
    # detections.apply_nms_per_object_id(
    #     nms_thresh=0.3
    # )
    detections.to_numpy()

    for ext in [".json", ".npz"]:
        file = file_path + ext
        if os.path.exists(file):
            os.remove(file)
            print(f"File {file} has been deleted.")
        else:
            print(f"File {file} does not exist.")

    # save detections to file
    results = detections.save_to_file_2(
        scene_id=int(scene_id),
        frame_id=int(frame_id),
        runtime=1,
        file_path=file_path,
        dataset_name="icbin",
        return_results=True,
    )

    # num_workers = 10
    # result_paths = [file_path+".npz"]
    # pool = multiprocessing.Pool(processes=num_workers)
    # convert_npz_to_json_with_idx = partial(
    #     convert_npz_to_json,
    #     list_npz_paths=result_paths,
    # )
    # detections = list(
    #     tqdm(
    #         pool.imap_unordered(
    #             convert_npz_to_json_with_idx, range(len(result_paths))
    #         ),
    #         total=len(result_paths),
    #         desc="Converting npz to json",
    #     )
    # )
    # formatted_detections = []
    # for detection in tqdm(detections, desc="Loading results ..."):
    #     formatted_detections.extend(detection)

    # detections_path = f"{file_path}.json"
    # save_json_bop23(detections_path, formatted_detections)
    # print(f"Saved predictions to {detections_path}")

def custom_visualize_2(dataset_name, rgb_path, dets) -> None:
    ''' For classfication model- contrastive loss'''
    if dataset_name in ["hb", "tless"]:
        split = "test_primesense"
    else:
        split = "test"
        
    num_max_objs = 50
    green = [0, 1, 0]  # RGB values for green
    colors = [green] * num_max_objs  # Create a list of green colors
    
    logging.info("Loading detections...")
    # with open(input_file, 'r') as f:
    #     dets = json.load(f)
    
    # conf_threshold = 0.5
    # dets = [det for det in dets if det['score'] > conf_threshold] # keeps the det if it has score > threshold
    # logging.info(f'Keeping only {len(dets)} detections having score > {conf_threshold}') # keeps the det if it has score > threshold
    
    
    # sort by (scene_id, frame_id)
    # dets = sorted(dets, key=lambda x: (x['scene_id'], x['image_id']))
    list_scene_id_and_frame_id = [(det['scene_id'], det['image_id']) for det in dets]
    
    output_dir = "contrastive_learning/outputs" 
    os.makedirs(output_dir, exist_ok=True)
    counter = 0
    for scene_id, image_id in tqdm(list_scene_id_and_frame_id):
        img = Image.open(rgb_path)
        rgb = img.copy()
        img = np.array(img)
        masks, object_ids, scores = [], [], []
        for det in dets:
            if det['scene_id'] == scene_id and det['image_id'] == image_id:
                masks.append(det['segmentation'])
                object_ids.append(det['category_id']-1)  ## category_id is object_id
                # scores.append(det['score'])
        # color_map = {obj_id: color for obj_id in np.unique(object_ids)}
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # img = (255*img).astype(np.uint8)
        
        alpha = 0.33
        for mask_idx, mask in enumerate(masks):
            edge = canny(mask)
            edge = binary_dilation(edge, np.ones((2, 2)))
            obj_id = object_ids[mask_idx]
            temp_id = obj_id - 1

            r = int(255*colors[temp_id][0])
            g = int(255*colors[temp_id][1])
            b = int(255*colors[temp_id][2])
            img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
            img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
            img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
            img[edge, :] = 255

        scene_dir = f"{output_dir}/{dataset_name}{scene_id:06d}" 
        os.makedirs(scene_dir, exist_ok=True)
        save_path = f"{scene_dir}/{image_id:06d}.png"
        img = Image.fromarray(np.uint8(img))
        img.save(save_path)
        prediction = Image.open(save_path)
        # concat side by side in PIL
        img = np.array(img)
        concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        # concat.save(save_path)
        # if counter % 10 == 0:
        #     logging.info(f"Saving {save_path}")
        # counter+=1
    return concat

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


def kmeans_clustering(pca_patches_descriptors, ncentroids = 2048, niter = 20, verbose = True):
    # https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization

    d = pca_patches_descriptors.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True, seed=5)
    kmeans.train(pca_patches_descriptors)
    return kmeans


def calculate_templates_labels(num_valid_patches, kmeans, pca_patches_descriptors):
    labels = kmeans.index.search(pca_patches_descriptors, 1)[1]
    templates_labels = list()
    start_idx = 0
    for num in num_valid_patches:
        end_idx = start_idx + num
        template_labels = labels[start_idx:end_idx].reshape(-1)
        templates_labels.append(template_labels)
        start_idx = end_idx
    return templates_labels


def calculate_templates_vector(templates_labels, num_clusters = 2048):
    # Calculate bag-of-words descriptors of the templates

    templates_vector = list()
    all_occurrences = [np.bincount(templates_label, minlength=2048) for templates_label in templates_labels]
    ni_array = np.sum(np.array(all_occurrences), axis = 0)
    N = len(templates_labels) # Number of templates
    for t in range(len(templates_labels)):
        template_vector = list()
        occurrences = np.bincount(templates_labels[t], minlength=2048)
        for i in range(num_clusters):
            n_it = occurrences[i]
            nt = len(templates_labels[t])
            ni = ni_array[i]
            if ni==0 or nt==0:
                print(i)
            bi = n_it / nt * math.log(N / ni)
            template_vector.append(bi)
        templates_vector.append(np.array(template_vector))
    return templates_vector


def calculate_similarity(crop_rgb, query_features, ref_features, templates, synthetic=False):
    '''
    feature_decriptors: num_proposal, features_dim - here 1,1024
    ref_features : num_templates, features_dim - here 42, 1024
    goal convert both inputs to num_proposal, num_templates, features_dim
    '''
    num_proposals = query_features.shape[0] # Here = 1 for 1 crop
    num_obj = 1
    num_templates = ref_features.shape[0]
    queries = query_features.clone().unsqueeze(1).repeat(1, num_templates, 1) # num_proposal/N_query, num_templates, features_dim
    references = ref_features.clone().unsqueeze(0).repeat(num_proposals, 1, 1)  # num_proposals, num_templates, features_dim
    
    ## ask Evin if we need to normalize- supposingly no 
    # queries = F.normalize(queries, dim=-1)
    # references = F.normalize(references, dim=-1)

    scores = F.cosine_similarity(queries, references, dim=-1) # num_proposals, num_templates

    # get scores per proposal
    score_per_detection, similar_template_indices = torch.topk(scores, k=5, dim=-1) # get top 5 most similar templates
    # get the final confidence score
    score_per_detection = torch.mean(
        score_per_detection, dim=-1
    ) 
    # Check the confidence scores for the similar templates
    similar_scores = scores[:, similar_template_indices[0].to("cpu")]

    similar_templates = []
    for idx in similar_template_indices[0]:
           similar_templates.append(templates[idx])

    # Display the crop
    plt.imshow(crop_rgb)
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    # Round up to two decimal places
    
    rounded_scores = [
        math.ceil(score * 1000) / 1000 if not math.isnan(score) else 0
        for score in similar_scores[0]
    ]
    if not math.isnan(score_per_detection):
        rounded_avg_score = math.ceil(score_per_detection.item() * 1000) / 1000
    else:
        rounded_avg_score = 0 # Set = 0 to make it small to be ignored

    width = 50
    height = 50
    fig = plt.figure(figsize=(7, 7))
    columns = 3
    rows = 2

    for index, template in enumerate(similar_templates):
        fig.add_subplot(rows, columns, index + 1)
        img = template # transpose(1, 2, 0)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Top Template {similar_template_indices[0][index]}')

    plt.tight_layout()
    plt.show()

    # Print the results
    print("Top 5 scores:", rounded_scores)
    print("Average score:", rounded_avg_score)

    return rounded_avg_score, rounded_scores , similar_templates


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


def calculate_crop_vector(crop_labels, templates_labels, num_clusters = 2048):
    # For word_i, term frequency = occurences of word_i within the crop / number of occurences of word_i in all templates). 
    
    # Calculate bag-of-words descriptors of the templates
    all_occurrences_crop = np.bincount(crop_labels, minlength=num_clusters)

    all_occurrences_templates = [np.bincount(templates_label, minlength=num_clusters) for templates_label in templates_labels]
    ni_array = np.sum(np.array(all_occurrences_templates), axis = 0)
    N = len(templates_labels) # Number of templates = 642 

    crop_vector = list()
    for i in range(num_clusters):
        n_it = all_occurrences_crop[i]
        nt = crop_labels.shape[0] # Number of words in crop = 400 
        ni = ni_array[i] + n_it
        bi = n_it / nt * math.log((N+1) / ni)
        crop_vector.append(bi)
    return torch.tensor(crop_vector).view(1,-1) # Goal having features size (1,2048)


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


def resize_and_pad_image(image, target_max=420):
    # Scale image to 420
    scale_factor = target_max / torch.max(torch.tensor(image.shape)) # 420/max of x1,y1,x2,y2
    scaled_image = F.interpolate(image.unsqueeze(0), scale_factor=scale_factor.item())[0] # unsqueeze at  0 - B,C, H, W
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
    
    if scaled_padded_image.shape[1] == 29:
        scaled_padded_image = F.pad(scaled_padded_image, (0, 1, 0, 1), mode='constant', value=0)
    
    if scaled_padded_image.shape[-1] == 223:
        scaled_padded_image = F.pad(scaled_padded_image, (0, 1, 0, 1), mode='constant', value=0)
        
    if scaled_padded_image.shape[-1] == 419:
        scaled_padded_image = F.pad(scaled_padded_image, (0, 1, 0, 1), mode='constant', value=0)
        
    return scaled_padded_image


def filter_out_invalid_templates(patch_features, masks):
    num_valid_patches = list() # List of numbers of valid patches for each template
    valid_patch_features = list()
    for patch_feature, mask in zip(patch_features, masks):
        valid_patches = patch_feature[mask==1]
        valid_patch_features.append(valid_patches)
        num_valid_patches.append(valid_patches.shape[0]) # Append number of  valid patches for the template to the list
    valid_patch_features = torch.cat(valid_patch_features)
    return num_valid_patches, valid_patch_features


def _create_mask(image, threshold=10):
    return (np.sum(np.array(image), axis=2) > threshold).astype(np.uint8)

def _create_mask2(image, threshold=10): # for 3, H, W images
    return (np.sum(np.array(image), axis=0) > threshold).astype(np.uint8)


def crop_feature_extraction_3(crop_rgb, dino_model, device):
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )
    normalized_crop_rgb = rgb_normalize(crop_rgb/255.0).float()
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)

    scaled_padded_crop_rgb = resize_and_pad_image(normalized_crop_rgb).unsqueeze(0) # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals

    # Mask out the crop by clampping at 0,1 for resize image with size of (3, 30, 30)
    resized_crop = resize_and_pad_image(torch.tensor(crop_rgb).permute(2,0,1), target_max=30)
    # mask = np.clip(np.sum(np.array(resized_crop), axis=0), 0, 1, dtype="uint8").reshape(-1)
    mask = _create_mask2(resized_crop).reshape(-1)

    plt.imshow(resized_crop.permute(1,2,0))
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    # Display the image - 10* see lb the crop is normalized same way as the templates- ready to compare the similarity now
    plt.imshow(np.clip(np.sum(np.array(resized_crop), axis=0), 0, 1), cmap=plt.cm.gray)
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    # Extract features from 18th layer of Dinov2 
    layers_list = list(range(24))
    torch.cuda.empty_cache()
    with torch.no_grad(): 
        foundpose_feature_patches= dino_model.module.get_intermediate_layers(scaled_padded_crop_rgb.to(device),
                                                                             n=layers_list, 
                                                                             return_class_token=True)[18][0].reshape(-1,1024)
        cnos_feature_patches = dino_model.module.get_intermediate_layers(scaled_padded_crop_rgb.to(device),
                                                                         n=layers_list, return_class_token=True)[23][0].reshape(-1,1024)
                                                                        #  return_class_token=True)[23][1].reshape(-1,1024).repeat(900,1)
    del dino_model

    # For new idea - concatenating 2 features
    final_feature_patches = torch.cat((foundpose_feature_patches, cnos_feature_patches), dim=1)
    num_valid_patches, valid_patch_features = filter_out_invalid_templates(cnos_feature_patches.unsqueeze(0), torch.tensor(mask).unsqueeze(0))

    # PCA
    # pca = PCA(n_components=256, random_state=5)
    # pca_crop_patches_descriptors = pca.fit_transform(np.array(valid_patch_features.cpu()))
    # print(pca_crop_patches_descriptors.shape)

    return num_valid_patches, valid_patch_features


def templates_feature_extraction_3(templates, template_masks, dino_model, num_templates, device):
    '''
    Use GT masks instead of create that
    Also use filter_out_invalid_templates
    not do any pca- let do pca together later
    '''
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    normalized_templates = [rgb_normalize(template/255.0).float() for template in templates]
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)
    # print("normalized_templates shape", normalized_templates[0].shape)

    scaled_padded_templates = [resize_and_pad_image(normalized_template)
                            for normalized_template in normalized_templates] # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    # print("scaled_padded_templates.shape", len(scaled_padded_templates), scaled_padded_templates[0].shape) 

    # Mask out the templates by clampping at 0,1 for resize image with size of (3, 30, 30)
    masks = [resize_and_pad_image(torch.tensor(mask).unsqueeze(0), target_max=30).flatten() for mask in template_masks]    
    
    # plt.imshow(templates[0])
    # plt.axis('off')  # Optional: Turn off the axis
    # plt.show()

    # plt.imshow(masks[0], cmap=plt.cm.gray)
    # plt.axis('off')  # Optional: Turn off the axis
    # plt.show()

    batch_size = 16
    layers_list = list(range(24))
    template_batches = [scaled_padded_templates[i:i+batch_size] for i in range(0, len(scaled_padded_templates), batch_size)]
    patch_features= list()

    for batch in template_batches:
        batch = torch.stack(batch)
        size = batch.shape[0]
        torch.cuda.empty_cache()
        with torch.no_grad(): 
            foundpose_batch_feature = dino_model.module.get_intermediate_layers(
                batch.to(device), n=layers_list, return_class_token=True
                )[18][0].reshape(size,-1,1024).cpu()
            cnos_batch_feature = dino_model.module.get_intermediate_layers(
                batch.to(device), n=layers_list, return_class_token=True
                )[23][0].reshape(size,-1,1024).cpu()
            # batch_feature = torch.cat((foundpose_batch_feature, cnos_batch_feature.repeat(1,900,1)), dim=-1)
            batch_feature = torch.cat((foundpose_batch_feature, cnos_batch_feature), dim=-1)
        patch_features.append(cnos_batch_feature.to('cpu'))
        del batch_feature
    patch_features = torch.cat(patch_features)
    del dino_model

    num_valid_patches, valid_patch_features = filter_out_invalid_templates(patch_features, masks)

    # # PCA
    # pca = PCA(n_components=128, random_state=5)
    # pca_patches_descriptors = pca.fit_transform(np.array(valid_patch_features))
    # print("pca_crop_patches_descriptors.shape", pca_patches_descriptors.shape)

    return num_valid_patches, valid_patch_features

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


def foundpose_with_cnos_single(rgb_path, custom_sam_model, scene_id, frame_id, obj_id=1, dataset="icbin"):
    '''
    Use both local and global features for FoundPose
    '''
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
    
    # Load dinov2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    if torch.cuda.is_available():
        dinov2_vitl14 = torch.nn.DataParallel(dinov2_vitl14).to(device)  # Use DataParallel for multiple GPUs

    # Extract features for templates
    syn_data_type = "train_pbr" # test
    out_folder = f"foundpose_analysis/{dataset}/templates"

    syn_template_path_1 = f"{out_folder}/{syn_data_type}_images_templates/obj_{obj_id:06d}_original" 
    # syn_template_path_1 = f"datasets/bop23_challenge/datasets/templates_pyrender/icbin/obj_000001"
    syn_template_files_1 = sorted(glob.glob(os.path.join(syn_template_path_1, "*.png")), key=os.path.getmtime)
    syn_template_files = syn_template_files_1 
    syn_num_templates = len(syn_template_files)
    syn_templates = [np.array(Image.open(template_file).convert("RGB"))[:,:,:3] for template_file in syn_template_files] # This image has 4 channels- the last one is not crucial - maybe about opacity


    # Foundpose results
    # Load original templates when before putting through dinov2 we also apply transformation.
    # def _create_mask(image, threshold=10):
    #     return (np.sum(np.array(image), axis=2) > threshold).astype(np.uint8)
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
    foundpose_similar_templates = list()
    for i in trange(55, len(masked_images)):
    # for i in range(0,2):
        crop_rgb = np.array( masked_images[i]) # (124, 157, 3)
        rounded_avg_score, rounded_scores, similar_templates = _bow_retrieval(crop_rgb, syn_templates, syn_valid_patch_features, syn_num_valid_patches, dino_model=dinov2_vitl14, device=device)
        foundpose_average_scores.append(rounded_avg_score)
        foundpose_top_5_scores.append(rounded_scores)
        # foundpose_similar_templates.append(similar_templates)
    # save the score dict
    score_dict = {
        "foundpose_avg_scores" : foundpose_average_scores,
        "foundpose_top_5_scores" : foundpose_top_5_scores,
        # "foundpose_similar_templates" : foundpose_similar_templates
    }

    # with open(f'cnos_foundpose_analysis/score_dicts/score_dict_{scene_id:06d}_{frame_id:06d}.pkl', 'wb') as file:
    #     pickle.dump(score_dict, file)

    # foundpose_selected_proposals_indices = [i for i, a_s in enumerate(foundpose_average_scores) if a_s >0.2]
    # foundpose_selected_proposals_scores = [a_s for i, a_s in enumerate(foundpose_average_scores) if a_s >0.2]

    # Cnos
    # Foundpose
    # _save_final_results(selected_proposals_indices=foundpose_selected_proposals_indices, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "foundpose")
    # Cnos_foundpose

    return score_dict
    # final_result   