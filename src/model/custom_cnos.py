from src.model.foundpose import resize_and_pad_image

import torchvision.transforms as T
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.model.loss import PairwiseSimilarity, Similarity
from torchvision.io import read_image
import math
import os
import sys
import shutil
from tqdm import tqdm
import time
import logging
import os.path as osp
from hydra import initialize, compose
from hydra.utils import instantiate
import argparse
import glob
from src.utils.bbox_utils import CropResizePad
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
from src.model.utils import Detections, convert_npz_to_json
from src.utils.inout import save_json_bop23
import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
from types import SimpleNamespace

# set level logging
logging.basicConfig(level=logging.INFO)

def visualize(rgb, detections, save_path="./tmp/tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # img = (255*img).astype(np.uint8)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    for mask_idx, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = det["category_id"]
        temp_id = obj_id - 1

        r = int(255*colors[temp_id][0])
        g = int(255*colors[temp_id][1])
        b = int(255*colors[temp_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        img[edge, :] = 255
    
    out_path = save_path + "/tmp.png"
    img = Image.fromarray(np.uint8(img))
    img.save(out_path)
    prediction = Image.open(out_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat
        
def modified_run_inference(template_dir, rgb_path, detections, ref_feats, decriptors, num_max_dets = 20, conf_threshold = 0.5, stability_score_thresh = 0.97):
    '''
    descriptor are the descriptors of all SAM proposals for query image'''
    detections = Detections(detections)

    mask_post_processing = SimpleNamespace(
        min_box_size=0.05,  # relative to image size
        min_mask_size=3e-4  # relative to image size
    )

    detections.remove_very_small_detections(
            config=mask_post_processing
        )
    # decriptors = model.descriptor_model.forward(np.array(rgb), detections)
    
    # get scores per proposal
    metric = Similarity() 
    scores = metric(decriptors[:, None, :], ref_feats[None, :, :])
    score_per_detection = torch.topk(scores, k=1, dim=-1)[0]
    score_per_detection = torch.mean(
        score_per_detection, dim=-1
    )
    
    # get top-k detections
    scores, index = torch.topk(score_per_detection, k=num_max_dets, dim=-1)
    detections.filter(index)
    
    # keep only detections with score > conf_threshold
    detections.filter(scores>conf_threshold)
    detections.add_attribute("scores", scores)
    detections.add_attribute("object_ids", torch.zeros_like(scores))
        
    detections.to_numpy()

    if "test" in template_dir:  
        templates_type = "test"
    elif "train_pbr" in template_dir:
        templates_type = "train_pbr"
    elif "pyrender" in template_dir:
        templates_type = "pyrender"
    
    # save detections to file to submit in bop
    # results = detections.save_to_file(
    #     scene_id=int(scene_id),
    #     frame_id=int(frame_id),
    #     runtime=runtime,
    #     file_path=file_path,
    #     dataset_name=self.dataset_name,
    #     return_results=True,
    # )

    save_path = f"output_cnos_analysis_5/{templates_type}/cnos_results/detection"
    # Create the directory if it does not exist
    os.makedirs(save_path, exist_ok=True)
    detections.save_to_file(0, 0, 0, save_path, "custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path +".json", detections)
    vis_img = visualize(rgb, detections, save_path)

    plt.figure(figsize=(12, 8))  # width, height in inches
    # Display the image
    plt.imshow(vis_img)
    plt.axis('off')  # Optionally turn off the axis
    plt.show()
    vis_img.save(f"output_cnos_analysis_5/{templates_type}/cnos_results/vis.png")

def calculate_similarity(crop_rgb, feature_decriptors, ref_features,templates):
    metric = Similarity() 
    # get scores per proposal
    scores = metric(feature_decriptors[:, None, :], ref_features[None, :, :]) # should get  # N_proposals x N_objects x N_templates -get only 1,42 as num_prosals*num_templates instead
    score_per_detection, similar_template_indices = torch.topk(scores, k=5, dim=-1) # get top 5 most similar templates
    # get the final confidence score
    score_per_detection = torch.mean(score_per_detection, dim=-1) 
    # Check the confidence scores for the similar templates
    similar_scores = scores[:, similar_template_indices[0].to("cpu")]

    # Display the crop
    plt.imshow(crop_rgb)
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    # Round up to two decimal places
    rounded_scores = [math.ceil(score * 1000) / 1000 for score in similar_scores[0]]
    rounded_avg_score = math.ceil(score_per_detection.item() * 1000) / 1000

    width = 50
    height = 50
    fig = plt.figure(figsize=(7, 7))
    columns = 3
    rows = 2

    for i, index in enumerate(similar_template_indices[0]):
        fig.add_subplot(rows, columns, i + 1)
        img = templates[index] # permute(1, 2, 0)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Top Template {index}')

    plt.tight_layout()
    plt.show()

    # Print the results
    print("Top 5 scores:", rounded_scores)
    print("Average score:", rounded_avg_score)

    return


def modified_cnos_crop_feature_extraction(crop_rgb, dino_model, device):
    ''' 
    resize input to 420*420
    Extract features at layer 18th - size 30*30,1024
    Apply PCA to reduce size to 30*30*3 only ( can test with different size)
    Flatten the tensor to 2700
    '''
    
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    normalized_crop_rgb = rgb_normalize(crop_rgb/255.0).float()
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)

    scaled_padded_crop_rgb = resize_and_pad_image(normalized_crop_rgb, target_max=420).unsqueeze(0) # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    print("scaled_padded_crop_rgb.shape", scaled_padded_crop_rgb.shape)


    # Extract features from 18th layer of Dinov2 
    layers_list = list(range(24))
    torch.cuda.empty_cache()
    with torch.no_grad(): 
        feature_patches= dino_model.module.get_intermediate_layers(
            scaled_padded_crop_rgb.to(device), n=layers_list, return_class_token=True)[18][0].reshape(-1,1024).to("cpu")
    del dino_model

    # PCA
    pca = PCA(n_components=3)
    pca_patches_descriptors = pca.fit_transform(np.array(feature_patches)).flatten()

    return torch.tensor(pca_patches_descriptors).unsqueeze(0) # patch_features.squeeze().to("cuda:0")


def cnos_crop_feature_extraction(crop_rgb, dino_model, device):
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    normalized_crop_rgb = rgb_normalize(crop_rgb/255.0).float()
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)

    scaled_padded_crop_rgb = resize_and_pad_image(normalized_crop_rgb, target_max=224).unsqueeze(0) # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    # print("scaled_padded_crop_rgb.shape", scaled_padded_crop_rgb.shape)


    # Extract features from 18th layer of Dinov2 
    layers_list = list(range(24))
    torch.cuda.empty_cache()
    with torch.no_grad(): 
        feature_patches= dino_model.module.get_intermediate_layers(
            scaled_padded_crop_rgb.to(device), n=layers_list, return_class_token=True)[23][1].reshape(-1,1024)
    del dino_model

    return feature_patches


def cnos_templates_feature_extraction(templates, dino_model, num_templates, device):
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    normalized_templates = [rgb_normalize(template/255.0).float() for template in templates]
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)
    print("normalized_templates shape", normalized_templates[0].shape)

    scaled_padded_templates = [resize_and_pad_image(normalized_template, target_max=224)
                            for normalized_template in normalized_templates] # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    print("scaled_padded_templates.shape", len(scaled_padded_templates), scaled_padded_templates[0].shape) 

    
    plt.imshow(templates[0]) #, cmap=plt.cm.gray)
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    batch_size = 16
    layers_list = list(range(24))
    template_batches = [scaled_padded_templates[i:i+batch_size] for i in range(0, len(scaled_padded_templates), batch_size)]
    patch_features= list()

    for batch in template_batches:
        batch = torch.stack(batch)
        size = batch.shape[0]
        torch.cuda.empty_cache()
        with torch.no_grad(): 
            batch_feature = dino_model.module.get_intermediate_layers(
                batch.to(device), n=layers_list, return_class_token=True
                )[23][1].reshape(size,-1,1024).cpu()
        patch_features.append(batch_feature.to('cpu'))
        del batch_feature
    patch_features = torch.cat(patch_features)
    del dino_model

    return patch_features.squeeze().to("cuda:0")


def modified_cnos_templates_feature_extraction(templates, dino_model, num_templates, device):
    ''' 
    Extract features at layer 18th - size 30*30,1024
    Apply PCA to reduce size to 30*30*3 only ( can test with different size)
    Flatten the tensor to 2700
    '''

    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    normalized_templates = [rgb_normalize(template/255.0).float() for template in templates]
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)
    print("normalized_templates shape", normalized_templates[0].shape)

    scaled_padded_templates = [resize_and_pad_image(normalized_template, target_max=420)
                            for normalized_template in normalized_templates] # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    print("scaled_padded_templates.shape", len(scaled_padded_templates), scaled_padded_templates[0].shape) 

    
    plt.imshow(templates[0]) #, cmap=plt.cm.gray)
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    batch_size = 16
    layers_list = list(range(24))
    template_batches = [scaled_padded_templates[i:i+batch_size] for i in range(0, len(scaled_padded_templates), batch_size)]
    patch_features= list()

    for batch in template_batches:
        batch = torch.stack(batch)
        size = batch.shape[0]
        torch.cuda.empty_cache()
        with torch.no_grad(): 
            batch_feature = dino_model.module.get_intermediate_layers(
                batch.to(device), n=layers_list, return_class_token=True
                )[18][0].reshape(size,-1,1024).cpu() 
        patch_features.append(batch_feature.to('cpu'))
        del batch_feature
    patch_features = torch.cat(patch_features)
    del dino_model

    # PCA
    pca = PCA(n_components=3)
    pca_patches_descriptors = [pca.fit_transform(np.array(patch_feature)).flatten() for patch_feature in patch_features]

    return torch.tensor(pca_patches_descriptors)# patch_features.squeeze().to("cuda:0")







