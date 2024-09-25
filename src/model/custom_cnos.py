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

from src.model.constrastive_learning import ContrastiveLearningModel
from src.model.constrastive_learning_utils import resize_and_pad_image
from src.model.loss import PairwiseSimilarity, Similarity
from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam
from src.model.utils import BatchedData, Detections, convert_npz_to_json
from src.utils.bbox_utils import CropResizePad
from src.utils.inout import load_json, save_json_bop23

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
    descriptor are the descriptors of all SAM proposals for query image
    '''
    rgb = Image.open(rgb_path).convert("RGB")
    detections = Detections(detections)

    mask_post_processing = SimpleNamespace(
        min_box_size=0.05,  # relative to image size
        min_mask_size=3e-4  # relative to image size
    )

    # detections.remove_very_small_detections(
    #         config=mask_post_processing
    #     )
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
    # plt.imshow(crop_rgb)
    # plt.axis('off')  # Optional: Turn off the axis
    # plt.show()

    # Round up to two decimal places
    rounded_scores = [math.ceil(score * 1000) / 1000 for score in similar_scores[0]]
    rounded_avg_score = math.ceil(score_per_detection.item() * 1000) / 1000

    width = 50
    height = 50
    fig = plt.figure(figsize=(7, 7))
    columns = 3
    rows = 2

    # for i, index in enumerate(similar_template_indices[0]):
    #     fig.add_subplot(rows, columns, i + 1)
    #     img = templates[index] # permute(1, 2, 0)
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.title(f'Top Template {index}')

    # plt.tight_layout()
    # plt.show()

    # # Print the results
    # print("Top 5 scores:", rounded_scores)
    # print("Average score:", rounded_avg_score)

    return rounded_avg_score, rounded_scores


def check_similarity(best_model_path, crop_rgb, templates, device):
    '''
    Use Model to check if the 2 images are similar 
    1 yes, 0 no
    '''

    transform = T.Compose(
        [
            # transforms.Lambda(lambda x: x / 255.0),  # Ensures the scaling by 255.0
            # T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model = ContrastiveLearningModel(device)
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)
    img1 = transform(resize_and_pad_image(crop_rgb, target_max=224)).unsqueeze(0).float().to(device) # Must be normalized with size of 3, 224,224 and in torch

    for i, temp in enumerate(templates[:]):
        img2 = transform(resize_and_pad_image(np.transpose(temp, (2,0,1)))).unsqueeze(0).float().to(device)

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            
            outputs_test = model(img1, img2)
            # _, predicted = torch.max(outputs_test.cpu(), 1)
            if outputs_test>=0.5:
                predicted = 1
            else:
                predicted = 0     

            print(f"Prediction: {predicted}")
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(np.transpose(crop_rgb, (1,2,0)))
            # plt.axis('off')
            # plt.subplot(1, 2, 2)
            # plt.imshow(temp)
            # plt.axis('off')
            # plt.show()

def check_similarity_2(best_model_path, crop_rgb, templates, device):
    '''
    Use Model to check if the 2 images are similar  but for multiple templates at the same time - then check if one of the prediction = 1 then it means that 
    1 yes, 0 no
    '''
    transform = T.Compose(
        [
            # transforms.Lambda(lambda x: x / 255.0),  # Ensures the scaling by 255.0
            # T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model = ContrastiveLearningModel(device)
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    transformed_temps = []
    for i, temp in enumerate(templates[:]):
        img2 = transform(resize_and_pad_image(np.transpose(temp, (2,0,1)))).float().to(device)
        transformed_temps.append(img2)
    # Randomly select 5 unique images
    num_templates = 162
    selected_temps = random.sample(transformed_temps, num_templates)
    # Stack the selected images to create a tensor of shape (5, 2, 224, 224)
    stacked_temps = torch.stack(selected_temps)

    # Convert from 1,3,224,224 to 162,3,224,224 - but 162 is too much -try with 5 templates only
    img1 = transform(resize_and_pad_image(crop_rgb, target_max=224)).unsqueeze(0).float().to(device) # Must be normalized with size of 3, 224,224 and in torch
    stacked_img1 = img1.repeat(num_templates,1,1,1)

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        
        output1, output2 = model(stacked_img1, stacked_temps)
        # _, predicted = torch.max(outputs_test.cpu(), 1)   
        euclidean_distance = nn.functional.pairwise_distance(output1, output2).cpu()

        average_score = torch.sum(euclidean_distance, dim=0)/euclidean_distance.shape[0]
        if average_score < 1:
            class_name = 1
        else:
            class_name = 0
        print(f"Prediction is: {euclidean_distance}, {average_score} as {class_name}")

    # result = torch.sum(predicted) >= 1
    return class_name


def check_similarity_3(best_model_path, masked_images, templates, aggreation_num_templates, device):
    '''
    Use Model to check if the 2 images are similar  but for multiple templates at the same time - then check if one of the prediction = 1 then it means that 
    1 yes, 0 no
    '''
    # Set radom seed
    random.seed(10)
    
    transform = T.Compose(
        [
            # transforms.Lambda(lambda x: x / 255.0),  # Ensures the scaling by 255.0
            # T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model = ContrastiveLearningModel(device)
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    transformed_temps = []
    for i, temp in enumerate(templates):
        img2 = transform(resize_and_pad_image(np.transpose(temp, (2,0,1)))).float().to(device)
        transformed_temps.append(img2)

    selected_temps = torch.stack(random.choices(transformed_temps, k=len(masked_images)*aggreation_num_templates)) # Generate a list of 167 images randomly chosen from the 162 images
    # Stack the selected images to create a tensor of shape (5, 2, 224, 224)
    # stacked_temps = torch.stack(selected_temps)
    # transformed_temps = torch.stack(transformed_temps)

    # Convert from 1,3,224,224 to 162,3,224,224 - but 162 is too much -try with 5 templates only
    crops = [np.transpose(np.array(img)[:,:,:3]/255.0, (2,0,1)) for img in masked_images]
    transformed_crops = torch.stack([transform(resize_and_pad_image(crop, target_max=224)).float().to(device)
                                        for crop in crops]) # Must be normalized with size of 3, 224,224 and in torch
    # stacked_img1 = img1.repeat(num_templates,1,1,1)

    model.eval()
    with torch.no_grad():
        outputs_test = []
        for i in range(aggreation_num_templates):
            t = selected_temps[len(masked_images)*i : len(masked_images)*(i+1)]
            outputs1, outputs2 = model(transformed_crops, t)
            euclidean_distance = nn.functional.pairwise_distance(outputs1, outputs2)
            outputs_test.append(euclidean_distance)

        outputs_test = torch.stack(outputs_test)
        
        # Average 5 times
        average_outputs_test = torch.sum(outputs_test, dim=0)/outputs_test.shape[0]
        average_predicted = (average_outputs_test < 1).int() 
        # Max out of 5
        max_outputs_test, _ = torch.max(outputs_test, dim=0)
        max_predicted = (max_outputs_test < 1).int() 
        # Min out of 5
        min_outputs_test, _ = torch.min(outputs_test, dim=0)
        min_predicted = (min_outputs_test < 1).int() 

        # print(f"Prediction of index{i} is: {outputs_test} as {predicted}")

    average_indices = (average_predicted.squeeze() == 1).nonzero(as_tuple=False).flatten()
    max_indices = (max_predicted.squeeze() == 1).nonzero(as_tuple=False).flatten()
    min_indices = (min_predicted.squeeze() == 1).nonzero(as_tuple=False).flatten()

    average_prob = average_outputs_test[average_indices]
    max_prob = max_outputs_test[max_indices]
    min_prob = max_outputs_test[min_indices]
    return average_indices, max_indices, min_indices, average_prob, max_prob, min_prob, average_outputs_test, outputs_test

def check_similarity_cosine(best_model_path, masked_images, templates, aggreation_num_templates, device):
    '''
    This is similarity checking_3 but the scores will be as the cosine similariy
    '''
    # Set radom seed
    random.seed(10)
    
    transform = T.Compose(
        [
            # transforms.Lambda(lambda x: x / 255.0),  # Ensures the scaling by 255.0
            # T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model = ContrastiveLearningModel(device)
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    transformed_temps = []
    for i, temp in enumerate(templates):
        img2 = transform(resize_and_pad_image(np.transpose(temp, (2,0,1)))).float().to(device)
        transformed_temps.append(img2)

    selected_temps = torch.stack(random.choices(transformed_temps, k=len(masked_images)*aggreation_num_templates)) # Generate a list of 167 images randomly chosen from the 162 images
    # Stack the selected images to create a tensor of shape (5, 2, 224, 224)
    # stacked_temps = torch.stack(selected_temps)
    # transformed_temps = torch.stack(transformed_temps)

    # Convert from 1,3,224,224 to 162,3,224,224 - but 162 is too much -try with 5 templates only
    crops = [np.transpose(np.array(img)[:,:,:3]/255.0, (2,0,1)) for img in masked_images]
    transformed_crops = torch.stack([transform(resize_and_pad_image(crop, target_max=224)).float().to(device)
                                        for crop in crops]) # Must be normalized with size of 3, 224,224 and in torch
    # stacked_img1 = img1.repeat(num_templates,1,1,1)

    model.eval()
    with torch.no_grad():
        outputs_test = []
        for i in range(aggreation_num_templates):
            t = selected_temps[len(masked_images)*i : len(masked_images)*(i+1)]
            outputs1, outputs2 = model(transformed_crops, t)
            euclidean_distance = nn.functional.pairwise_distance(outputs1, outputs2)
            outputs_test.append(euclidean_distance)

        outputs_test = torch.stack(outputs_test)
        
        # Average 5 times
        average_outputs_test = torch.sum(outputs_test, dim=0)/outputs_test.shape[0]
        average_predicted = (average_outputs_test < 1).int() 
        # Max out of 5
        max_outputs_test, _ = torch.max(outputs_test, dim=0)
        max_predicted = (max_outputs_test < 1).int() 
        # Min out of 5
        min_outputs_test, _ = torch.min(outputs_test, dim=0)
        min_predicted = (min_outputs_test < 1).int() 

        # print(f"Prediction of index{i} is: {outputs_test} as {predicted}")

    average_indices = (average_predicted.squeeze() == 1).nonzero(as_tuple=False).flatten()
    average_scores = (4-(average_outputs_test[average_indices]))/4
    max_indices = (max_predicted.squeeze() == 1).nonzero(as_tuple=False).flatten()
    min_indices = (min_predicted.squeeze() == 1).nonzero(as_tuple=False).flatten()

    average_prob = average_outputs_test[average_indices]
    max_prob = max_outputs_test[max_indices]
    min_prob = max_outputs_test[min_indices]
    return average_indices, average_scores, max_indices, min_indices, average_prob, max_prob, min_prob, average_outputs_test, outputs_test


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

    # # PCA
    # pca = PCA(n_components=3, random_state=5)
    # pca_patches_descriptors = pca.fit_transform(np.array(feature_patches)).flatten()

    return feature_patches # torch.tensor(pca_patches_descriptors).unsqueeze(0) # patch_features.squeeze().to("cuda:0")


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

    # Display the crop
    # plt.imshow(scaled_padded_crop_rgb.squeeze(0).permute(1,2,0))
    # plt.axis('off')  # Optional: Turn off the axis
    # plt.show()

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
    # print("normalized_templates shape", normalized_templates[0].shape)

    scaled_padded_templates = [resize_and_pad_image(normalized_template, target_max=224)
                            for normalized_template in normalized_templates] # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    # print("scaled_padded_templates.shape", len(scaled_padded_templates), scaled_padded_templates[0].shape) 

    
    # plt.imshow(templates[0]) #, cmap=plt.cm.gray)
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
    # print("normalized_templates shape", normalized_templates[0].shape)

    scaled_padded_templates = [resize_and_pad_image(normalized_template, target_max=420)
                            for normalized_template in normalized_templates] # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    #print("scaled_padded_templates.shape", len(scaled_padded_templates), scaled_padded_templates[0].shape) 

    
    # plt.imshow(templates[0]) #, cmap=plt.cm.gray)
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
            batch_feature = dino_model.module.get_intermediate_layers(
                batch.to(device), n=layers_list, return_class_token=True
                )[18][0].reshape(size,-1,1024).cpu() 
        patch_features.append(batch_feature.to('cpu'))
        del batch_feature
    patch_features = torch.cat(patch_features)
    del dino_model

    # PCA
    # pca = PCA(n_components=3, random_state=5)
    # pca_patches_descriptors = [pca.fit_transform(np.array(patch_feature)).flatten() for patch_feature in patch_features]

    return patch_features, # torch.tensor(pca_patches_descriptors)# patch_features.squeeze().to("cuda:0")


def custom_detections(sam_detections, proposals_features, real_ref_features, file_path, scene_id=1, frame_id=1):
    # rgb_path = "cnos_analysis/crop_proposals/000001.png"
    # template_dir = "foundpose_analysis/icbin/templates/test_images_templates/obj_000001_original"

    # rgb = Image.open(rgb_path).convert("RGB")
    detections = Detections(sam_detections)
    mask_post_processing = SimpleNamespace(
        min_box_size=0.05,  # relative to image size
        min_mask_size=3e-4  # relative to image size
    )

    # get scores per proposal
    cosine_metric = PairwiseSimilarity()
    scores = cosine_metric(proposals_features, real_ref_features.unsqueeze(dim=0))
    score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
    score_per_proposal_and_object = torch.mean(
        score_per_proposal_and_object, dim=-1
    )
    # assign each proposal to the object with highest scores
    score_per_proposal, assigned_idx_object = torch.max(
        score_per_proposal_and_object, dim=-1
    )  # N_query

    idx_selected_proposals = torch.arange(
        len(score_per_proposal), device=score_per_proposal.device
    )[score_per_proposal > 0.5]
    pred_idx_objects = assigned_idx_object[idx_selected_proposals]
    pred_scores = score_per_proposal[idx_selected_proposals]


    # keep only detections with score > conf_threshold
    detections.filter(idx_selected_proposals)
    detections.add_attribute("scores", pred_scores)
    detections.add_attribute("object_ids", pred_idx_objects)
    detections.apply_nms_per_object_id(
        nms_thresh=0.3
    )
    detections.to_numpy()

    for ext in [".json", ".npz"]:
        file = file_path + ext
        if os.path.exists(file):
            os.remove(file)
            print(f"File {file} has been deleted.")
        else:
            print(f"File {file} does not exist.")

    # save detections to file
    results = detections.save_to_file(
        scene_id=int(scene_id),
        frame_id=int(frame_id),
        runtime=1,
        file_path=file_path,
        dataset_name="icbin",
        return_results=True,
    )

    num_workers = 10
    result_paths = [file_path+".npz"]
    pool = multiprocessing.Pool(processes=num_workers)
    convert_npz_to_json_with_idx = partial(
        convert_npz_to_json,
        list_npz_paths=result_paths,
    )
    detections = list(
        tqdm(
            pool.imap_unordered(
                convert_npz_to_json_with_idx, range(len(result_paths))
            ),
            total=len(result_paths),
            desc="Converting npz to json",
        )
    )
    formatted_detections = []
    for detection in tqdm(detections, desc="Loading results ..."):
        formatted_detections.extend(detection)

    detections_path = f"{file_path}.json"
    save_json_bop23(detections_path, formatted_detections)
    print(f"Saved predictions to {detections_path}")


def custom_detections_2(sam_detections, idx_selected_proposals, file_path, scene_id=1, frame_id=1):
    '''
    For classicication model (constrastive loss)
    '''

    # rgb = Image.open(rgb_path).convert("RGB")
    detections = Detections(sam_detections)
    mask_post_processing = SimpleNamespace(
        min_box_size=0.05,  # relative to image size
        min_mask_size=3e-4  # relative to image size
    )

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

    num_workers = 10
    result_paths = [file_path+".npz"]
    pool = multiprocessing.Pool(processes=num_workers)
    convert_npz_to_json_with_idx = partial(
        convert_npz_to_json,
        list_npz_paths=result_paths,
    )
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


def custom_visualize(input_file, dataset_name, rgb_path) -> None:
    if dataset_name in ["hb", "tless"]:
        split = "test_primesense"
    else:
        split = "test"
        
    num_max_objs = 50
    green = [0, 1, 0]  # RGB values for green
    colors = [green] * num_max_objs  # Create a list of green colors
    
    logging.info("Loading detections...")
    with open(input_file, 'r') as f:
        dets = json.load(f)
    
    conf_threshold = 0.5
    dets = [det for det in dets if det['score'] > conf_threshold] # keeps the det if it has score > threshold
    logging.info(f'Keeping only {len(dets)} detections having score > {conf_threshold}') # keeps the det if it has score > threshold
    
    
    # sort by (scene_id, frame_id)
    dets = sorted(dets, key=lambda x: (x['scene_id'], x['image_id']))
    list_scene_id_and_frame_id = [(det['scene_id'], det['image_id']) for det in dets]
    
    output_dir = "output_cnos_analysis_5" 
    os.makedirs(output_dir, exist_ok=True)
    counter = 0
    for scene_id, image_id in tqdm(list_scene_id_and_frame_id):
        img = Image.open(rgb_path)
        rgb = img.copy()
        img = np.array(img)
        masks, object_ids, scores = [], [], []
        for det in dets:
            if det['scene_id'] == scene_id and det['image_id'] == image_id:
                masks.append(rle_to_mask(det['segmentation']))
                object_ids.append(det['category_id']-1)  ## category_id is object_id
                scores.append(det['score'])
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


def _save_final_results(selected_proposals_indices, scene_id, frame_id, sam_detections, dataset, rgb_path, type = "contrastive"):
    # Cnos final results
    file_path = f"contrastive_learning/output_npz/{scene_id:06d}_{frame_id:06d}_{type}"
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
        saved_path = f"contrastive_learning/output_images/{scene_id:06d}_{frame_id:06d}_{type}.png"
        final_result.save(saved_path)
    return 0


def _tighten_bboxes(sam_detections, device="cuda"):
    # Apply morphological opening - it does eorion then dilation to remove noise
    kernel = np.ones((5,5), np.uint8)

    filtered_masks = list()
    for mask in sam_detections["masks"].cpu():
        filtered_mask = cv2.morphologyEx(np.array(mask, dtype="uint8"), cv2.MORPH_OPEN, kernel)
        filtered_masks.append(torch.tensor(filtered_mask))
    sam_detections["masks"] = torch.stack(filtered_masks).to(device)
    return sam_detections

def full_pipeline(custom_sam_model, rgb_path, scene_id, frame_id, best_model_path, obj_id=1, dataset = "icbin", aggreation_num_templates=5): 

    rgb = Image.open(rgb_path).convert("RGB")
    sam_detections = custom_sam_model.generate_masks(np.array(rgb))
    
    noise_remove_sam_detections = _tighten_bboxes(sam_detections)

    keep_ids = _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"])

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

    syn_data_type = "train_pbr" # test
    out_folder = f"foundpose_analysis/{dataset}/templates"
    # Load original templates when before putting through dinov2 we also apply transformation.
    syn_template_path_1 = f"{out_folder}/{syn_data_type}/obj_{obj_id:06d}_original" 
    syn_template_files_1 = sorted(glob.glob(os.path.join(syn_template_path_1, "*.png")), key=os.path.getmtime)
    syn_template_files = syn_template_files_1 # + syn_template_files_2
    syn_num_templates = len(syn_template_files)
    syn_templates = [np.array(Image.open(template_file).convert("RGB"))[:,:,:3]/255.0 for template_file in syn_template_files] 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_indices, max_indices, min_indices, average_prob, max_prob, min_prob, _, _ = check_similarity_3(best_model_path=best_model_path, masked_images=masked_images, templates=syn_templates, aggreation_num_templates = aggreation_num_templates, device=device)

    _save_final_results(selected_proposals_indices=average_indices, scene_id=scene_id, frame_id=frame_id, sam_detections=selected_sam_detections, dataset=dataset, rgb_path=rgb_path, type = "contrastive")
    return 0


def full_pipeline_non_filtered(custom_sam_model, rgb_path, scene_id, frame_id, best_model_path, obj_id=1, dataset = "icbin", aggreation_num_templates=5): 

    '''
    Do not remove any small detection or tighten the bbox
    '''
    rgb = Image.open(rgb_path).convert("RGB")
    sam_detections = custom_sam_model.generate_masks(np.array(rgb))
    
    # noise_remove_sam_detections = _tighten_bboxes(sam_detections)

    # keep_ids = _remove_very_small_detections(noise_remove_sam_detections["masks"], noise_remove_sam_detections["boxes"])

    # selected_masks = [noise_remove_sam_detections["masks"][i] for i in range(len(keep_ids)) if keep_ids[i]]
    # selected_bboxes = [noise_remove_sam_detections["boxes"][i] for i in range(len(keep_ids)) if keep_ids[i]]

    # selected_sam_detections = {
    #         "masks" : torch.stack(selected_masks),
    #         "boxes" : torch.stack(selected_bboxes)
    # }

    masked_images = []
    for mask in sam_detections["masks"].cpu():
        binary_mask = np.array(mask) * 255
        binary_mask = binary_mask.astype(np.uint8)
        masked_image = _extract_object_by_mask(rgb, binary_mask)
        masked_images.append(masked_image)

    syn_data_type = "train_pbr" # test
    out_folder = f"foundpose_analysis/{dataset}/templates"
    # Load original templates when before putting through dinov2 we also apply transformation.
    syn_template_path_1 = f"{out_folder}/{syn_data_type}_images_templates/obj_{obj_id:06d}_original" 
    syn_template_files_1 = sorted(glob.glob(os.path.join(syn_template_path_1, "*.png")), key=os.path.getmtime)
    syn_template_files = syn_template_files_1 # + syn_template_files_2
    syn_num_templates = len(syn_template_files)
    syn_templates = [np.array(Image.open(template_file).convert("RGB"))[:,:,:3]/255.0 for template_file in syn_template_files] 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_indices, average_scores, max_indices, min_indices, average_prob, max_prob, min_prob = check_similarity_3(best_model_path=best_model_path, masked_images=masked_images, templates=syn_templates, aggreation_num_templates = aggreation_num_templates, device=device)

    _save_final_results(selected_proposals_indices=average_indices, average_scores= average_scores, scene_id=scene_id, frame_id=frame_id, sam_detections=sam_detections, dataset=dataset, rgb_path=rgb_path, type = "contrastive")
    return 0
