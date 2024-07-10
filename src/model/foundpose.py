import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from torchvision.utils import make_grid, save_image
from src.model.utils import BatchedData
from copy import deepcopy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import faiss
import math
from src.model.loss import PairwiseSimilarity, Similarity
from torchvision.io import read_image

from src.utils.bbox_utils import CropResizePad, CustomResizeLongestSide


class SmallDinov2(pl.LightningModule):
    def __init__(
        self,
        dinov2_vitl14=None,
        num_block=18,
    ):
        super().__init__()
        # Load the pre-trained model only if it's not provided
        if dinov2_vitl14 is None:
            dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        # Extract the layers
        
        self.num_block = num_block
        self.patch_embed = dinov2_vitl14.patch_embed
        self.blocks = nn.ModuleList(dinov2_vitl14.blocks[:num_block])
        self.norm = dinov2_vitl14.norm
        self.dinov2_vitl14 = dinov2_vitl14
        self.head = dinov2_vitl14.head

    @torch.no_grad()
    def forward_features(self, x, masks=None):
        x = self.dinov2_vitl14.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
        }
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return self.head(ret["x_norm_clstoken"])


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


def templates_feature_extraction(templates, dino_model, num_templates, device):
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    normalized_templates = [rgb_normalize(template).float() for template in templates]
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)
    print("normalized_templates shape", normalized_templates[0].shape)

    scaled_padded_templates = [resize_and_pad_image(normalized_template)
                            for normalized_template in normalized_templates] # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    print("scaled_padded_templates.shape", len(scaled_padded_templates), scaled_padded_templates[0].shape) 

    # Mask out the templates by clampping at 0,1 for resize image with size of (3, 30, 30)
    resized_templates = [resize_and_pad_image(torch.tensor(template).permute(2,0,1), target_max=30)
                         for template in templates]
    masks = [np.clip(np.sum(np.array(resized_template), axis=0), 0, 1) for resized_template in resized_templates]
    masks = torch.tensor(masks).reshape(num_templates, -1)

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
                )[17][0].reshape(size,-1,1024).cpu()
        patch_features.append(batch_feature.to('cpu'))
        del batch_feature
    patch_features = torch.cat(patch_features)
    del dino_model

    num_valid_patches, valid_patch_features = filter_out_invalid_templates(patch_features, masks)

    # PCA
    pca = PCA(n_components=256)
    pca_patches_descriptors = pca.fit_transform(np.array(valid_patch_features))
    print("pca_crop_patches_descriptors.shape", pca_patches_descriptors.shape)

    return pca_patches_descriptors, num_valid_patches


def crop_feature_extraction(crop_rgb, dino_model, device):
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
    normalized_crop_rgb = rgb_normalize(crop_rgb).float()
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)

    scaled_padded_crop_rgb = resize_and_pad_image(normalized_crop_rgb).unsqueeze(0) # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    print("scaled_padded_crop_rgb.shape", scaled_padded_crop_rgb.shape) 
    plt.imshow(torch.tensor(scaled_padded_crop_rgb).squeeze().permute(1,2,0))
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    # Mask out the crop by clampping at 0,1 for resize image with size of (3, 30, 30)
    resized_crop = resize_and_pad_image(torch.tensor(crop_rgb).permute(2,0,1), target_max=30)
    mask = np.clip(np.sum(np.array(resized_crop), axis=0), 0, 1).reshape(-1)

    # Extract features from 18th layer of Dinov2 
    layers_list = list(range(24))
    torch.cuda.empty_cache()
    with torch.no_grad(): 
        feature_patches= dino_model.module.get_intermediate_layers(scaled_padded_crop_rgb.to(device), n=layers_list, return_class_token=True)[17][0].reshape(-1,1024)
    del dino_model

    num_valid_patches, valid_patch_features = filter_out_invalid_templates(feature_patches, torch.tensor(mask))

    # PCA
    pca = PCA(n_components=256)
    pca_crop_patches_descriptors = pca.fit_transform(np.array(feature_patches.cpu()))
    print(pca_crop_patches_descriptors.shape)

    top3_pca = pca_crop_patches_descriptors.reshape(30,30,-1)[:,:,:3]
    # normalized_image = ((top3_pca - np.min(top3_pca)) / (np.max(top3_pca) - np.min(top3_pca))* 255).astype(np.uint8)
    normalized_image = ((top3_pca - np.min(top3_pca)) / (np.max(top3_pca) - np.min(top3_pca)))
    plt.imshow(torch.tensor(normalized_image))
    plt.axis('off')  
    plt.show()

    return pca_crop_patches_descriptors, num_valid_patches


def kmeans(pca_patches_descriptors, ncentroids = 2048, niter = 20, verbose = True):
    # https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization

    d = pca_patches_descriptors.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
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


def calculate_crop_vector(crop_labels, templates_labels, num_clusters = 2048):
    # For word_i, term frequency = occurences of word_i within the crop / number of occurences of word_i in all templates). 
    # Calculate bag-of-words descriptors of the templates
    all_occurrences_crop = np.bincount(crop_labels, minlength=2048)

    all_occurrences_templates = [np.bincount(templates_label, minlength=2048) for templates_label in templates_labels]
    ni_array = np.sum(np.array(all_occurrences_templates), axis = 0)
    N = len(templates_labels) # Number of templates = 642 

    crop_vector = list()
    for i in range(num_clusters):
        n_it = all_occurrences_crop[i]
        nt = crop_labels.shape[0] # Number of words in crop = 400 
        ni = ni_array[i]
        bi = n_it / nt * math.log(N / ni)
        crop_vector.append(bi)
    return torch.tensor(crop_vector).view(1,-1) # Goal having features size (1,2048)


def calculate_similarity(crop_rgb, feature_decriptors, ref_features, metric, dataset, synthetic=False):
    # get scores per proposal
    scores = metric(feature_decriptors[:, None, :], ref_features[None, :, :]) # should get  # N_proposals x N_objects x N_templates -get only 1,42 as num_prosals*num_templates instead
    score_per_detection, similar_template_indices = torch.topk(scores, k=5, dim=-1) # get top 5 most similar templates
    # get the final confidence score
    score_per_detection = torch.mean(
        score_per_detection, dim=-1
    ) 
    # Check the confidence scores for the similar templates
    similar_scores = scores[:, similar_template_indices[0].to("cpu")]

    similar_templates = []
    for i in range(len(similar_template_indices[0])):
        if synthetic:
            img = read_image(f"foundpose_analysis/{dataset}/templates/synthetic_images_templates/{dataset}/train_pbr/obj_{obj_id:06d}_original/{(similar_template_indices[0][i]):06d}.png")            
        # else:
        #     img = read_image(f"cnos_analysis/real_images_templates/icbin/obj_000001_original/{(similar_template_indices[0][i]):06d}.png")        
        similar_templates.append(img)
    template_images = torch.stack(similar_templates)

    # Display the crop
    plt.imshow(crop_rgb)
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    print("top 5 confidence scores", similar_scores)
    print("final average confidence score", score_per_detection)

    width = 50
    height = 50
    fig = plt.figure(figsize=(15, 15))
    columns = 3
    rows = 2

    for index in range(len(template_images)):
        fig.add_subplot(rows, columns, index + 1)
        img = template_images[index].permute(1, 2, 0)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Top Template {index + 1}')

    plt.tight_layout()
    plt.show()

    return
