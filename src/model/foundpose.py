import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from torchvision.utils import make_grid, save_image

from copy import deepcopy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import faiss
import math
from src.model.loss import PairwiseSimilarity, Similarity
from torchvision.io import read_image

from src.utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from src.model.utils import BatchedData

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

    normalized_templates = [rgb_normalize(template/255.0).float() for template in templates]
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)
    print("normalized_templates shape", normalized_templates[0].shape)

    scaled_padded_templates = [resize_and_pad_image(normalized_template)
                            for normalized_template in normalized_templates] # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    print("scaled_padded_templates.shape", len(scaled_padded_templates), scaled_padded_templates[0].shape) 

    # Mask out the templates by clampping at 0,1 for resize image with size of (3, 30, 30)
    resized_templates = [resize_and_pad_image(torch.tensor(template).permute(2,0,1), target_max=30)
                         for template in templates]
    masks = [np.clip(np.sum(np.array(resized_template), axis=0), 0, 1, dtype="uint8") for resized_template in resized_templates]
    masks = torch.tensor(masks).reshape(num_templates, -1)
    
    plt.imshow(resized_templates[0].permute(1,2,0), cmap=plt.cm.gray)
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()

    plt.imshow(np.clip(np.sum(np.array(resized_templates[0]), axis=0), 0, 1), cmap=plt.cm.gray)
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

    num_valid_patches, valid_patch_features = filter_out_invalid_templates(patch_features, masks)

    # PCA
    pca = PCA(n_components=256)
    pca_patches_descriptors = pca.fit_transform(np.array(valid_patch_features))
    print("pca_crop_patches_descriptors.shape", pca_patches_descriptors.shape)

    return pca_patches_descriptors, num_valid_patches, patch_features


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
    normalized_crop_rgb = rgb_normalize(crop_rgb/255.0).float()
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)

    scaled_padded_crop_rgb = resize_and_pad_image(normalized_crop_rgb).unsqueeze(0) # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals

    # Mask out the crop by clampping at 0,1 for resize image with size of (3, 30, 30)
    resized_crop = resize_and_pad_image(torch.tensor(crop_rgb).permute(2,0,1), target_max=30)
    mask = np.clip(np.sum(np.array(resized_crop), axis=0), 0, 1, dtype="uint8").reshape(-1)

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
        feature_patches= dino_model.module.get_intermediate_layers(scaled_padded_crop_rgb.to(device), n=layers_list, return_class_token=True)[18][0].reshape(-1,1024)
    del dino_model

    num_valid_patches, valid_patch_features = filter_out_invalid_templates(feature_patches.unsqueeze(0), torch.tensor(mask).unsqueeze(0))

    # PCA
    pca = PCA(n_components=256, random_state=5)
    pca_crop_patches_descriptors = pca.fit_transform(np.array(valid_patch_features.cpu()))
    print(pca_crop_patches_descriptors.shape)

    return pca_crop_patches_descriptors, num_valid_patches, feature_patches


# def crop_feature_extraction(crop_rgb, dino_model, device):
#     rgb_normalize = T.Compose(
#         [
#             T.ToTensor(),
#             T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ]
#     )
#     inv_rgb_transform = T.Compose(
#         [
#             T.Normalize(
#                 mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
#             ),
#         ]
#     )
#     normalized_crop_rgb = rgb_normalize(crop_rgb/255.0).float()
#     # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)

#     scaled_padded_crop_rgb = resize_and_pad_image(normalized_crop_rgb).unsqueeze(0) # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals

#     # Mask out the crop by clampping at 0,1 for resize image with size of (3, 30, 30)
#     resized_crop = resize_and_pad_image(torch.tensor(crop_rgb).permute(2,0,1), target_max=30)
#     mask = np.clip(np.sum(np.array(resized_crop), axis=0), 0, 1, dtype="uint8").reshape(-1)

#     plt.imshow(resized_crop.permute(1,2,0))
#     plt.axis('off')  # Optional: Turn off the axis
#     plt.show()

#     # Display the image - 10* see lb the crop is normalized same way as the templates- ready to compare the similarity now
#     plt.imshow(np.clip(np.sum(np.array(resized_crop), axis=0), 0, 1), cmap=plt.cm.gray)
#     plt.axis('off')  # Optional: Turn off the axis
#     plt.show()

#     # Extract features from 18th layer of Dinov2 
#     layers_list = list(range(24))
#     torch.cuda.empty_cache()
#     with torch.no_grad(): 
#         feature_patches= dino_model.module.get_intermediate_layers(scaled_padded_crop_rgb.to(device), n=layers_list, return_class_token=True)[18][0].reshape(-1,1024)
#     del dino_model

#     num_valid_patches, valid_patch_features = filter_out_invalid_templates(feature_patches.unsqueeze(0), torch.tensor(mask).unsqueeze(0))

#     # PCA
#     pca = PCA(n_components=256, random_state=5)
#     pca_crop_patches_descriptors = pca.fit_transform(np.array(valid_patch_features.cpu()))
#     print(pca_crop_patches_descriptors.shape)

#     return pca_crop_patches_descriptors, num_valid_patches

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


def calculate_templates_vector(templates_labels, num_clusters=2048):
    # Calculate bag-of-words descriptors of the templates
    N = len(templates_labels)  # Number of templates
    
    # Calculate occurrences for all templates
    all_occurrences = np.array([np.bincount(template_label, minlength=num_clusters) for template_label in templates_labels])
    
    # Calculate document frequency (number of templates containing each word)
    doc_frequency = np.sum(all_occurrences > 0, axis=0)
    
    # Calculate IDF (add 1 to avoid division by zero)
    # idf = np.log(N / (doc_frequency + 1))
    idf = np.log(N / (doc_frequency + 1))
    
    # Calculate TF-IDF for each template
    templates_vector = []
    for t, occurrences in enumerate(all_occurrences):
        nt = len(templates_labels[t])
        tf = occurrences / nt
        tfidf = tf * idf
        templates_vector.append(tfidf)
    
    return templates_vector

# def calculate_templates_vector(templates_labels, num_clusters = 2048):
#     # Calculate bag-of-words descriptors of the templates

#     templates_vector = list()
#     all_occurrences = [np.bincount(templates_label, minlength=2048) for templates_label in templates_labels]
#     ni_array = np.sum(np.array(all_occurrences), axis = 0)
#     N = len(templates_labels) # Number of templates
#     for t in range(len(templates_labels)):
#         template_vector = list()
#         occurrences = np.bincount(templates_labels[t], minlength=2048)
#         for i in range(num_clusters):
#             n_it = occurrences[i]
#             nt = len(templates_labels[t])
#             ni = ni_array[i]
#             if ni==0 or nt==0:
#                 print(i)
#             bi = n_it / nt * math.log(N / ni)
#             template_vector.append(bi)
#         templates_vector.append(np.array(template_vector))
#     return templates_vector


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

    print("top 5 confidence scores", similar_scores)
    print("final average confidence score", score_per_detection)

    width = 50
    height = 50
    fig = plt.figure(figsize=(15, 15))
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

    return


# Code for first approach check
def _resize_and_pad_image(image, target_max=420):
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


# Code for first approach check - only use Dinov2 to extract features
def _templates_feature_extraction(templates, dino_model, num_templates, device):
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    normalized_templates = [rgb_normalize(template/255.0).float() for template in templates]
    # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)
    print("normalized_templates shape", normalized_templates[0].shape)

    scaled_padded_templates = [_resize_and_pad_image(normalized_template)
                            for normalized_template in normalized_templates] # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
    print("scaled_padded_templates.shape", len(scaled_padded_templates), scaled_padded_templates[0].shape) 

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
                )[18][1][:].reshape(size,-1,1024).cpu()
        patch_features.append(batch_feature.to('cpu'))
        del batch_feature
    patch_features = torch.cat(patch_features)
    del dino_model
    return patch_features


# Code for first approach check - only use Dinov2 to extract features
def _crop_feature_extraction(crop_rgb, dino_model, device):
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

    scaled_padded_crop_rgb = _resize_and_pad_image(normalized_crop_rgb).unsqueeze(0) # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals

    # Extract features from 18th layer of Dinov2 
    layers_list = list(range(24))
    torch.cuda.empty_cache()
    with torch.no_grad(): 
        feature_patches= dino_model.module.get_intermediate_layers(scaled_padded_crop_rgb.to(device), n=layers_list, return_class_token=True)[18][1].reshape(-1,1024)
    del dino_model

    return feature_patches