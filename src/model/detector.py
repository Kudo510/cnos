import torch
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import os
import os.path as osp
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from src.utils.inout import save_json, load_json, save_json_bop23
from src.model.utils import BatchedData, Detections, convert_npz_to_json
from hydra.utils import instantiate
import time
import glob
from functools import partial
import multiprocessing
from src.model.foundpose import resize_and_pad_image
from sklearn.decomposition import PCA
import faiss
import os
from copy import deepcopy
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src_eloftr.utils.plotting import make_matching_figure
from src_eloftr.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from PIL import Image
from src_eloftr.custom_utils import extract_correspondences
# from src.model.utils import get_pixel_counts
from src.model.foundpose import calculate_templates_labels, calculate_templates_vector, calculate_crop_vector
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json

# def save_selected_images(images_tensor, selected_indices, save_path='combined_images.png', scores=None):
#     """
#     Save selected images from a batch tensor into a single horizontal image with optional scores.
    
#     Args:
#         images_tensor (torch.Tensor): Tensor of shape (B, C, H, W) containing RGB images
#         selected_indices (list): List of indices to select from the batch
#         scores (list or torch.Tensor, optional): Scores corresponding to the selected indices
#         save_path (str): Path where to save the combined image
#     """
#     # Select the specified images
#     selected_images = images_tensor[selected_indices]
    
#     # Ensure values are in valid range [0, 1]
#     if selected_images.max() > 1.0:
#         selected_images = selected_images / 255.0
    
#     # Make grid of images (here we use nrow=5 to arrange horizontally)
#     grid = vutils.make_grid(selected_images, nrow=5, padding=2, normalize=False)
    
#     # Convert to numpy and transpose for plotting
#     grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
#     # Create figure with appropriate size
#     plt.figure(figsize=(20, 4))
#     plt.imshow(grid_np)
    
#     # Add scores if provided
#     if scores is not None:
#         n_images = len(selected_indices)
        
#         # Calculate the position for each score
#         for idx in range(n_images):
#             # Get grid position
#             row = idx // 5
#             col = idx % 5
            
#             # Calculate x position (center of each image)
#             x_spacing = grid_np.shape[1] / 5
#             x_pos = (col + 0.5) * x_spacing
            
#             # Calculate y position (top of each image)
#             y_spacing = grid_np.shape[0]
#             y_pos = y_spacing * 0.05  # Place text at 5% from top
            
#             # Add score text
#             if isinstance(scores, torch.Tensor):
#                 score = scores[idx].item()
#             else:
#                 score = scores[idx]
#             plt.text(x_pos, y_pos, f'{score:.3f}', 
#                     color='white', fontsize=12, fontweight='bold',
#                     horizontalalignment='center',
#                     bbox=dict(facecolor='black', alpha=0.7, pad=2))
    
#     plt.axis('off')
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

def save_selected_images(images_tensor, selected_indices, save_path='combined_images.png', crop=None, scores=None):
    """
    Save selected images from a batch tensor into a single horizontal image with optional scores,
    with a crop image displayed above.
    
    Args:
        images_tensor (torch.Tensor): Tensor of shape (B, C, H, W) containing RGB images
        selected_indices (list): List of indices to select from the batch
        save_path (str): Path where to save the combined image
        scores (list or torch.Tensor, optional): Scores corresponding to the selected indices
        crop (torch.Tensor, optional): Single crop image to display above templates
    """
    # Select the specified images
    selected_images = images_tensor[selected_indices]
    
    # Ensure values are in valid range [0, 1]
    if selected_images.max() > 1.0:
        selected_images = selected_images / 255.0
    if crop is not None and crop.max() > 1.0:
        crop = crop / 255.0
    
    # Create figure with appropriate size (taller to accommodate crop)
    plt.figure(figsize=(20, 8))
    
    # Create a subplot layout: crop on top, templates below
    if crop is not None:
        plt.subplot2grid((2, 1), (0, 0))
        # Scale up the crop image size using interpolate
        upscaled_crop = F.interpolate(crop.unsqueeze(0), scale_factor=5, mode='bilinear', align_corners=False).squeeze(0)
        crop_grid = vutils.make_grid([upscaled_crop], nrow=1, padding=2, normalize=False)
        plt.imshow(crop_grid.cpu().numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.title('Query Crop', pad=10)
        
        plt.subplot2grid((2, 1), (1, 0))
    # Make grid of template images
    grid = vutils.make_grid(selected_images, nrow=5, padding=2, normalize=False)
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(grid_np)
    
    # Add scores if provided
    if scores is not None:
        n_images = len(selected_indices)
        
        # Calculate the position for each score
        for idx in range(n_images):
            # Get grid position
            row = idx // 5
            col = idx % 5
            
            # Calculate x position (center of each image)
            x_spacing = grid_np.shape[1] / 5
            x_pos = (col + 0.5) * x_spacing
            
            # Calculate y position (top of each image)
            y_spacing = grid_np.shape[0]
            y_pos = y_spacing * 0.05  # Place text at 5% from top
            
            # Add score text
            if isinstance(scores, torch.Tensor):
                score = scores[idx].item()
            else:
                score = scores[idx]
            plt.text(x_pos, y_pos, f'{score:.3f}', 
                    color='white', fontsize=12, fontweight='bold',
                    horizontalalignment='center',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)

class CNOS(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir, # './datasets/bop23_challenge/results/cnos_exps'
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model
        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir

        # loead eloftr matcher
        model_type = 'full' # 'full' for best quality, 'opt' for best efficiency
        precision = 'fp16' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).
        _default_cfg = deepcopy(full_default_cfg)
        _default_cfg['half'] = True
            
        # print(_default_cfg)
        # matcher = LoFTR(config=_default_cfg)
        # matcher.load_state_dict(torch.load("src_eloftr/weights/eloftr_outdoor.ckpt")['state_dict'])
        # matcher = reparameter(matcher) # no reparameterization will lead to low performance
        # matcher = matcher.half()
        # self.matcher = matcher.eval().cuda()
        self.pca = PCA(n_components=256, random_state=5) # 128 or 256 is just the same


        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info(f"Init CNOS done!")

    def set_reference_objects(self): ## to calculate/return features of all templates from dinov2
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data = {"descriptors": BatchedData(None)}
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        if self.onboarding_config.rendering_type == "pbr": # not our case
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors # reset_descriptors = False
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device)
        else: # our case - so computer ref features here
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                # import pdb; pdb.set_trace()
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                )
                self.ref_data["descriptors"].append(ref_feats)

            ## self.ref_data["descriptors"] is stack of ref_features of all templates
            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size # 
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            # torch.save(self.ref_data["descriptors"], descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}"
        )

    def templates_patch_feature_extraction(self):
        '''
        Extract pach templates in Founpose for templates
        '''

        self.ref_patch_data = {"patch_descriptors": BatchedData(None)}
        for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing patch descriptor for FoundPose ...",
            ):
                # import pdb; pdb.set_trace()
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_patch_feats = self.descriptor_model.get_intermediate_layers(ref_imgs)
                self.ref_patch_data["patch_descriptors"].append(ref_patch_feats)

        self.ref_patch_data["patch_descriptors"].stack()  # N_objects x descriptor_size # 
        self.ref_patch_data["patch_descriptors"] = self.ref_patch_data["patch_descriptors"].data
    
    def filter_out_invalid_templates_old(self):
        self.templates_patch_feature_extraction()

        num_valid_patches = list() # List of numbers of valid patches for each template
        valid_patch_features = list()
        resized_masks = list()
        # self.ref_dataset) len = num_objs here = 1 - each has shape of 42, 224,224
        
        for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="resizing masks for FoundPose ...",
            ):
            num_template = self.ref_dataset[idx]["templates_masks"].shape[0]
            mask = self.ref_dataset[idx]["templates_masks"].reshape(num_template, -1)  #  self.ref_dataset[idx]["templates_masks"] has shape of 42,224,224
            resized_masks.append(mask)
        
        for patch_feature, mask in zip(self.ref_patch_data["patch_descriptors"][0], resized_masks[0]): # 0 here measn we have only 1 object- or we are doing it jsut for first object only 
            valid_patches = patch_feature[mask==1]
            valid_patch_features.append(valid_patches)
            num_valid_patches.append(valid_patches.shape[0]) # Append number of  valid patches for the template to the list
        valid_patch_features = torch.cat(valid_patch_features)

        self.ref_patch_data["templates_num_valid_patches"] = num_valid_patches
        self.ref_patch_data["templates_valid_patch_features"] = valid_patch_features


    def filter_out_invalid_templates(self):
        self.templates_patch_feature_extraction()

        num_valid_patches_list = list() # List of numbers of valid patches for each template
        valid_patch_features_list = list()
        resized_masks = list()
        # self.ref_dataset) len = num_objs here = 1 - each has shape of 42, 224,224
        
        for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="resizing masks for FoundPose ...",
            ):
            num_template = self.ref_dataset[idx]["templates_masks"].shape[0]
            mask = self.ref_dataset[idx]["templates_masks"].reshape(num_template, -1)  #  self.ref_dataset[idx]["templates_masks"] has shape of 42,224,224
            resized_masks.append(mask)

        for idx in range(len(resized_masks)):
            num_valid_patches = list() # List of numbers of valid patches for each template
            valid_patch_features = list()
            for patch_feature, mask in zip(self.ref_patch_data["patch_descriptors"][idx], resized_masks[idx]): # 0 here measn we have only 1 object- or we are doing it jsut for first object only 
                valid_patches = patch_feature[mask==1]
                valid_patch_features.append(valid_patches)
                num_valid_patches.append(valid_patches.shape[0]) # Append number of  valid patches for the template to the list
            valid_patch_features = torch.cat(valid_patch_features)
            num_valid_patches_list.append(num_valid_patches)
            valid_patch_features_list.append(valid_patch_features)

        self.ref_patch_data["templates_num_valid_patches"] = num_valid_patches_list
        self.ref_patch_data["templates_valid_patch_features"] = valid_patch_features_list


    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(self.segmentor_model, "predictor"):
            self.segmentor_model.predictor.model = (
                self.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

    def find_matched_proposals(self, proposal_decriptors): 
        '''
        Compare with the threshold of 0.5 and return the index of sam proposals that has score > 0.5, also return the obj_id(which object the proposal is) for this sam proposal and also the average top 5 score for this proposals
        '''

        # best_template_indices = None
        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_decriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":      ## our case
            best_template_scores, best_template_indices = torch.topk(scores, k=5, dim=-1)
            score_per_proposal_and_object = torch.mean( # N_proposals x N_objects
                best_template_scores, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_proposals 

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        best_template_scores = best_template_scores[idx_selected_proposals]
        best_template_indices = best_template_indices[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]
        return idx_selected_proposals, pred_idx_objects, pred_scores , best_template_scores, best_template_indices
        '''
        idx_selected_proposals : index of the proposals that has score > 0.5
        pred_idx_objects : object that the proposals shows
        pred_scores : final similarity score
        '''

    def find_matched_proposals_5(self, scores):
        '''
        given the scores- get the top5 averages- also returns the template indices'''

        best_template_scores, best_template_indices = torch.topk(scores, k=5, dim=-1)
        score_per_proposal_and_object = torch.mean( # N_proposals x N_objects
                best_template_scores, dim=-1
            )

        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_proposals 

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        best_template_scores = best_template_scores[idx_selected_proposals]
        best_template_indices = best_template_indices[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]
        return idx_selected_proposals, pred_idx_objects, pred_scores , best_template_scores, best_template_indices
    
    def find_matched_proposals_3(self, proposal_decriptors): 
        '''
        get cnos scores        
        '''

        # best_template_indices = None
        # compute matching scores for each proposals
        cnos_matching_scores = self.matching_config.metric(
            proposal_decriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates

        return cnos_matching_scores
    
    def find_matched_proposals_4(self, proposal_decriptors, templates_descriptors): 
        '''
        Compare with the threshold of 0.5 and return the index of sam proposals that has score > 0.5, also return the obj_id(which object the proposal is) for this sam proposal and also the average top 5 score for this proposals
        '''

        # compute matching scores for each proposals
        bow_matching_scores = self.matching_config.metric(
            proposal_decriptors, templates_descriptors)
        return bow_matching_scores

    def find_matched_proposals_2(self, proposal_decriptors, templates_descriptors): 
        '''
        Compare with the threshold of 0.5 and return the index of sam proposals that has score > 0.5, also return the obj_id(which object the proposal is) for this sam proposal and also the average top 5 score for this proposals
        '''

        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_decriptors, templates_descriptors
        )  # N_proposals x N_objects x N_templates
        # if self.matching_config.aggregation_function == "mean":
        #     score_per_proposal_and_object = (
        #         torch.sum(scores, dim=-1) / scores.shape[-1]
        #     )  # N_proposals x N_objects
        # elif self.matching_config.aggregation_function == "median":
        #     score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        # elif self.matching_config.aggregation_function == "max":
        #     score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        # elif self.matching_config.aggregation_function == "avg_5":      ## our case
        
        best_scores, best_indices = torch.topk(scores, k=7, dim=-1)
        score_per_proposal_and_object = torch.mean( # N_proposals x N_objects
            best_scores, dim=-1
        )
        
        # else:
        #     raise NotImplementedError

        # score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_proposals 

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )# [score_per_proposal > self.matching_config.confidence_thresh]
        # pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]
        return best_indices[0][0], pred_scores, best_scores[0][0]
    

    def kmeans_clustering(self, pca_patches_descriptors, ncentroids = 2048, niter = 20, verbose = True):
    # https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization

        d = pca_patches_descriptors.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True, seed=5)
        kmeans.train(pca_patches_descriptors)
        return kmeans

    def test_step(self, batch, idx): # idx = 0
        '''
        batch is a dict of 'image', 'scene_id', 'frame_id'
        batch["image"] with shape of batch_size, 3, H,W 
            here: [1, 3, 480, 640])
            also 10* batch_size must be 1
        '''
        print("Path of input image:", batch["rgb_path"][0])
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin
                ),
                exist_ok=True,
            )
            onboarding_time = time.time()
            print("onboarding_time start", onboarding_time)
            self.set_reference_objects()
            self.move_to_device()
            self.filter_out_invalid_templates()
            onboarding_time = time.time()
            print("onboarding_time end", onboarding_time)

            self.target_bop = json.load(open(f"datasets/bop23_challenge/datasets/{self.dataset_name}/test_targets_bop19.json", "r"))
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        # scene_id = int(batch["rgb_path"][0].split("/")[5])
        # im_id = int(batch["rgb_path"][0].split("/")[-1].split(".")[0])
        # file_path = f"datasets/bop23_challenge/results/cnos_exps/predictions/{self.dataset_name}/CustomSamAutomaticMaskGenerator_template_pbr1_aggavg_5_{self.dataset_name}/scene{scene_id:06d}_frame{im_id}.npz"
        # if os.path.exists(file_path):
        #     print("The file exists.")
        #     return 0
        # else:
        #     print("The file does not exist.")


        # obj_id = None
        # for target in self.target_bop:
        #     if target["scene_id"] == scene_id and target["im_id"] == im_id:
        #         obj_id = target["obj_id"]
        #         break
        # if obj_id is None:
        #     return 0
        
        proposal_stage_start_time = time.time()

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image

        # run propoals

        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')
        print(f"Number of sam proposals: {proposals['masks'].shape}")

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        print(f"Number of sam proposals: -rechecked {len(detections)}")
        detections.remove_very_small_detections(

            config=self.post_processing_config.mask_post_processing
        )
        print(f"Number of detections after removing all small proposals: {len(detections)}")
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the 224,224

        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2

        (
            idx_cnos_selected_proposals,
            pred_cnos_idx_objects,
            pred_cnos_scores,
            best_cnos_template_scores,
            best_cnos_template_indices,
        ) = self.find_matched_proposals(query_decriptors)

        # # update detections)
        detections.add_attribute("scores", pred_cnos_scores.cuda())
        # detections.filter(idx_selected_proposals)
        # print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")
        # pred_idx_objects = torch.tensor([1]).repeat(len(idx_selected_proposals)) # temperary class 1 for object 1 only
        # # detections.add_attribute("scores",(pred_scores+1)/2)
        # detections.add_attribute("object_ids", pred_idx_objects)
        # print(f"Number of chosen detections before applying nms: {len(detections)}")
        # # breakpoint()
        # keep_idx = detections.apply_nms_per_object_id(
        #     nms_thresh= self.post_processing_config.nms_thresh
        # )
        # print(f"Number of chosen detections after applying nms: {len(detections)}")



        # Building BoW

        _ , valid_crop_feature_patches = self.descriptor_model.compute_patch_features(image_np, detections) # valid_crop_feature_patches is a list of num_proposals, each is the valid patches
        proposal_stage_end_time = time.time()

        matching_stage_start_time = time.time()
        all_valid_patch_features = [torch.cat((valid_crop_feature_patch, self.ref_patch_data["templates_valid_patch_features"][pred_cnos_idx_objects[i].cpu().item()]), dim=0) for i, valid_crop_feature_patch in enumerate(valid_crop_feature_patches)]

        # crops = self.descriptor_model.process_rgb_proposals_3(image_np, detections.masks, detections.boxes)
        # templates = self.ref_dataset[0]["unnormalized_templates"]
    
        bow_scores = list()
        # final_bow_scores = list()
        # proposal_stage_start_time = time.time()
        for i in range(len(all_valid_patch_features)):
            # display/save the crops here:
            pca_crop_patches_descriptors = self.pca.fit_transform(np.array(all_valid_patch_features[i].cpu()))

            pca_crop = pca_crop_patches_descriptors[:valid_crop_feature_patches[i].shape[0]]
            pca_templates = pca_crop_patches_descriptors[valid_crop_feature_patches[i].shape[0]:]

            ## Change here from 2048 to 1024
            num_clusters = 512 # 2048
            kmeans = self.kmeans_clustering(pca_templates, ncentroids = num_clusters, niter = 20, verbose = True)
            templates_labels = calculate_templates_labels(self.ref_patch_data["templates_num_valid_patches"][pred_cnos_idx_objects[i].cpu().item()], kmeans, pca_templates)
            templates_vector = calculate_templates_vector(templates_labels = templates_labels, num_clusters = num_clusters)

            # Assign labels to the data points
            crop_labels = kmeans.index.search(pca_crop, 1)[1].reshape(-1)
            
            crop_vector = calculate_crop_vector(crop_labels = crop_labels, templates_labels = templates_labels, num_clusters = num_clusters)
            concat_templates_vector = torch.from_numpy(np.stack(templates_vector, axis=0)).float()  # Will be shape [42, 1024]

            (
                _, # best_templates_indices,
                pred_bow_scores,
                _, #scores
            ) = self.find_matched_proposals_2(crop_vector, concat_templates_vector.unsqueeze(0))

            bow_scores.append(pred_bow_scores)

            # final_crop_vector = torch.cat((query_decriptors[i].unsqueeze(0), crop_vector.cuda()), dim=-1)
            # final_template_vector = torch.cat((self.ref_data["descriptors"], concat_templates_vector.unsqueeze(0).cuda()), dim=-1)

            # (
            #     _,
            #     final_pred_bow_scores,
            # ) = self.find_matched_proposals_2(torch.cat((query_decriptors[i].unsqueeze(0), crop_vector.cuda()), dim=-1), torch.cat((self.ref_data["descriptors"], concat_templates_vector.unsqueeze(0).cuda()), dim=-1))
            # final_bow_scores.append(final_pred_bow_scores)

            # if pred_bow_scores > 0.2:
            #     saved_folder_path = f"foundpose_analysis_2/{self.dataset_name}"
            #     if not os.path.exists(saved_folder_path):
            #         os.makedirs(saved_folder_path)  # It's makedirs, not mkdirs

            #     # Fix save paths and parameters
            #     save_selected_images(templates, best_templates_indices, 
            #                         save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_best_templates.png",
            #                         # crop=crops[i], 
            #                         scores=scores)

            #     save_selected_images(templates, best_cnos_template_indices[i][0].cpu(),                            save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_cnos_best_templates.png",
            #                         # crop=crops[i], 
            #                         scores=best_cnos_template_scores[i][0].cpu())
            #     save_selected_images(crops, [i], save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_crop.png")
        
        pred_final_scores = (pred_cnos_scores.cpu() + torch.tensor(bow_scores))/2 #torch.tensor(bow_scores) # 
        # pred_final_scores = torch.tensor(final_bow_scores)
        # pred_final_scores = pred_cnos_scores.cpu()
        # proposal_stage_end_time = time.time()
        # print(f"Calculating time per test image {proposal_stage_end_time-proposal_stage_start_time}")
        
        selected_proposals_indices = [i for i, a_s in enumerate(pred_final_scores) if a_s >0.01]
        # selected_proposals_scores = [a_s for i, a_s in enumerate(bow_scores) if a_s >0.01]
        # matching descriptors
        # matching_stage_start_time = time.time()

        # update detections)
        detections.scores = pred_final_scores.cuda()
        detections.add_attribute("object_ids", (pred_cnos_idx_objects+1))

        detections.filter(selected_proposals_indices)
        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")
        # model_path = glob.glob("datasets/bop23_challenge/datasets/"+ self.dataset_name+"/models/obj_*.ply")
        # obj_id = int(model_path[0].split("obj_")[-1].split(".")[0])
        # pred_idx_objects = torch.tensor([obj_id]).repeat(len(selected_proposals_indices)) # temperary class 1 for object 1 only
        # detections.add_attribute("scores",(pred_scores+1)/2)
        # detections.add_attribute("object_ids", (pred_cnos_idx_objects+1))
        print(f"Number of chosen detections before applying nms: {len(detections)}")
        # breakpoint()
        detections.apply_nms_per_object_id(
            nms_thresh= self.post_processing_config.nms_thresh
        )

        matching_stage_end_time = time.time()

        del pred_final_scores
        torch.cuda.empty_cache()

        print(f"Number of chosen detections after applying nms: {len(detections)}")

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        ) 
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0


    def test_step_ori_cnos_bow(self, batch, idx): # idx = 0
        '''
        batch is a dict of 'image', 'scene_id', 'frame_id'
        batch["image"] with shape of batch_size, 3, H,W 
            here: [1, 3, 480, 640])
            also 10* batch_size must be 1
        '''
        print("Path of input image:", batch["rgb_path"][0])
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin
                ),
                exist_ok=True,
            )
            onboarding_time = time.time()
            print("onboarding_time start", onboarding_time)
            self.set_reference_objects()
            self.move_to_device()
            self.filter_out_invalid_templates()
            onboarding_time = time.time()
            print("onboarding_time end", onboarding_time)
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        proposal_stage_start_time = time.time()

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image

        # run propoals

        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')
        print(f"Number of sam proposals: {proposals['masks'].shape}")

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        print(f"Number of sam proposals: -rechecked {len(detections)}")
        detections.remove_very_small_detections(

            config=self.post_processing_config.mask_post_processing
        )
        print(f"Number of detections after removing all small proposals: {len(detections)}")
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the 224,224

        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2

        (
            idx_cnos_selected_proposals,
            pred_cnos_idx_objects,
            pred_cnos_scores,
            best_cnos_template_scores,
            best_cnos_template_indices,
        ) = self.find_matched_proposals(query_decriptors)

        # # update detections)
        detections.add_attribute("scores", pred_cnos_scores.cuda())
        # detections.filter(idx_selected_proposals)
        # print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")
        # pred_idx_objects = torch.tensor([1]).repeat(len(idx_selected_proposals)) # temperary class 1 for object 1 only
        # # detections.add_attribute("scores",(pred_scores+1)/2)
        # detections.add_attribute("object_ids", pred_idx_objects)
        # print(f"Number of chosen detections before applying nms: {len(detections)}")
        # # breakpoint()
        # keep_idx = detections.apply_nms_per_object_id(
        #     nms_thresh= self.post_processing_config.nms_thresh
        # )
        # print(f"Number of chosen detections after applying nms: {len(detections)}")



        # Building BoW

        _ , valid_crop_feature_patches = self.descriptor_model.compute_patch_features(image_np, detections) # valid_crop_feature_patches is a list of num_proposals, each is the valid patches
        proposal_stage_end_time = time.time()

        matching_stage_start_time = time.time()
        all_valid_patch_features = [torch.cat((valid_crop_feature_patch, self.ref_patch_data["templates_valid_patch_features"]), dim=0) for valid_crop_feature_patch in valid_crop_feature_patches]

        # crops = self.descriptor_model.process_rgb_proposals_3(image_np, detections.masks, detections.boxes)
        # templates = self.ref_dataset[0]["unnormalized_templates"]
    
        bow_scores = list()
        # final_bow_scores = list()
        # proposal_stage_start_time = time.time()
        for i in range(0, len(all_valid_patch_features)):
            # display/save the crops here:
            pca_crop_patches_descriptors = self.pca.fit_transform(np.array(all_valid_patch_features[i].cpu()))

            pca_crop = pca_crop_patches_descriptors[:valid_crop_feature_patches[i].shape[0]]
            pca_templates = pca_crop_patches_descriptors[valid_crop_feature_patches[i].shape[0]:]

            ## Change here from 2048 to 1024
            num_clusters = 256 # 2048
            kmeans = self.kmeans_clustering(pca_templates, ncentroids = num_clusters, niter = 20, verbose = True)
            templates_labels = calculate_templates_labels(self.ref_patch_data["templates_num_valid_patches"], kmeans, pca_templates)
            templates_vector = calculate_templates_vector(templates_labels = templates_labels, num_clusters = num_clusters)

            # Assign labels to the data points
            crop_labels = kmeans.index.search(pca_crop, 1)[1].reshape(-1)
            
            crop_vector = calculate_crop_vector(crop_labels = crop_labels, templates_labels = templates_labels, num_clusters = num_clusters)
            concat_templates_vector = torch.from_numpy(np.stack(templates_vector, axis=0)).float()  # Will be shape [42, 1024]

            (
                _, # best_templates_indices,
                pred_bow_scores,
                _, #scores
            ) = self.find_matched_proposals_2(crop_vector, concat_templates_vector.unsqueeze(0))

            bow_scores.append(pred_bow_scores)

            # final_crop_vector = torch.cat((query_decriptors[i].unsqueeze(0), crop_vector.cuda()), dim=-1)
            # final_template_vector = torch.cat((self.ref_data["descriptors"], concat_templates_vector.unsqueeze(0).cuda()), dim=-1)

            # (
            #     _,
            #     final_pred_bow_scores,
            # ) = self.find_matched_proposals_2(torch.cat((query_decriptors[i].unsqueeze(0), crop_vector.cuda()), dim=-1), torch.cat((self.ref_data["descriptors"], concat_templates_vector.unsqueeze(0).cuda()), dim=-1))
            # final_bow_scores.append(final_pred_bow_scores)

            # if pred_bow_scores > 0.2:
            #     saved_folder_path = f"foundpose_analysis_2/{self.dataset_name}"
            #     if not os.path.exists(saved_folder_path):
            #         os.makedirs(saved_folder_path)  # It's makedirs, not mkdirs

            #     # Fix save paths and parameters
            #     save_selected_images(templates, best_templates_indices, 
            #                         save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_best_templates.png",
            #                         # crop=crops[i], 
            #                         scores=scores)

            #     save_selected_images(templates, best_cnos_template_indices[i][0].cpu(),                            save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_cnos_best_templates.png",
            #                         # crop=crops[i], 
            #                         scores=best_cnos_template_scores[i][0].cpu())
            #     save_selected_images(crops, [i], save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_crop.png")
        
        pred_final_scores = (pred_cnos_scores.cpu() + torch.tensor(bow_scores))/2 #torch.tensor(bow_scores) # 
        # pred_final_scores = torch.tensor(final_bow_scores)
        # pred_final_scores = pred_cnos_scores.cpu()
        # proposal_stage_end_time = time.time()
        # print(f"Calculating time per test image {proposal_stage_end_time-proposal_stage_start_time}")
        
        selected_proposals_indices = [i for i, a_s in enumerate(pred_final_scores) if a_s >0.01]
        # selected_proposals_scores = [a_s for i, a_s in enumerate(bow_scores) if a_s >0.01]
        # matching descriptors
        # matching_stage_start_time = time.time()

        # update detections)
        detections.scores = pred_final_scores.cuda()
        detections.filter(selected_proposals_indices)
        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")
        model_path = glob.glob("datasets/bop23_challenge/datasets/"+ self.dataset_name+"/models/obj_*.ply")
        obj_id = int(model_path[0].split("obj_")[-1].split(".")[0])
        pred_idx_objects = torch.tensor([obj_id]).repeat(len(selected_proposals_indices)) # temperary class 1 for object 1 only
        # detections.add_attribute("scores",(pred_scores+1)/2)
        detections.add_attribute("object_ids", pred_idx_objects)
        print(f"Number of chosen detections before applying nms: {len(detections)}")
        # breakpoint()
        detections.apply_nms_per_object_id(
            nms_thresh= self.post_processing_config.nms_thresh
        )

        matching_stage_end_time = time.time()

        del pred_final_scores
        torch.cuda.empty_cache()

        print(f"Number of chosen detections after applying nms: {len(detections)}")

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        ) 
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0


    def test_step_cnos_bows_newidea(self, batch, idx): # idx = 0
        '''
        new idea for each crop we calculate the sum of cosine simlarity
        '''
        print("Path of input image:", batch["rgb_path"][0])
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()
            self.filter_out_invalid_templates()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image

        # run propoals
        
        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')
        print(f"Number of sam proposals: {proposals['masks'].shape}")

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        print(f"Number of sam proposals: -rechecked {len(detections)}")
        detections.remove_very_small_detections(

            config=self.post_processing_config.mask_post_processing
        )
        print(f"Number of detections after removing all small proposals: {len(detections)}")
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the 224,224

        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2

        cnos_matching_scores = self.find_matched_proposals_3(query_decriptors) # .squeeze(1)

        # # update detections)
        # detections.add_attribute("scores", pred_cnos_scores.cuda())
        # detections.filter(idx_selected_proposals)
        # print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")
        # pred_idx_objects = torch.tensor([1]).repeat(len(idx_selected_proposals)) # temperary class 1 for object 1 only
        # # detections.add_attribute("scores",(pred_scores+1)/2)
        # detections.add_attribute("object_ids", pred_idx_objects)
        # print(f"Number of chosen detections before applying nms: {len(detections)}")
        # # breakpoint()
        # keep_idx = detections.apply_nms_per_object_id(
        #     nms_thresh= self.post_processing_config.nms_thresh
        # )
        # print(f"Number of chosen detections after applying nms: {len(detections)}")



        # Building BoW

        _ , valid_crop_feature_patches = self.descriptor_model.compute_patch_features(image_np, detections) # valid_crop_feature_patches is a list of num_proposals, each is the valid patches
        
        all_valid_patch_features = [torch.cat((valid_crop_feature_patch, self.ref_patch_data["templates_valid_patch_features"]), dim=0) for valid_crop_feature_patch in valid_crop_feature_patches]

        # crops = self.descriptor_model.process_rgb_proposals_3(image_np, detections.masks, detections.boxes)
        # templates = self.ref_dataset[0]["unnormalized_templates"]
    
        final_matching_scores = list()
        # final_bow_scores = list()
        proposal_stage_start_time = time.time()
        for i in range(0, len(all_valid_patch_features)):
            # display/save the crops here:
            pca_crop_patches_descriptors = self.pca.fit_transform(np.array(all_valid_patch_features[i].cpu()))

            pca_crop = pca_crop_patches_descriptors[:valid_crop_feature_patches[i].shape[0]]
            pca_templates = pca_crop_patches_descriptors[valid_crop_feature_patches[i].shape[0]:]

            ## Change here from 2048 to 1024
            num_clusters = 256 # 2048
            kmeans = self.kmeans_clustering(pca_templates, ncentroids = num_clusters, niter = 20, verbose = True)
            templates_labels = calculate_templates_labels(self.ref_patch_data["templates_num_valid_patches"], kmeans, pca_templates)
            templates_vector = calculate_templates_vector(templates_labels = templates_labels, num_clusters = num_clusters)

            # Assign labels to the data points
            crop_labels = kmeans.index.search(pca_crop, 1)[1].reshape(-1)
            
            crop_vector = calculate_crop_vector(crop_labels = crop_labels, templates_labels = templates_labels, num_clusters = num_clusters)
            concat_templates_vector = torch.from_numpy(np.stack(templates_vector, axis=0)).float()  # Will be shape [42, 1024]


            bow_matching_scores = self.find_matched_proposals_4(crop_vector, concat_templates_vector.unsqueeze(0)).squeeze(0)
            final_matching_score = (bow_matching_scores.to("cuda:0")+cnos_matching_scores[i])/2

            final_matching_scores.append(final_matching_score)

            # final_crop_vector = torch.cat((query_decriptors[i].unsqueeze(0), crop_vector.cuda()), dim=-1)
            # final_template_vector = torch.cat((self.ref_data["descriptors"], concat_templates_vector.unsqueeze(0).cuda()), dim=-1)

            # (
            #     _,
            #     final_pred_bow_scores,
            # ) = self.find_matched_proposals_2(torch.cat((query_decriptors[i].unsqueeze(0), crop_vector.cuda()), dim=-1), torch.cat((self.ref_data["descriptors"], concat_templates_vector.unsqueeze(0).cuda()), dim=-1))
            # final_bow_scores.append(final_pred_bow_scores)

            # if pred_bow_scores > 0.2:
            #     saved_folder_path = f"foundpose_analysis_2/{self.dataset_name}"
            #     if not os.path.exists(saved_folder_path):
            #         os.makedirs(saved_folder_path)  # It's makedirs, not mkdirs

            #     # Fix save paths and parameters
            #     save_selected_images(templates, best_templates_indices, 
            #                         save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_best_templates.png",
            #                         # crop=crops[i], 
            #                         scores=scores)

            #     save_selected_images(templates, best_cnos_template_indices[i][0].cpu(),                            save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_cnos_best_templates.png",
            #                         # crop=crops[i], 
            #                         scores=best_cnos_template_scores[i][0].cpu())
            #     save_selected_images(crops, [i], save_path=f"foundpose_analysis_2/{self.dataset_name}/{i}_crop.png")
        
        final_matching_scores = torch.stack(final_matching_scores) # (pred_cnos_scores.cpu() + torch.tensor(bow_scores))/2 #torch.tensor(bow_scores) # 
        # pred_final_scores = torch.tensor(final_bow_scores)
        # pred_final_scores = pred_cnos_scores.cpu()

        idx_selected_proposals, pred_idx_objects, pred_scores , best_template_scores, best_template_indices = self.find_matched_proposals_5(final_matching_scores)
        
        proposal_stage_end_time = time.time()
        print(f"Calculating time per test image {proposal_stage_end_time-proposal_stage_start_time}")
        
        selected_proposals_indices = [i for i, a_s in enumerate(pred_scores) if a_s >-1]
        # selected_proposals_scores = [a_s for i, a_s in enumerate(bow_scores) if a_s >0.01]
        # matching descriptors
        matching_stage_start_time = time.time()

        # update detections)
        detections.scores = pred_scores
        detections.filter(selected_proposals_indices)
        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")
        model_path = glob.glob("datasets/bop23_challenge/datasets/"+ self.dataset_name+"/models/obj_*.ply")
        obj_id = int(model_path[0].split("obj_")[-1].split(".")[0])
        pred_idx_objects = torch.tensor([obj_id]).repeat(len(selected_proposals_indices)) # temperary class 1 for object 1 only
        # detections.add_attribute("scores",(pred_scores+1)/2)
        detections.add_attribute("object_ids", pred_idx_objects)
        print(f"Number of chosen detections before applying nms: {len(detections)}")
        # breakpoint()
        detections.apply_nms_per_object_id(
            nms_thresh= self.post_processing_config.nms_thresh
        )
        detections.scores = pred_scores.cpu()

        matching_stage_end_time = time.time()

        del pred_scores
        torch.cuda.empty_cache()

        print(f"Number of chosen detections after applying nms: {len(detections)}")

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        ) 
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def test_step_cnos_bow_new_method_2(self, batch, idx): # idx = 0
        '''
        just clcualte the patch embedding for the best templates from CNOS
        '''
        print("Path of input image:", batch["rgb_path"][0])
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()
            self.filter_out_invalid_templates()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image

        # run propoals
        
        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')
        print(f"Number of sam proposals: {proposals['masks'].shape}")

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        print(f"Number of sam proposals: -rechecked {len(detections)}")
        detections.remove_very_small_detections(

            config=self.post_processing_config.mask_post_processing
        )
        print(f"Number of detections after removing all small proposals: {len(detections)}")
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the 224,224

        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2

        cnos_matching_scores = self.find_matched_proposals_3(query_decriptors) # .squeeze(1)

        # # update detections)
        # detections.add_attribute("scores", pred_cnos_scores.cuda())
        # detections.filter(idx_selected_proposals)
        # print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")
        # pred_idx_objects = torch.tensor([1]).repeat(len(idx_selected_proposals)) # temperary class 1 for object 1 only
        # # detections.add_attribute("scores",(pred_scores+1)/2)
        # detections.add_attribute("object_ids", pred_idx_objects)
        # print(f"Number of chosen detections before applying nms: {len(detections)}")
        # # breakpoint()
        # keep_idx = detections.apply_nms_per_object_id(
        #     nms_thresh= self.post_processing_config.nms_thresh
        # )
        # print(f"Number of chosen detections after applying nms: {len(detections)}")



        # Building BoW

        _ , valid_crop_feature_patches = self.descriptor_model.compute_patch_features(image_np, detections) # valid_crop_feature_patches is a list of num_proposals, each is the valid patches
        
        all_valid_patch_features = [torch.cat((valid_crop_feature_patch, self.ref_patch_data["templates_valid_patch_features"]), dim=0) for valid_crop_feature_patch in valid_crop_feature_patches]

        # crops = self.descriptor_model.process_rgb_proposals_3(image_np, detections.masks, detections.boxes)
        # templates = self.ref_dataset[0]["unnormalized_templates"]
    
        final_matching_scores = list()
        # final_bow_scores = list()
        proposal_stage_start_time = time.time()
        for i in range(0,len(all_valid_patch_features)):
            # display/save the crops here:
            pca_crop_patches_descriptors = self.pca.fit_transform(np.array(all_valid_patch_features[i].cpu()))

            pca_crop = pca_crop_patches_descriptors[:valid_crop_feature_patches[i].shape[0]]
            pca_templates = pca_crop_patches_descriptors[valid_crop_feature_patches[i].shape[0]:]

            ## Change here from 2048 to 1024
            num_clusters = 256 # 2048
            kmeans = self.kmeans_clustering(pca_templates, ncentroids = num_clusters, niter = 20, verbose = True)
            templates_labels = calculate_templates_labels(self.ref_patch_data["templates_num_valid_patches"], kmeans, pca_templates)
            templates_vector = calculate_templates_vector(templates_labels = templates_labels, num_clusters = num_clusters)

            # Assign labels to the data points
            crop_labels = kmeans.index.search(pca_crop, 1)[1].reshape(-1)
            
            crop_vector = calculate_crop_vector(crop_labels = crop_labels, templates_labels = templates_labels, num_clusters = num_clusters)
            concat_templates_vector = torch.from_numpy(np.stack(templates_vector, axis=0)).float()  # Will be shape [42, 1024]


            bow_matching_scores = self.find_matched_proposals_4(crop_vector, concat_templates_vector.unsqueeze(0)).squeeze(0)

            top5_values, _ = torch.topk(cnos_matching_scores[i].cpu(), 5, dim=1)
            top5_cnos_average = top5_values.mean(dim=1)  
            argmax_index = torch.argmax(cnos_matching_scores[i].cpu(), dim=1)
            bow_score_max = bow_matching_scores[0][argmax_index]

            final_score =(bow_score_max + top5_cnos_average)/2
            # final_matching_score = (bow_matching_scores.to("cuda:0")+cnos_matching_scores[i])/2

            final_matching_scores.append(final_score)
        
        final_matching_scores = torch.tensor(final_matching_scores) # (pred_cnos_scores.cpu() + torch.tensor(bow_scores))/2 #torch.tensor(bow_scores) # 
        # pred_final_scores = torch.tensor(final_bow_scores)
        # pred_final_scores = pred_cnos_scores.cpu()

        # idx_selected_proposals, pred_idx_objects, pred_scores , best_template_scores, best_template_indices = self.find_matched_proposals_5(final_matching_scores)
        
        proposal_stage_end_time = time.time()
        print(f"Calculating time per test image {proposal_stage_end_time-proposal_stage_start_time}")
        
        selected_proposals_indices = [i for i, a_s in enumerate(final_matching_scores) if a_s >-1]
        # selected_proposals_scores = [a_s for i, a_s in enumerate(bow_scores) if a_s >0.01]
        # matching descriptors
        matching_stage_start_time = time.time()

        # update detections)
        detections.filter(selected_proposals_indices)
        detections.scores = final_matching_scores.to("cuda:0")

        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")
        model_path = glob.glob("datasets/bop23_challenge/datasets/"+ self.dataset_name+"/models/obj_*.ply")
        obj_id = int(model_path[0].split("obj_")[-1].split(".")[0])
        pred_idx_objects = torch.tensor([obj_id]).repeat(len(selected_proposals_indices)) # temperary class 1 for object 1 only
        # detections.add_attribute("scores",(pred_scores+1)/2)
        detections.add_attribute("object_ids", pred_idx_objects)
        print(f"Number of chosen detections before applying nms: {len(detections)}")
        # breakpoint()
        detections.apply_nms_per_object_id(
            nms_thresh= self.post_processing_config.nms_thresh
        )
        detections.scores = final_matching_scores

        matching_stage_end_time = time.time()

        del final_matching_scores
        torch.cuda.empty_cache()

        print(f"Number of chosen detections after applying nms: {len(detections)}")

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        ) 
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0
    
    def test_epoch_end(self, outputs):
        if self.global_rank == 0:  # only rank 0 process
            # can use self.all_gather to gather results from all processes
            # but it is simpler just load the results from files so no file is missing
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )
            num_workers = 10
            logging.info(f"Converting npz to json requires {num_workers} workers ...")
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

            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions to {detections_path}")   
   

    def test_step_eloftr_1(self, batch, idx): # idx = 0
        '''
        batch is a dict of 'image', 'scene_id', 'frame_id'
        batch["image"] with shape of batch_size, 3, H,W 
            here: [1, 3, 480, 640])
            also 10* batch_size must be 1
        '''
        print("Path of input image:", batch["rgb_path"][0])
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')
        print(f"Number of sam proposals: {proposals['masks'].shape}")

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        print(f"Number of sam proposals: -rechecked {len(detections)}")
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        print(f"Number of detections after removing all small proposals: {len(detections)}")
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the 224,224
        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
            best_template_scores,
            best_template_indices
        ) = self.find_matched_proposals(query_decriptors)
        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")

        
        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        # detections.add_attribute("scores",(pred_scores+1)/2)
        detections.add_attribute("object_ids", pred_idx_objects)
        print(f"Number of chosen detections before applying nms: {len(detections)}")
        # breakpoint()

        #apply eloftr
        eloftr_time_start = time.time()
        num_corres_list = list()
        crops = self.descriptor_model.process_rgb_proposals_3(image_np, detections.masks, detections.boxes)
        templates = self.ref_dataset[0]["unnormalized_templates"]
        # crops_valid_pixel_list = get_pixel_counts(crops)
        # templates_valid_pixel_list = get_pixel_counts(templates)

        best_template_weights = torch.zeros(len(crops), best_template_indices.shape[-1]) # # weight is actually the number of correspodnces in the templates
        for i, crop in enumerate(crops):
            best_template_weights[i,:] = torch.tensor([extract_correspondences(crop, templates[idx].cuda(), matcher=self.matcher, precision='fp16', model_type='full') for idx in best_template_indices[i][0]])
        eloftr_time_end = time.time()
        print("eloftr processing time: ", eloftr_time_end-eloftr_time_start)
        # for temp in  tqdm(self.ref_dataset[0]["unnormalized_templates"], desc="Running Eloftr"):
        #     num_corres_temp_list = np.array(
        #         [extract_correspondences(crop, temp.cuda(), matcher=self.matcher, precision='fp16', model_type = 'full') for crop in crops]
        #                          )
        #     num_corres_list.append(num_corres_temp_list)
        
        new_pred_scores = torch.sum(best_template_scores.squeeze(1)*best_template_weights.cuda()/torch.sum(best_template_weights, dim=1, keepdim=True).cuda(), dim=-1)
        # new_pred_scores = (torch.tensor(num_corres_list).cuda())/(max(num_corres_list)+1)
        assert len(new_pred_scores)==len(detections)
        detections.scores = new_pred_scores

        # input for eloftr - 1, 1, H,W as gray images- the value also need to dvide by 255.0 - as 

        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        print(f"Number of chosen detections after applying nms: {len(detections)}")


        matching_stage_end_time = time.time()
        print("time per image: ", matching_stage_end_time - matching_stage_start_time)
        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0


    def test_step_original_old_version(self, batch, idx): # idx = 0
        '''
        This one changes object ide code- specifically for only 1 scene
        batch is a dict of 'image', 'scene_id', 'frame_id'
        batch["image"] with shape of batch_size, 3, H,W 
            here: [1, 3, 480, 640])
            also 10* batch_size must be 1
        '''

        print("Path of input image:", batch["rgb_path"][0])
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin
                ),
                exist_ok=True,
            )
            onboarding_time = time.time()
            print("onboarding_time start", onboarding_time)
            self.set_reference_objects()
            onboarding_time = time.time()
            print("onboarding_time end", onboarding_time)
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image
        


        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')
        print(f"Number of sam proposals: {proposals['masks'].shape}")

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        print(f"Number of sam proposals: -rechecked {len(detections)}")
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        print(f"Number of detections after removing all small proposals: {len(detections)}")
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the 224,224
        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
            _,
            _
        ) = self.find_matched_proposals(query_decriptors)

        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")

        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        # detections.add_attribute("scores",(pred_scores+1)/2)
        model_path = glob.glob("datasets/bop23_challenge/datasets/"+ self.dataset_name+"/models/obj_*.ply")
        obj_id = int(model_path[0].split("obj_")[-1].split(".")[0])
        pred_idx_objects = torch.tensor([obj_id]).repeat(len(pred_idx_objects))

        detections.add_attribute("object_ids", pred_idx_objects)
        print(f"Number of chosen detections before applying nms: {len(detections)}")
        # breakpoint()
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )

        matching_stage_end_time = time.time()

        print(f"Number of chosen detections after applying nms: {len(detections)}")

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0
    

    def test_step_ori(self, batch, idx): # idx = 0
        '''
        batch is a dict of 'image', 'scene_id', 'frame_id'
        batch["image"] with shape of batch_size, 3, H,W 
            here: [1, 3, 480, 640])
            also 10* batch_size must be 1
        '''

        print("Path of input image:", batch["rgb_path"][0])
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin
                ),
                exist_ok=True,
            )
            onboarding_time = time.time()
            print("onboarding_time start", onboarding_time)
            self.set_reference_objects()
            onboarding_time = time.time()
            print("onboarding_time end", onboarding_time)
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image
        


        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')
        print(f"Number of sam proposals: {proposals['masks'].shape}")

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        print(f"Number of sam proposals: -rechecked {len(detections)}")
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        print(f"Number of detections after removing all small proposals: {len(detections)}")
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the 224,224
        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
            _,
            _
        ) = self.find_matched_proposals(query_decriptors)

        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")

        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        # detections.add_attribute("scores",(pred_scores+1)/2)
        # model_path = glob.glob("datasets/bop23_challenge/datasets/"+ self.dataset_name+"/models/obj_*.ply")
        # obj_id = int(model_path[0].split("obj_")[-1].split(".")[0])
        # pred_idx_objects = torch.tensor([obj_id]).repeat(len(pred_idx_objects))

        detections.add_attribute("object_ids", pred_idx_objects+1)
        print(f"Number of chosen detections before applying nms: {len(detections)}")
        # breakpoint()
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )

        matching_stage_end_time = time.time()

        print(f"Number of chosen detections after applying nms: {len(detections)}")

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def test_step_eloftr_ver1(self, batch, idx): # idx = 0
        '''
        batch is a dict of 'image', 'scene_id', 'frame_id'
        batch["image"] with shape of batch_size, 3, H,W 
            here: [1, 3, 480, 640])
            also 10* batch_size must be 1
        '''
        print("Path of input image:", batch["rgb_path"][0])
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')
        print(f"Number of sam proposals: {proposals['masks'].shape}")

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        print(f"Number of sam proposals: -rechecked {len(detections)}")
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        print(f"Number of detections after removing all small proposals: {len(detections)}")
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the 224,224
        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
            _,
            _
        ) = self.find_matched_proposals(query_decriptors)
        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")

        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        # detections.add_attribute("scores",(pred_scores+1)/2)
        detections.add_attribute("object_ids", pred_idx_objects)
        print(f"Number of chosen detections before applying nms: {len(detections)}")
        # breakpoint()

        #apply eloftr
        num_corres_list = list()
        crops = self.descriptor_model.process_rgb_proposals_3(image_np, detections.masks, detections.boxes)
        # templates = self.ref_dataset[0]["unnormalized_templates"]
        # crops_valid_pixel_list = get_pixel_counts(crops)
        # templates_valid_pixel_list = get_pixel_counts(self.ref_dataset[0]["unnormalized_templates"])

        # for crop in crops:
        #     num_corres_temp_list = np.array(
        #         [extract_correspondences(crop, temp.cuda(), matcher=self.matcher, precision='fp16', model_type = 'full') for temp in self.ref_dataset[0]["unnormalized_templates"]]
        #                                      )
        #     top5_average = sum(sorted(num_corres_temp_list)[-5:]) / 5
        #     print(f"max num correpodences: {max(num_corres_temp_list)}")
        #     print(f"top 5 average num correpodences: {top5_average}")
        #     num_corres_list.append(top5_average)

        for temp_id, temp in  tqdm(enumerate(self.ref_dataset[0]["unnormalized_templates"]), desc="Running Eloftr"):
            num_corres_temp_list = np.array(
                [extract_correspondences(crop, temp.cuda(), matcher=self.matcher, precision='fp16', model_type = 'full')/(crops_valid_pixel_list[i]+templates_valid_pixel_list[temp_id]) for i, crop in enumerate(crops)]
                                 )
            num_corres_list.append(num_corres_temp_list)
        
        num_corres_list = np.stack(num_corres_list)
        top5_averages = np.mean(np.sort(num_corres_list, axis=0)[-5:, :], axis=0)  # Shape: (101,)    
        new_pred_scores = torch.tensor(top5_averages).cuda() / (np.max(top5_averages) + 1)  # dm should not + 1 cos it will be low. actually it is no problem at all

        # new_pred_scores = (torch.tensor(num_corres_list).cuda())/(max(num_corres_list)+1)
        assert len(new_pred_scores)==len(detections)
        detections.scores = new_pred_scores

        # input for eloftr - 1, 1, H,W as gray images- the value also need to dvide by 255.0 - as 

        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        print(f"Number of chosen detections after applying nms: {len(detections)}")


        matching_stage_end_time = time.time()
        print("time per image: ", matching_stage_end_time - matching_stage_start_time)
        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0


    def custom_test_step(self, batch, idx): # just remove 0 zB at batch["image"][0]  # idx = 0
        '''
        batch is a dict of 'image', 'scene_id', 'frame_id'
        batch["image"] with shape of batch_size, 3, H,W 
            here: [1, 3, 480, 640])
            also 10* batch_size must be 1
        '''
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}", #self.name_prediction_file: CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin

                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()

        image_np = (
            self.inv_rgb_transform(batch["image"]) # just to get the image in the batch - return shape of 3, 480, 640
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        ) # 480, 640, 3
        image_np = np.uint8(image_np.clip(0, 1) * 255) # turn back as normal image

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np) # a dict of 'masks', 'boxes')

        # init detections with masks and boxes
        detections = Detections(proposals) # just turn the dict to a class thoi- still keys as masks, boxes
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        # compute descriptors
        # Use image_np, to conver the bboxes as well as the masks to size of the input 
        query_decriptors = self.descriptor_model(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
        ) = self.find_matched_proposals(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"]
        frame_id = batch["frame_id"]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def custom_test_epoch_end(self):
        if self.global_rank == 0:  # only rank 0 process
            # can use self.all_gather to gather results from all processes
            # but it is simpler just load the results from files so no file is missing
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )
            num_workers = 10
            logging.info(f"Converting npz to json requires {num_workers} workers ...")
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

            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions to {detections_path}")