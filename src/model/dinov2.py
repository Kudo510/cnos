import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from src.utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from torchvision.utils import make_grid, save_image
from src.model.utils import BatchedData
from copy import deepcopy

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class CustomDINOv2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model,
        token_name,
        image_size,
        chunk_size,
        descriptor_width_size,
        patch_size=14,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        # self.model.load_state_dict(torch.load("contrastive_learning/saved_checkpoints/icbin_best_model_checkpoint_obj1_correct_one.pth"))
        # self.model.load_state_dict(torch.load("contrastive_learning/saved_checkpoints/daoliuzhao_ver4_neg-weighted_pos-no-rotate_neg-heudi_cosine_loss_best_model_checkpoint.pth"))
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.rgb_proposal_processor_2 = CropResizePad(224) # founpose needs 420 not 224
        self.rgb_resize = CustomResizeLongestSide(
            descriptor_width_size, dividable_size=self.patch_size
        )
        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """
        1. Normalize image with DINOv2 transfom
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks) # masks shape 55, 480, 640]
        rgb = self.rgb_normalize(image_np).to(masks.device).float() # nomralize the image - 3,480, 640
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1) # 55, 3, 480, 640
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size]
        return processed_masked_rgbs

    def process_rgb_proposals_2(self, image_np, masks, boxes):
        """
        get the masks as well 
        """
        num_proposals = len(masks) # masks shape 55, 480, 640]
        rgb = self.rgb_normalize(image_np).to(masks.device).float() # nomralize the image - 3,480, 640
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1) # 55, 3, 480, 640
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs, processed_masks  = self.rgb_proposal_processor_2.process_images_masks(
            masked_rgbs, boxes, target_size_mask=16
        )  # [N, 3, target_size, target_size]
        return processed_masked_rgbs, processed_masks
    
    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                features = self.forward_by_chunk(images)
            else:
                # features = self.model.forward_one(images)
                features = self.model(images)
        else:  # get both features
            raise NotImplementedError
        return features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_features(
                batch_rgbs[idx_batch], token_name="x_norm_clstoken"
            )
            features.cat(feats)
        return features.data


    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        return self.forward_by_chunk(processed_rgbs)

    @torch.no_grad()
    def forward(self, image_np, proposals):
        return self.forward_cls_token(image_np, proposals)
    
    @torch.no_grad()
    def get_intermediate_layers(self, images):
        '''
        For Foundpose  to get patch features size of 30,30,1024 for templates/images
        check template- if it is thorugh rgb_normalized and with shape of 3,H,W 
        '''
        # patch_features = list()
        layers_list = list(range(24))
        num_images = images.shape[0]
        with torch.no_grad(): 
            patch_feature = self.model.get_intermediate_layers(
                    images, n=layers_list, return_class_token=True
                    )[18][0].reshape(num_images, -1,1024)
            # patch_features.append(patch_feature)
        return patch_feature # torch.cat(patch_features)
    
    def filter_out_invalid_templates(self, patch_features, masks):
        num_valid_patches = list() # List of numbers of valid patches for each template
        valid_patch_features = list()
        num_images = masks.shape[0]
        reshaped_masks = masks.reshape(num_images, -1)
        for patch_feature, mask in zip(patch_features, reshaped_masks):
            valid_patches = patch_feature[mask==1]
            valid_patch_features.append(valid_patches)
            num_valid_patches.append(valid_patches.shape[0]) # Append number of  valid patches for the template to the list
        # valid_patch_features = torch.cat(valid_patch_features)
        return num_valid_patches, valid_patch_features
    
    @torch.no_grad()
    def compute_patch_features(self, image_np, proposals):
        processed_rgbs, processed_masks = self.process_rgb_proposals_2(
            image_np, proposals.masks, proposals.boxes
        ) # processed_masks is tensor of num_proposal, H, W 

        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory
        patch_features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.get_intermediate_layers(
                batch_rgbs[idx_batch]
            )
            patch_features.cat(feats)
        patch_features = patch_features.data # patch_features (num_proposal, 900, 1024)

        # processed_masks is tensor of num_proposal, H, W 
        # patch_features (num_proposal, 900, 1024)
        num_valid_patches, valid_patch_features = self.filter_out_invalid_templates(patch_features, processed_masks)

        return num_valid_patches, valid_patch_features


