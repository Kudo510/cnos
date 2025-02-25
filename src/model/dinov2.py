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
import torch
import numpy as np
from src_eloftr.utils.plotting import make_matching_figure
from src_eloftr.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
import torch.nn as nn


descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

def get_dino_model(img_size):
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = CustomDINOv2_2(model_name="dinov2_vitl14",
                         model=dinov2_vitl14,
                         token_name="x_norm_clstoken",
                         image_size=img_size,
                         chunk_size=16,
                         descriptor_width_size=640
                        )
    return model

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels=1024, out_channels=1):
        super(LinearClassifier, self).__init__()
        self.classifier = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, (1,1)),
                torch.nn.Sigmoid()
        )


    def forward(self, embeddings):
        embeddings = embeddings[0].reshape(-1, 16, 16, 1024)
        embeddings = embeddings.permute(0,3,1,2)
        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(nn.Module):
  def __init__(self):
    super().__init__()
    self.img_size = 420
    self.dinov2 = get_dino_model(self.img_size)
    self.classifier = LinearClassifier()

  def forward(self, x):
    # use frozen features
    # x = torch.stack(x)

    _, H, W, _ = x.shape
    patch_embeddings = self.dinov2.get_intermediate_layers(x),
    # convert to logits and upsample to the size of the pixel values
    # logits = self.classifier(patch_embeddings)
    # logits = torch.nn.functional.interpolate(logits, size=self.img_size, mode="bilinear", align_corners=False)
    logits = None

    return patch_embeddings, logits


class CustomDINOv2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        # finetuned_dinov2,
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
        # self.finetuned_dinov2 = finetuned_dinov2 # model
        # self.finetuned_dinov2.load_state_dict(torch.load('datasets/bop23_challenge/pretrained/dinov2_finetned_ver_275.pth'))
        # self.model.load_state_dict(torch.load("contrastive_learning/saved_checkpoints/icbin_best_model_checkpoint_obj1_correct_one.pth"))
        # self.model.load_state_dict(torch.load("contrastive_learning/saved_checkpoints/daoliuzhao_ver4_neg-weighted_pos-no-rotate_neg-heudi_cosine_loss_best_model_checkpoint.pth"))
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.template_proposal_size = 224 # siize for template size
        self.proposals_size = 420 # size for input size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
        self.rgb_normalize_2 = T.Compose(
            [
                T.ToTensor()            
            ]
        )
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.template_proposal_size) # this for templates
        self.rgb_proposal_processor_2 = CropResizePad(self.proposals_size) # founpose needs 420 not 224
        # self.rgb_resize = CustomResizeLongestSide(
        #     descriptor_width_size, dividable_size=self.patch_size
        # )
        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposals_size} done!"
        )

        # Loading eloftr
        _default_cfg = deepcopy(full_default_cfg)
        _default_cfg['half'] = True
        self.matcher = LoFTR(config=_default_cfg)
        self.matcher.load_state_dict(torch.load("src_eloftr/weights/eloftr_outdoor.ckpt")['state_dict'])
        self.matcher = reparameter(self.matcher) # no reparameterization will lead to low performance
        self.matcher = self.matcher.half()
        self.matcher = self.matcher.eval().cuda()

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
    
    def process_rgb_proposals_3(self, image_np, masks, boxes):
        """
        same as process_rgb_proposals just that the rgbs are not normalized just converted to tensor
        remmber output images is divided by 255.0 already
        """
        num_proposals = len(masks) # masks shape 55, 480, 640]
        rgb = self.rgb_normalize_2(image_np).to(masks.device).float() # nomralize the image - 3,480, 640
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1) # 55, 3, 480, 640
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size] but value is divided by 255.0 - so need to covnert back to original images
        return processed_masked_rgbs #*255.0.clamp(0,255).to(torch.unit8)
    

    def process_rgb_proposals_2(self, image_np, masks, boxes):
        """
        get the masks as well 
        """
        num_proposals = len(masks) # masks shape 55, 480, 640]
        rgb = self.rgb_normalize(image_np).to(masks.device).float() # nomralize the image - 3,480, 640
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1) # 55, 3, 480, 640
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs, processed_masks  = self.rgb_proposal_processor_2.process_images_masks(
            masked_rgbs, boxes, target_size_mask=int(self.proposals_size/14)
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
        selected_layer = 18
        # selected_layer_1 = 23
        selected_layer_2 = 18
        with torch.no_grad(): 
            patch_feature = self.model.get_intermediate_layers(
                    images, n=layers_list, return_class_token=True
                    )[selected_layer][0].reshape(num_images, -1,1024)
            
            # # For finetuned dinov2
            # patch_feature, _ = self.finetuned_dinov2(
            #         images
            #         )
            # patch_feature = patch_feature[0]
            
            # concatenated_patch_features = torch.cat((
            #     patch_feature[selected_layer_1][0].reshape(num_images, -1,1024), 
            #     patch_feature[selected_layer_2][0].reshape(num_images, -1,1024)), dim=-1)
            
            # concatenated_patch_features = torch.cat((concatenated_patch_features, patch_feature[selected_layer][0].reshape(num_images, -1,1024)), dim=-1)

            # cnos_feature = self.model.get_intermediate_layers(
            #         images, n=layers_list, return_class_token=True
            #         )[selected_layer_2][1].reshape(num_images, -1,1024).repeat(1,patch_feature.shape[1],1)
            # concatenated_patch_features = torch.cat((patch_feature, cnos_feature), dim=-1)
        
        # Using eloftr
        # Convert to gray scale first
        
        # weights = torch.tensor([0.299, 0.587, 0.114]).view(1,3,1,1).to("cuda")
        # gray_images = torch.sum(images*weights, dim=1, keepdim=True)
        # batch = {'image0': gray_images, 'image1': gray_images[:1]}
        # with torch.no_grad(): 
        #     patch_feature = self.matcher.custom_forward(batch) # (batch size, 256, 784)

        return patch_feature # concatenated_patch_features # patch_feature
    
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
    



class CustomDINOv2_2_2(pl.LightningModule):
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
        self.template_proposal_size = 420 # size for template size
        self.proposals_size = 420  # size for input size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
        self.rgb_normalize_2 = T.Compose(
            [
                T.ToTensor()            
            ]
        )
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.template_proposal_size) # this for templates
        self.rgb_proposal_processor_2 = CropResizePad(self.proposals_size) # founpose needs 420 not 224
        # self.rgb_resize = CustomResizeLongestSide(
        #     descriptor_width_size, dividable_size=self.patch_size
        # )
        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposals_size} done!"
        )

        # Loading eloftr
        # _default_cfg = deepcopy(full_default_cfg)
        # _default_cfg['half'] = True
        # self.matcher = LoFTR(config=_default_cfg)
        # self.matcher.load_state_dict(torch.load("src_eloftr/weights/eloftr_outdoor.ckpt")['state_dict'])
        # self.matcher = reparameter(self.matcher) # no reparameterization will lead to low performance
        # self.matcher = self.matcher.half()
        # self.matcher = self.matcher.eval().cuda()

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
    
    def process_rgb_proposals_3(self, image_np, masks, boxes):
        """
        same as process_rgb_proposals just that the rgbs are not normalized just converted to tensor
        remmber output images is divided by 255.0 already
        """
        num_proposals = len(masks) # masks shape 55, 480, 640]
        rgb = self.rgb_normalize_2(image_np).to(masks.device).float() # nomralize the image - 3,480, 640
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1) # 55, 3, 480, 640
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size] but value is divided by 255.0 - so need to covnert back to original images
        return processed_masked_rgbs #*255.0.clamp(0,255).to(torch.unit8)
    

    def process_rgb_proposals_2(self, image_np, masks, boxes):
        """
        get the masks as well 
        """
        num_proposals = len(masks) # masks shape 55, 480, 640]
        rgb = self.rgb_normalize(image_np).to(masks.device).float() # nomralize the image - 3,480, 640
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1) # 55, 3, 480, 640
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs, processed_masks  = self.rgb_proposal_processor_2.process_images_masks(
            masked_rgbs, boxes, target_size_mask=int(self.proposals_size/14)
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
        selected_layer = 18
        selected_layer_1 = 13
        selected_layer_2 = 23
        with torch.no_grad(): 
            patch_feature = self.model.get_intermediate_layers(
                    images, n=layers_list, return_class_token=True
                    )[selected_layer][0].reshape(num_images, -1,1024)
            
            # For finetuned dinov2
            # patch_feature = self.finetuned_dinov2(
            #         images
            #         )
            
            # concatenated_patch_features = torch.cat((
            #     patch_feature[selected_layer_1][0].reshape(num_images, -1,1024), 
            #     patch_feature[selected_layer_2][0].reshape(num_images, -1,1024)), dim=-1)
            # concatenated_patch_features = torch.cat((concatenated_patch_features, patch_feature[selected_layer][0].reshape(num_images, -1,1024)), dim=-1)
            # cnos_feature = self.model.get_intermediate_layers(
            #         images, n=layers_list, return_class_token=True
            #         )[selected_layer][1].reshape(num_images, -1,1024).repeat(1,concatenated_patch_features.shape[1],1)
            # concatenated_patch_features = torch.cat((concatenated_patch_features, cnos_feature), dim=-1)
        
        # Using eloftr
        # Convert to gray scale first
        
        # weights = torch.tensor([0.299, 0.587, 0.114]).view(1,3,1,1).to("cuda")
        # gray_images = torch.sum(images*weights, dim=1, keepdim=True)
        # batch = {'image0': gray_images, 'image1': gray_images[:1]}
        # with torch.no_grad(): 
        #     patch_feature = self.matcher.custom_forward(batch) # (batch size, 256, 784)

        return patch_feature # concatenated_patch_features # patch_feature
    
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



