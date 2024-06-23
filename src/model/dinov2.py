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

class SmallDinov2(nn.Module):
    def __init__(self, dinov2_vitl14, num_block=18):
        super().__init__()
        self.dinov2_vitl14 = dinov2_vitl14

        # Extract the layers
        self.embedding = self.dinov2_vitl14.patch_embed
        #self.blocks =  nn.Sequential(*list(self.dinov2_vitl14.blocks)[:19])
        self.blocks = self.dinov2_vitl14.blocks[:num_block+1]
        self.norms = self.dinov2_vitl14.norm
        self.head = self.dinov2_vitl14.head

        self.model = nn.Sequential(
            self.embedding,
            self.blocks,
            self.norms,
            self.head
        )

    def forward(self, x):
        x = self.model(x)
        return x


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
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [num_proposals, 3, 224, 224]
        return processed_masked_rgbs # # [num_proposals, 3, 224, 224] - bascially we got the stack of proposals image (they are already resized and scaled)

    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken": #our case
            if images.shape[0] > self.chunk_size:
                features = self.forward_by_chunk(images)
            else:
                features = self.model(images)
        else:  # get both features
            raise NotImplementedError
        return features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs): # processed_rgbs [num_proposals, 3, 224, 224]
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
        ) #[num_proposals, 3, 224, 224]
        return self.forward_by_chunk(processed_rgbs)

    @torch.no_grad()
    def forward(self, image_np, proposals):
        return self.forward_cls_token(image_np, proposals)

