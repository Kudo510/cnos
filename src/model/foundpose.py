import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from src.utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from torchvision.utils import make_grid, save_image
from src.model.utils import BatchedData
from copy import deepcopy

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