import logging, os
import os.path as osp
from tqdm import tqdm
import time
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
from src.utils.inout import load_json, save_json, casting_format_to_save_json
from src.poses.utils import (
    load_index_level_in_level2,
    get_obj_poses_from_template_level,
    NearestTemplateFinder,
    farthest_sampling,
    combine_R_and_T,
)
import torch
from src.utils.bbox_utils import CropResizePad
import pytorch_lightning as pl
from functools import partial
import multiprocessing
from src.dataloader.bop import BaseBOP


class BOPTemplatePBR(BaseBOP):
    def __init__(
        self,
        root_dir,
        out_dir,
        obj_ids, 
        template_dir, # ${machine.root_dir}/datasets/
        processing_config,
        level_templates,
        pose_distribution,
        split="train_pbr", # test
        min_visib_fract=0.8,
        max_num_scenes=10,  # not need to search all scenes since it is slow
        max_num_frames=1000,  # not need to search all frames since it is slow
        **kwargs,
    ):
        self.template_dir = template_dir # './datasets/bop23_challenge/datasets/templates_pyrender/icbin'
        # obj_ids = [
        #     int(obj_id[4:])
        #     for obj_id in os.listdir(template_dir)
        #     if osp.isdir(osp.join(template_dir, obj_id))
        # ] # all object in the template_dir folder - here [1,2]
        # obj_ids = sorted(obj_ids)
        # logging.info(f"Found {obj_ids} objects in {self.template_dir}")
        self.obj_ids = obj_ids # all the scene

        self.level_templates = level_templates # 0
        self.pose_distribution = pose_distribution # all
        self.load_template_poses(level_templates, pose_distribution) # then we get  self.index_templates and self.template_poses
        self.processing_config = processing_config #{'image_size': 224, 'max_num_scenes': 10, 'max_num_frames': 500, 'min_visib_fract': 0.8, 'num_references': 200, 'use_visible_mask': True}}
        self.root_dir = root_dir # ./datasets/bop23_challenge/datasets/icbin
        self.split = split # train_pbr
        self.out_dir = out_dir
        self.load_list_scene(split=split)
        logging.info(
            f"Found {len(self.list_scenes)} scene, but using only {max_num_scenes} scene for faster runtime"
        )

        self.list_scenes = self.list_scenes[:max_num_scenes] # just the first 10 scenes in each dataset
        self.max_num_frames = max_num_frames #1000
        self.min_visib_fract = min_visib_fract #0.8
        self.rgb_transform = T.Compose(
            [
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.proposal_processor = CropResizePad(self.processing_config.image_size)

    def __len__(self):
        return len(self.obj_ids)

    def load_metaData(self, reset_metaData): # reset_metaData =True
        start_time = time.time()
        metaData = {
            "scene_id": [],
            "frame_id": [],
            "rgb_path": [],
            "visib_fract": [],
            "obj_id": [],
            "idx_obj": [],
            "obj_poses": [],
        }
        logging.info(f"Loading metaData for split {self.split}")
        # metaData_path = osp.join(self.root_dir, f"{self.split}_metaData.csv") # load the train_pbr_metadata.csv file
        metaData_path = osp.join(self.out_dir, f"{self.split}_metaData.csv") # load the train_pbr_metadata.csv file 
        if reset_metaData:
            for scene_path in tqdm(self.list_scenes, desc="Loading metaData"): # load only first 10 scene of the dataset
                scene_id = scene_path.split("/")[-1]
                if osp.exists(osp.join(scene_path, "rgb")):
                    rgb_paths = sorted(Path(scene_path).glob("rgb/*.[pj][pn][g]")) # regex -hoac png or jpg do
                else:
                    rgb_paths = sorted(Path(scene_path).glob("gray/*.tif")) # if no rgb exists- load the gray folder instead and in gray folder the images are in .tif not png or jpg

                # load poses
                scene_gt_info = load_json(osp.join(scene_path, "scene_gt_info.json"))
                scene_gt = load_json(osp.join(scene_path, "scene_gt.json"))
                for idx_frame in range(len(rgb_paths)):
                    rgb_path = rgb_paths[idx_frame]
                    frame_id = int(str(rgb_path).split("/")[-1].split(".")[0])
                    obj_ids = [int(x["obj_id"]) for x in scene_gt[f"{frame_id}"]]
                    obj_poses = np.array(
                        [
                            combine_R_and_T(x["cam_R_m2c"], x["cam_t_m2c"])
                            for x in scene_gt[f"{frame_id}"]
                        ]
                    )
                    visib_fracts = [
                        float(x["visib_fract"]) for x in scene_gt_info[f"{frame_id}"] # list of visiable fraction of the object in the scene
                    ]

                    # add to metaData
                    metaData["visib_fract"].extend(visib_fracts)
                    metaData["obj_id"].extend(obj_ids)
                    metaData["idx_obj"].extend(range(len(obj_ids)))
                    metaData["obj_poses"].extend(obj_poses)

                    metaData["scene_id"].extend([scene_id] * len(obj_ids))
                    metaData["frame_id"].extend([frame_id] * len(obj_ids))
                    metaData["rgb_path"].extend([str(rgb_path)] * len(obj_ids))

                    if idx_frame > self.max_num_frames:
                        break
            self.metaData = pd.DataFrame.from_dict(metaData, orient="index")
            self.metaData = self.metaData.transpose()
            self.metaData.to_csv(metaData_path)
        else:
            self.metaData = pd.read_csv(metaData_path)

        # shuffle data
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index(
            drop=True
        )
        finish_time = time.time()
        logging.info(
            f"Finish loading metaData of size {len(self.metaData)} in {finish_time - start_time:.2f} seconds"
        )
        return metaData

    def load_template_poses(self, level_templates, pose_distribution):  ## here is where we choose only 42 templates from all the images only
        if pose_distribution == "all":
            self.index_templates = load_index_level_in_level2(level_templates, "all") # idx_all_level0_in_level2.npy
            self.template_poses = get_obj_poses_from_template_level( # self.template_poses  will be a list of index(from 0 to 41) and 42 poses
                self.level_templates, self.pose_distribution
            ) # level_templates = 0 # load poses from src/poses/predefined_poses/obj_poses_level0.npy
        else:
            raise NotImplementedError

    def load_processed_metaData(self, reset_metaData):
        finder = NearestTemplateFinder(
            level_templates=self.level_templates, # 0
            pose_distribution=self.pose_distribution, #all
            return_inplane=False,
        )
        # metaData_path = osp.join(self.root_dir, f"{self.split}_processed_metaData.json") # the train_pbtMeta
        metaData_path = osp.join(self.out_dir, f"{self.split}_processed_metaData.csv") ## acthung csv not json- we want to update the new csv
        if reset_metaData or not osp.exists(metaData_path): # reset_metaData = True
            self.load_metaData(reset_metaData=reset_metaData) # self.metaData now is the data frame for the metascv file
            # keep only objects having visib_fract > self.processing_config.min_visib_fract
            init_size = len(self.metaData)

            idx_keep = np.array(self.metaData["visib_fract"]) > self.min_visib_fract
            self.metaData = self.metaData.iloc[np.arange(len(self.metaData))[idx_keep]]
            self.metaData = self.metaData.reset_index(drop=True)

            selected_index = []
            index_dataframe = np.arange(0, len(self.metaData))

            # for each object, find reference frames by taking top k frames with farthest distance
            for obj_id in tqdm(
                self.obj_ids, desc="Finding nearest rendering close to template poses" # self.obj_ids is just the obj ide from the models folder- basically zB we have 2 types of obj 00001,00002
            ):
                selected_index_obj = index_dataframe[self.metaData["obj_id"] == obj_id]
                # subsample a bit if there are too many frames
                selected_index_obj = np.random.choice(selected_index_obj, 5000) # list of random 5000 number
                obj_poses = np.array(
                    self.metaData.iloc[selected_index_obj].obj_poses.tolist() # shape of obj_poses = 5000,4,4
                )
                # normalize translation to have unit norm
                obj_poses = np.array(obj_poses).reshape(-1, 4, 4)
                distance = np.linalg.norm(obj_poses[:, :3, 3], axis=1, keepdims=True)
                # print(distance[:10], distance.shape)
                obj_poses[:, :3, 3] = obj_poses[:, :3, 3] / distance

                idx_keep = finder.search_nearest_query(obj_poses) # idx_keep is 42 poses- so basically we are getting the indices of the frame_id, whose poses are most similar to the 42 templates poses
                # update metaData
                selected_index.extend(selected_index_obj[idx_keep])
            self.metaData = self.metaData.iloc[selected_index]
            logging.info(
                f"Finish processing metaData from {init_size} to {len(self.metaData)}"
            )
            self.metaData = self.metaData.reset_index(drop=True) ## self.metaData. is now for icbin will have shape of 84, 7 - 84 cos 42 templates for the 2 objects in icbin , 7 ist for all the infod s.t scene_id, frame_id ,etc
            # self.metaData = casting_format_to_save_json(self.metaData)
            self.metaData.to_csv(metaData_path)
            

        else:
            self.metaData = pd.read_csv(metaData_path).reset_index(drop=True)

    def __getitem__(self, idx): # idx is the object id ddos- for icbin we have only 2 object 00001 and 00002
        templates, boxes = [], []
        obj_ids = []
        idx_range = range(
            idx * len(self.template_poses),
            (idx + 1) * len(self.template_poses),
        ) # basically the range is 42*idx to 42*(id+1) - so we have 42 indices for the indx_range

        for i in idx_range:
            rgb_path = self.metaData.iloc[i].rgb_path
            obj_id = self.metaData.iloc[i].obj_id
            obj_ids.append(obj_id)
            idx_obj = self.metaData.iloc[i].idx_obj
            scene_id = self.metaData.iloc[i].scene_id
            frame_id = self.metaData.iloc[i].frame_id
            mask_path = osp.join(
                self.root_dir,
                self.split,
                f"{int(scene_id):06d}",
                "mask_visib",
                f"{frame_id:06d}_{idx_obj:06d}.png",
            )
            rgb = Image.open(rgb_path) # rgb path = datasets/bop23_challenge/datasets/icbin/train_pbr/000003/rgb/000725.jpg
            mask = Image.open(mask_path) # mask_path = /datasets/bop23_challenge/datasets/icbin/train_pbr/000003/mask_visib/000725_000018.png
            masked_rgb = Image.composite(
                rgb, Image.new("RGB", rgb.size, (0, 0, 0)), mask
            )
            boxes.append(mask.getbbox())
            image = torch.from_numpy(np.array(masked_rgb.convert("RGB")) / 255).float()
            templates.append(image)

        assert (
            len(np.unique(obj_ids)) == 1
        ), f"Only support one object per batch but found {np.unique(obj_ids)}"

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        templates_croped = self.proposal_processor(images=templates, boxes=boxes)
        return {
            "templates": self.rgb_transform(templates_croped),
            "original_templates": templates_croped,
            } # to normalize the template # 
        # return {"templates": templates_croped} # to normalize the template # 
        ### see at the end we get 42 templates for each idx/object id - we will get 42*7 as df with all information s.t scnene id, frame id, poses for the tempaltes. for icbin we only have 2 indices , cos we have only 2 cad models/object in icbin

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import DictConfig, OmegaConf
    from torchvision.utils import make_grid, save_image

    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )
    dataset = BOPTemplatePBR(
        root_dir="/gpfsscratch/rech/tvi/uyb58rn/datasets/bop23_challenge/datasets/lmo",
        template_dir="/gpfsscratch/rech/tvi/uyb58rn/datasets/bop23_challenge/datasets/templates_pyrender/lmo",
        obj_ids=None,
        level_templates=0,
        pose_distribution="all",
        processing_config=processing_config,
    )
    os.makedirs("./tmp", exist_ok=True)
    dataset.load_processed_metaData(reset_metaData=True)
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        sample["templates"] = inv_rgb_transform(sample["templates"])
        save_image(sample["templates"], f"./tmp/lm_{idx}.png", nrow=7)
