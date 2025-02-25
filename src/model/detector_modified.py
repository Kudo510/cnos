import torch
import torchvision.transforms as T
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
import cv2


class CNOS(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
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
        

    def set_reference_objects(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data = {"descriptors": BatchedData(None)}
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth") # ./datasets/bop23_challenge/datasets/templates_pyrender/icbin/descriptors_pbr.pth for pbr renderer

        ## self.ref_dataset is src.dataloader.bop.BOPTemplatePBR
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                )
                self.ref_data["descriptors"].append(ref_feats)

            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["descriptors"], descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}"
        )

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
        # compute matching scores for each proposals
        scores = self.matching_config.metric( # src.model.loss.PairwiseSimilarity
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
        elif self.matching_config.aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0] # average the top 5 scores to get teh final score
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_query

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]
        return idx_selected_proposals, pred_idx_objects, pred_scores 

    def test_step(self, batch, idx):
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # batch["image"][0] 0 to retrun image 3,H,W cos the batch has size of batch_size,3,W,H - where batch_size = 1
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255) # just get image in numpy in range of 0,255

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np) ## proposals are the dict with masks and boxes of the objects

        # init detections with masks and boxes
        detections = Detections(proposals) 
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        ) # here we get 68 proposals of 489*640 but filtering out all small proposals we get 39 detections

        # compute descriptors
        query_decriptors = self.descriptor_model(image_np, detections) # descriptor_model = dinov2 # so here is for getting proposal_descriptors from the filtered masks and bboxes - shape 56,1024- means we have 56 proposals
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals, # from 0 to 55 - so 56 indices
            pred_idx_objects,
            pred_scores,
        ) = self.find_matched_proposals(query_decriptors) # query_decriptors shape of 56, 1024

        # update detections
        detections.filter(idx_selected_proposals) # takes only the proposals that found the matched - just save all the indices from idx_selected_proposals
        detections.add_attribute("scores", pred_scores) # score of this match
        detections.add_attribute("object_ids", pred_idx_objects) # matched obj_id 
        detections.apply_nms_per_object_id( # filter out - using nms to get rif of overlap proposals
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
        
        # basically getting the scene and frame id to get the adrress of the propsal- basically what we have at the end is the proposals and its matched in the frame id from the scene y
        scene_id = batch["scene_id"][0] # '000001'
        frame_id = batch["frame_id"][0] # '51'
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file.split("_")[1:]}/scene{scene_id}_frame{frame_id}",
        ) # './datasets/bop23_challenge/results/cnos_exps/predictions/icbin/CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin/scene000001_frame51' - so like output path

        # save detections to file # here to save the npz file dos zB  datasets/bop23_challenge/results/cnos_exps/predictions/icbin/CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin/scene000001_frame0.npz
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime, # just time to run the test thoi ys
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file # so this is the runtime file in predictions do - zB datasets/bop23_challenge/results/cnos_exps/predictions/icbin/CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin/scene000001_frame0_runtime.npz - it just contains the runing time for proposal and matching stage
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

    def test_an_image_step(self, image_path):
        image_np = cv2.imread(image_path)
        image_np = (
            self.inv_rgb_transform(batch["image"][0]) # batch["image"][0] 0 to retrun image 3,H,W cos the batch has size of batch_size,3,W,H - where batch_size = 1
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255) # just get image in numpy in range of 0,255

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np) ## proposals are the dict with masks and boxes of the objects

        # init detections with masks and boxes
        detections = Detections(proposals) 
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        ) # here we get 68 proposals of 489*640 but filtering out all small proposals we get 39 detections

        # compute descriptors
        query_decriptors = self.descriptor_model(image_np, detections) # descriptor_model = dinov2 # so here is for getting proposal_descriptors from the filtered masks and bboxes - shape 56,1024- means we have 56 proposals
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals, # from 0 to 55 - so 56 indices
            pred_idx_objects,
            pred_scores,
        ) = self.find_matched_proposals(query_decriptors) # query_decriptors shape of 56, 1024

        # update detections
        detections.filter(idx_selected_proposals) # takes only the proposals that found the matched - just save all the indices from idx_selected_proposals
        detections.add_attribute("scores", pred_scores) # score of this match
        detections.add_attribute("object_ids", pred_idx_objects) # matched obj_id 
        detections.apply_nms_per_object_id( # filter out - using nms to get rif of overlap proposals
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        detections.to_numpy()
        
        # basically getting the scene and frame id to get the adrress of the propsal- basically what we have at the end is the proposals and its matched in the frame id from the scene y
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        ) # './datasets/bop23_challenge/results/cnos_exps/predictions/icbin/CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin/scene000001_frame51' - so like output path

        # save detections to file # here to save the npz file dos zB  datasets/bop23_challenge/results/cnos_exps/predictions/icbin/CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin/scene000001_frame0.npz
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime, # just time to run the test thoi ys
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file # so this is the runtime file in predictions do - zB datasets/bop23_challenge/results/cnos_exps/predictions/icbin/CustomSamAutomaticMaskGenerator_template_pbr0_aggavg_5_icbin/scene000001_frame0_runtime.npz - it just contains the runing time for proposal and matching stage

        return 0