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
from src.model.foundpose import resize_and_pad_image
from sklearn.decomposition import PCA
import faiss

from src.model.foundpose import calculate_templates_labels, calculate_templates_vector, calculate_crop_vector

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

        self.ref_data = {"descriptors": BatchedData(None)}
        for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing patch descriptor for FoundPose ...",
            ):
                # import pdb; pdb.set_trace()
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_patch_feats = self.descriptor_model.get_intermediate_layers(ref_imgs)
                self.ref_data["patch_descriptors"].append(ref_patch_feats)

        self.ref_data["patch_descriptors"].stack()  # N_objects x descriptor_size # 
        self.ref_data["patch_descriptors"] = self.ref_data["patch_descriptors"].data
    
    def filter_out_invalid_templates(self):
        num_valid_patches = list() # List of numbers of valid patches for each template
        valid_patch_features = list()
        resized_masks = list()
        for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="resizing masks for FoundPose ...",
            ):
            mask = resize_and_pad_image(self.ref_dataset[idx]["templates_masks"], target_max=30).flatten()  
            resized_masks.append(mask)

        for patch_feature, mask in zip(self.ref_data["patch_descriptors"], resized_masks):
            valid_patches = patch_feature[mask==1]
            valid_patch_features.append(valid_patches)
            num_valid_patches.append(valid_patches.shape[0]) # Append number of  valid patches for the template to the list
        valid_patch_features = torch.cat(valid_patch_features)
        return num_valid_patches, valid_patch_features


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
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean( # N_proposals x N_objects
                score_per_proposal_and_object, dim=-1
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
        pred_scores = score_per_proposal[idx_selected_proposals]
        return idx_selected_proposals, pred_idx_objects, pred_scores 
        '''
        idx_selected_proposals : index of the proposals that has score > 0.5
        pred_idx_objects : object that the proposals shows
        pred_scores : final similarity score
        '''
    def find_matched_proposals_2(self, proposal_decriptors, templates_descriptors): 
        '''
        Compare with the threshold of 0.5 and return the index of sam proposals that has score > 0.5, also return the obj_id(which object the proposal is) for this sam proposal and also the average top 5 score for this proposals
        '''

        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_decriptors, templates_descriptors
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
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean( # N_proposals x N_objects
                score_per_proposal_and_object, dim=-1
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
        pred_scores = score_per_proposal[idx_selected_proposals]
        return idx_selected_proposals, pred_idx_objects, pred_scores 
    
    @classmethod
    def kmeans_clustering(pca_patches_descriptors, ncentroids = 2048, niter = 20, verbose = True):
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

        # Building BoW
        templates_num_valid_patches, templates_valid_patch_features = self.filter_out_invalid_templates()
        
        crop_num_valid_patches, valid_crop_feature_patches = self.descriptor_model.compute_patch_features(image_np, detections) # shape as 56 ,1024 as number of proposals, 1024 as fatures dim from Dinov2
        all_valid_patch_features = torch.cat((valid_crop_feature_patches, templates_valid_patch_features), dim=0)

        pca = PCA(n_components=256, random_state=5)
        pca_crop_patches_descriptors = pca.fit_transform(np.array(all_valid_patch_features.cpu()))

        pca_crop = pca_crop_patches_descriptors[:valid_crop_feature_patches.shape[0]]
        pca_templates = pca_crop_patches_descriptors[valid_crop_feature_patches.shape[0]:]

        kmeans = self.kmeans_clustering(pca_templates, ncentroids = 2048, niter = 20, verbose = True)
        templates_labels = calculate_templates_labels(templates_num_valid_patches, kmeans, pca_templates)
        templates_vector = calculate_templates_vector(templates_labels = templates_labels, num_clusters = 2048)

        # Assign labels to the data points
        crop_labels = kmeans.index.search(pca_crop, 1)[1].reshape(-1)
        
        crop_vector = calculate_crop_vector(crop_labels = crop_labels, templates_labels = templates_labels, num_clusters = 2048)
        concat_templates_vector = torch.cat([torch.tensor(vector).view(1,-1) for vector in templates_vector]) # Goal torch.Size([642, 2048])

        proposal_stage_end_time = time.time()
        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
        ) = self.find_matched_proposals_2(crop_vector, concat_templates_vector)
        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")

        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        # detections.add_attribute("scores",(pred_scores+1)/2)
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

    def test_step_original(self, batch, idx): # idx = 0
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
        ) = self.find_matched_proposals(query_decriptors)
        print(f"Number of chosen detections with scores bigger than 0.01: {len(detections)}")

        # update detections
        detections.filter(idx_selected_proposals)
        detections.add_attribute("scores", pred_scores)
        # detections.add_attribute("scores",(pred_scores+1)/2)
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