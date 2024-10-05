import torch
import numpy as np
import torchvision
from torchvision.ops.boxes import batched_nms, box_area
import logging
from src.utils.inout import save_json, load_json, save_npz
from src.utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy, force_binary_mask
import time
from PIL import Image

lmo_object_ids = np.array(
    [
        1,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
    ]
)  # object ID of occlusionLINEMOD is different


def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


class BatchedData: # just that it convert a list of data to list of batches of data with length as batch_size
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0): # here to stack data
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


class Detections:
    """
    A structure for storing detections.
    """

    def __init__(self, data) -> None: # data is dict of boxes and masks
        if isinstance(data, str):
            data = self.load_from_file(data)
        for key, value in data.items():
            setattr(self, key, value) # So now we have self.boxes and self.masks = ... for the class
        self.keys = list(data.keys())
        if "boxes" in self.keys:
            if isinstance(self.boxes, np.ndarray): # convert numpys to torch
                self.to_torch()
            self.boxes = self.boxes.long()

    def remove_very_small_detections(self, config): # after this step only valid boxes, masks are saved, other are filtered out
        # min_box_size: 0.05 # relative to image size 
        # min_mask_size: 3e-4 # relative to image size
        img_area = self.masks.shape[1] * self.masks.shape[2]
        box_areas = box_area(self.boxes) / img_area
        mask_areas = self.masks.sum(dim=(1, 2)) / img_area
        keep_idxs = torch.logical_and(
            box_areas > config.min_box_size**2, mask_areas > config.min_mask_size
        ) 
        logging.info(f"Removing {len(keep_idxs) - keep_idxs.sum()} detections")
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms_per_object_id(self, nms_thresh=0.5):
        '''
        self.object_ids actually is pred_idx_objects - a list of object id that the selected proposals are 
        '''
        keep_idxs = BatchedData(None) ## cos later we will use .cat to add data to the object
        all_indexes = torch.arange(len(self.object_ids), device=self.boxes.device)
        for object_id in torch.unique(self.object_ids): # so for icbin only returns 1 and 2
            idx = self.object_ids == object_id
            idx_object_id = all_indexes[idx]
            keep_idx = torchvision.ops.nms(
                self.boxes[idx].float(), self.scores[idx].float(), nms_thresh
            )
            keep_idxs.cat(idx_object_id[keep_idx])
        keep_idxs = keep_idxs.data.cpu()
        
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms(self, nms_thresh=0.5):       
        keep_idx = torchvision.ops.nms(
            self.boxes.float(), self.scores.float(), nms_thresh
        )
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idx])


    def _is_box_inside(self, box1, box2, noise = 5):
        """Check if box1 is completely inside box2."""
        return (box2[0] <= box1[0] + noise and box2[1] <= box1[1] + noise and
                box2[2] >= box1[2] - noise and box2[3] >= box1[3] - noise)


    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / float(area1 + area2 - intersection)
        return iou


    def filter_cluttered_bboxes(self, overlap_threshold=0.01, min_neighbors=2):
        """
        Filter bounding boxes to keep only those in cluttered scenes.
        
        Args:
        bboxes (list): List of bounding boxes, each in format [x1, y1, x2, y2]
        overlap_threshold (float): IoU threshold to consider boxes as overlapping
        min_neighbors (int): Minimum number of overlapping neighbors to consider a box as part of a cluttered scene
        
        Returns:
        list: Filtered list of bounding boxes
        """
              
        n = self.boxes.shape[0]
        neighbor_count = np.zeros(n)
        
        for i in range(n):
            for j in range(i+1, n):
                if self._calculate_iou(self.boxes[i], self.boxes[j]) > overlap_threshold:
                    neighbor_count[i] += 1
                    neighbor_count[j] += 1

        keep = [True if count >= min_neighbors else False for count in neighbor_count] ## choose id of bbox with at least 2 neighbor
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep])


    def filter_contained_boxes(self):
        """Filter out boxes that are completely inside other boxes."""
        n = self.boxes.shape[0]
        keep = torch.ones(n, dtype=torch.bool)
        
        for i in range(n):
            if keep[i]:
                for j in range(n):
                    if i != j and keep[j]:
                        if self._is_box_inside(self.boxes[i], self.boxes[j]):
                            keep[i] = False
                            break
        
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep])
            
    # def apply_nms_contrastive_learning(self, nms_thresh=0.5):
    #     self.filter_contained_boxes()
    #     self.apply_nms(nms_thresh)

    def add_attribute(self, key, value):
        setattr(self, key, value)
        self.keys.append(key)

    def __len__(self):
        return len(self.boxes)

    def check_size(self):
        mask_size = len(self.masks)
        box_size = len(self.boxes)
        score_size = len(self.scores)
        object_id_size = len(self.object_ids)
        assert (
            mask_size == box_size == score_size == object_id_size
        ), f"Size mismatch {mask_size} {box_size} {score_size} {object_id_size}"

    def to_numpy(self):
        for key in self.keys:
            setattr(self, key, getattr(self, key).cpu().numpy())

    def to_torch(self):
        for key in self.keys:
            a = getattr(self, key)
            setattr(self, key, torch.from_numpy(getattr(self, key)))

    def save_to_file(
        self, scene_id, frame_id, runtime, file_path, dataset_name, return_results=False
    ):
        """
        scene_id, image_id, category_id, bbox, time
        """
        boxes = xyxy_to_xywh(self.boxes)
        results = {
            "scene_id": scene_id,
            "image_id": frame_id,
            "category_id": self.object_ids + 1
            if dataset_name != "lmo"
            else lmo_object_ids[self.object_ids],
            "score": self.scores,
            "bbox": boxes,
            "time": runtime,
            "segmentation": self.masks,
        }
        save_npz(file_path, results)
        if return_results:
            return results
    
    def save_to_file_2(
        self, scene_id, frame_id, runtime, file_path, dataset_name, return_results=False
    ):
        """
        scene_id, image_id, category_id, bbox, time
        """
        boxes = xyxy_to_xywh(self.boxes)
        results = {
            "scene_id": scene_id,
            "image_id": frame_id,
            "category_id": self.object_ids
            if dataset_name != "lmo"
            else lmo_object_ids[self.object_ids],
            # "score": self.scores,
            "bbox": boxes,
            # "time": runtime,
            "segmentation": self.masks,
        }
        save_npz(file_path, results)
        if return_results:
            return results

    def load_from_file(self, file_path):
        data = np.load(file_path)
        masks = data["segmentation"]
        boxes = xywh_to_xyxy(np.array(data["bbox"]))
        data = {
            "object_ids": data["category_id"] - 1,
            "bbox": boxes,
            "scores": data["score"],
            "masks": masks,
        }
        logging.info(f"Loaded {file_path}")
        return data

    def filter(self, idxs):
        for key in self.keys:
            setattr(self, key, getattr(self, key)[idxs])

    def clone(self):
        """
        Clone the current object
        """
        return Detections(self.__dict__.copy())


def convert_npz_to_json(idx, list_npz_paths):
    npz_path = list_npz_paths[idx]
    detections = np.load(npz_path)
    results = []
    for idx_det in range(len(detections["bbox"])):
        result = {
            "scene_id": int(detections["scene_id"]),
            "image_id": int(detections["image_id"]),
            "category_id": int(detections["category_id"][idx_det]),
            "bbox": detections["bbox"][idx_det].tolist(),
            "score": float(detections["score"][idx_det]),
            "time": float(detections["time"]),
            "segmentation": mask_to_rle(
                force_binary_mask(detections["segmentation"][idx_det])
            ),
        }
        results.append(result)
    return results

def convert_npz_to_json_2(idx, list_npz_paths):
    ''' just remove key time'''
    npz_path = list_npz_paths[idx]
    detections = np.load(npz_path)
    results = []
    for idx_det in range(len(detections["bbox"])):
        result = {
            "scene_id": int(detections["scene_id"]),
            "image_id": int(detections["image_id"]),
            "category_id": int(detections["category_id"][idx_det]),
            "bbox": detections["bbox"][idx_det].tolist(),
            "score": 1.0, # float(detections["score"][idx_det]),
            "time": 1.0, # float(detections["time"]),
            "segmentation": mask_to_rle(
                force_binary_mask(detections["segmentation"][idx_det])
            ),
        }
        results.append(result)
    return results
