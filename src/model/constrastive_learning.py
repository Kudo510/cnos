from PIL import Image
import numpy as np
import glob
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
import json
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)

random.seed(10)

def compute_inplane(rot_query_openCV, rot_template_openCV):
    delta = rot_template_openCV.dot(rot_query_openCV.T)
    inp = extract_inplane_from_pose(delta)
    # double check to make sure that reconved rotation is correct
    R_inp = convert_inplane_to_rotation(inp)
    recovered_R1 = R_inp.dot(rot_template_openCV)
    err = geodesic_numpy(recovered_R1, rot_query_openCV)
    if err >= 15:
        print("WARINING, error of recovered pose is >=15, err=", err)
    return inp


def opencv2opengl(cam_matrix_world):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if len(cam_matrix_world.shape) == 2:
        return np.matmul(transform, cam_matrix_world)
    else:
        transform = np.tile(transform, (cam_matrix_world.shape[0], 1, 1))
        return np.matmul(transform, cam_matrix_world)


def extract_inplane_from_pose(pose):
    inp = Rotation.from_matrix(pose).as_euler("zyx", degrees=True)[0]
    return inp


def convert_inplane_to_rotation(inplane):
    R_inp = Rotation.from_euler("z", -inplane, degrees=True).as_matrix()
    return R_inp


def geodesic_numpy(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))


def extract_positive_pairs(all_pos_proposals, templates):
    pos_pairs = list()
    for proposals_id in range(len(all_pos_proposals)):
        obj_query_pose = all_pos_proposals[proposals_id]["pose"][None]
        obj_template_poses = templates["poses"]

        return_inplane = True

        obj_query_openGL_pose = opencv2opengl(obj_query_pose)
        obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3 # (translation components) -  It assumes that the 3D location is found in the third column of the pose matrices.
        obj_template_openGL_poses = opencv2opengl(obj_template_poses)
        obj_template_openGL_locations = obj_template_openGL_poses[:, 2, :3]  # Nx3 # (translation components)

        # find the nearest template
        # It computes the pairwise distances between each query pose location and each template pose location using cdist.
        distances = cdist(obj_query_openGL_location, obj_template_openGL_locations)
        best_index_in_pose_distribution = np.argmin(distances, axis=-1)  # M
        if return_inplane:
            nearest_poses = obj_template_poses[best_index_in_pose_distribution]
            inplanes = np.zeros(len(obj_query_pose))
            for idx in range(len(obj_query_pose)):
                rot_query_openCV = obj_query_pose[idx, :3, :3]
                rot_template_openCV = nearest_poses[idx, :3, :3]
                inplanes[idx] = compute_inplane(rot_query_openCV, rot_template_openCV)

        pos_pair = {
            "img1" : resize_and_pad_image(templates["rgb"][best_index_in_pose_distribution[0]], target_max=224), # resize and pad images
            "img2" : resize_and_pad_image(all_pos_proposals[proposals_id]["rgb"], target_max=224), 
            "label" : 1
        }
        pos_pairs.append(pos_pair)

    return pos_pairs


def extract_negative_pairs(all_neg_proposals, all_pos_proposals, templates):
    selected_neg_proposals = random.sample(all_neg_proposals, len(all_pos_proposals)) # Num of negative = num of positive
    copied_templates = templates.copy()

    neg_pairs = list()
    for neg_prop in selected_neg_proposals:
        selected_temp = random.choice(copied_templates)
        copied_templates.remove(selected_temp)

        neg_pair = {
            "img1" : resize_and_pad_image(neg_prop, target_max=224), 
            "img2" : resize_and_pad_image(selected_temp, target_max=224),
            "label" : 0
        }
        neg_pairs.append(neg_pair)
    return neg_pairs


def resize_and_pad_image(image, target_max=420):
    '''
    cnos target_max = 224
    foundpose target_max = 420
    '''
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

    
def preprocess_images(input_images, contrastive_model):
    '''
    Return the features of input images
    '''
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    normalized_images = [rgb_normalize(input_image/255.0).float() for img in input_images]
    resized_images = [resize_and_pad_image(normalized_image, target_max=224) for normalized_image in normalized_images]

    batch_size = 16
    layers_list = list(range(24))
    batches = [resized_images[i:i+batch_size] for i in range(0, len(resized_images), batch_size)]
    patch_features= list()

    for batch in batches:
        batch = torch.stack(batch)
        size = batch.shape[0]
        torch.cuda.empty_cache()
        with torch.no_grad(): 
            batch_feature = contrastive_model(batch).reshape(size,-1,1024).cpu()
        patch_features.append(batch_feature.to('cpu'))
        del batch_feature
    patch_features = torch.cat(patch_features)
    del contrastive_model

    return patch_features


def extract_object_by_mask(image, mask, width: int = 512):
    mask = Image.fromarray(mask)
    masked_image = Image.composite(
        image, Image.new("RGB", image.size, (0, 0, 0)), mask)
    cropped_image = masked_image.crop(masked_image.getbbox())
    # new_height = width * cropped_image.height // cropped_image.width
    return cropped_image


def calculate_iou(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def extract_dataset(dataset="icbin",data_type="test", scene_id=1):  # data_type test or train 
    '''
    Extract positive and negative proposals from sam on all frame in test folder in dataset icbin - scene id is 1 here for the first object
    Use IoU to choose the positive proposals (the most similar masks as the gt)
    '''
    model_type = "vit_h"
    checkpoint_dir =  "datasets/bop23_challenge/pretrained/segment-anything"
    log.info("loading sam")
    sam_model = load_sam(model_type, checkpoint_dir)
    custom_sam_model = CustomSamAutomaticMaskGenerator(sam=sam_model)
    custom_sam_model.predictor.model.to("cuda")

    frame_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/*.png" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
    frame_paths = sorted(glob.glob(frame_paths)) # only 50 not 55 paths - some ids are missing s.t 10

    scene_gt_json = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/scene_gt.json"
    scene_gt = json.load(open(scene_gt_json, 'r'))
        
    all_pos_proposals = []
    all_neg_proposals = []
    for frame_path in frame_paths:
        rgb = Image.open(frame_path).convert("RGB") # rotate(180)
        detections = custom_sam_model.generate_masks(np.array(rgb)) # Include masks and bboxes
        
        masked_images = []
        for mask in detections["masks"].cpu():
            binary_mask = np.array(mask) * 255
            binary_mask = binary_mask.astype(np.uint8)
            masked_image = extract_object_by_mask(rgb, binary_mask)
            masked_images.append(masked_image)

        frame_id = frame_path.split("/")[-1].split(".")[0]
        visib_mask_paths = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/mask_visib/{frame_id}_*.png" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
        mask_paths = sorted(glob.glob(visib_mask_paths))

        poses = list()
        for mask_path in mask_paths:
            mask_scene_id_ = int(mask_path.split('/')[-1].split('.')[0].split('_')[0])
            mask_frame_id = int(mask_path.split('/')[-1].split('.')[0].split('_')[1])

            # Extracting rotation (R) and translation (t)
            R = np.array(scene_gt[str(mask_scene_id_)][mask_frame_id]["cam_R_m2c"]).reshape(3,3)
            t = np.array(scene_gt[str(mask_scene_id_)][mask_frame_id]["cam_t_m2c"])

            # Construct the 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T)

        masks_gt = {
            "masks" : [(np.array(Image.open(mask_path).convert("L"))>0).astype(int) for mask_path in mask_paths],
            "poses" : poses
        }
        masks_pred = {
            "masks" : [np.array(mask.cpu()).astype(int) for mask in detections["masks"]],
            "rgb" : [rgb for rgb in masked_images]
        }

        best_mask_indices = []
        pos_proposals = []
        for i_gt, mask_gt in enumerate(masks_gt["masks"]):

            best_iou = 0
            best_mask_index = -1

            for i, mask_pred in enumerate(masks_pred["masks"]):
                iou = calculate_iou(mask_gt, mask_pred)
                if iou > best_iou:
                    best_iou = iou
                    best_mask_index = i

            if best_iou >0.5:
                best_mask_indices.append(best_mask_index)
                # pos_proposal rgb from prediction and pose from gt
                pos_proposal = {
                    "rgb": masks_pred["rgb"][best_mask_index],
                    "pose": masks_gt["poses"][i_gt]
                }
                pos_proposals.append(pos_proposal)
                
            log.info(f"For frame {frame_path.split('/')[-1]}, the best for {i_gt}th mask is at index {best_mask_index} with an IoU of {best_iou}")      
        
        del detections
    
        all_pos_proposals.append(pos_proposals)
        all_neg_proposals.append([masked_images[j] for j in range(len(masked_images)) if j not in best_mask_indices])
        log.info(f"Number of prediction masks: {len(masks_pred['masks'])}, positive proposals: {len(pos_proposals)}, negative proposals: {len(all_neg_proposals[-1])}")

    return all_pos_proposals, all_neg_proposals


# Custom dataset for paired images
class PairedDataset():
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img1 = self.transform(self.dataset["img1"])
        img2 = self.transform(self.dataset["img2"])
        label = self.dataset["label"]
        return img1, img2, label

    def __len__(self):
        return len(self.dataset)


class ContrastiveModel(nn.Module):
    def __init__(self,):
        self.layers_list = list(range(24))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dinov2_vitl14.patch_size = 14
        if torch.cuda.is_available():
            self.dinov2_vitl14 = torch.nn.DataParallel(dinov2_vitl14).to(device)  # Use DataParallel for multiple GPUs

    def forward(x):
        return self.dinov2_vitl14.module.get_intermediate_layers(
            x.to(device), n=layers_list, return_class_token=True
            )[18][0]


# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


def train(device, model, template_paths, template_poses_path, all_pos_proposals, all_neg_proposals):

    model = model.to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loading and preprocessing
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Lambda(lambda x: x / 255.0),  # Ensures the scaling by 255.0
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    template_paths = sorted(glob.glob(template_paths))
    template_poses = np.load(template_poses_path)

    templates = {
        "rgb" : [np.array(Image.open(template_path).convert("RGB")) for template_path in template_paths],
        "poses" : [template_pose for template_pose in template_poses]
    }

    postive_pairs = extract_positive_pairs(all_pos_proposals, templates)
    negative_pairs = extract_negative_pairs(all_neg_proposals, all_pos_proposals, templates)

    train_postive_pairs, remaining_postive_pairs = train_test_split(postive_pairs, test_size=0.2, random_state=0)
    val_postive_pairs, test_postive_pairs = train_test_split(remaining_postive_pairs, test_size=0.5, random_state=0)

    train_negative_pairs, remaining_negative_pairs = train_test_split(negative_pairs, test_size=0.2, random_state=0)
    val_negative_pairs, test_negative_pairs = train_test_split(remaining_negative_pairs, test_size=0.5, random_state=0)

    train_dataset = PairedDataset(train_postive_pairs + train_negative_pairs, transform=transform)
    val_dataset = PairedDataset(val_postive_pairs + val_negative_pairs, transform=transform)
    test_dataset = PairedDataset(test_postive_pairs + test_negative_pairs, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, ((img1, img2), label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img1), model(img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
