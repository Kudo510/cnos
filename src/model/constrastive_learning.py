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

from src.model.sam import CustomSamAutomaticMaskGenerator, load_sam

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)


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

    frames_path = f"datasets/bop23_challenge/datasets/{dataset}/{data_type}/{scene_id:06d}/rgb/*.png" #"datasets/bop23_challenge/datasets/icbin/test/000001/rgb/000008.png"
    frames_path = glob.glob(frames_path)

    all_pos_proposals = []
    all_neg_proposals = []
    for frame_path in frames_path:
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
        mask_paths = glob.glob(visib_mask_paths)
        masks_gt = [(np.array(Image.open(mask_path).convert("L"))>0).astype(int) for mask_path in mask_paths]
        masks_pred = [np.array(mask.cpu()).astype(int) for mask in detections["masks"]]

        best_mask_indices = []
        for gt_i, gt in enumerate(masks_gt):

            best_iou = 0
            best_mask_index = -1

            for i, mask in enumerate(masks_pred):
                iou = calculate_iou(gt, mask)
                if iou > best_iou:
                    best_iou = iou
                    best_mask_index = i
            if best_iou >0.5:
                best_mask_indices.append(best_mask_index)
            log.info(f"The best for {gt_i}th mask is at index {best_mask_index} with an IoU of {best_iou}")

            pos_proposals = [masked_images[i] for i in best_mask_indices]
            neg_proposals = [masked_images[j] for j in range(len(masked_images)) if j not in best_mask_indices]

        del detections
    
        all_pos_proposals.append(pos_proposals)
        all_neg_proposals.append(neg_proposals)

    return all_pos_proposals, all_neg_proposals


# Custom dataset for paired images
class PairedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        
        # Randomly choose the second image
        if random.random() > 0.5:  # Positive pair
            img2, label2 = self.dataset[random.choice(self.dataset.class_to_idx[self.dataset.classes[label1]])]
            target = 0
        else:  # Negative pair
            img2, label2 = self.dataset[random.randint(0, len(self.dataset) - 1)]
            while label2 == label1:
                img2, label2 = self.dataset[random.randint(0, len(self.dataset) - 1)]
            target = 1

        return (img1, img2), target

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


def train(device, model, train_dataset):

    model = model.to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = PairedDataset(root_dir='path/to/your/dataset', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

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
