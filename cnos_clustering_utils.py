import matplotlib.pyplot as plt
from math import ceil
from torchvision.ops.boxes import batched_nms, box_area
import cv2
import numpy as np
import logging
import torch
from PIL import Image
import torchvision.transforms as T
from src.model.constrastive_learning import resize_and_pad_image
import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)


def move_to_device(segmentor_model, device="cuda"):
# if there is predictor in the model, move it to device
    if hasattr(segmentor_model, "predictor"):
        segmentor_model.predictor.model = (
            segmentor_model.predictor.model.to(device)
        )
    else:
        segmentor_model.model.setup_model(device=device, verbose=True)


def plot_images(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(20, 30))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        ax.imshow(images[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def extract_object_by_mask(image, mask, width: int = 512):
    mask = Image.fromarray(mask)
    masked_image = Image.composite(
        image, Image.new("RGB", image.size, (0, 0, 0)), mask)
    cropped_image = masked_image.crop(masked_image.getbbox())
    # new_height = width * cropped_image.height // cropped_image.width
    return cropped_image


def tighten_bboxes(sam_detections, device="cuda"):
    # Apply morphological opening - it does eorion then dilation to remove noise
    kernel = np.ones((5,5), np.uint8)

    filtered_masks = list()
    for mask in sam_detections["masks"].cpu():
        filtered_mask = cv2.morphologyEx(np.array(mask, dtype="uint8"), cv2.MORPH_OPEN, kernel)
        filtered_masks.append(torch.tensor(filtered_mask))
    sam_detections["masks"] = torch.stack(filtered_masks).to(device)
    return sam_detections


def _remove_very_small_detections(masks, boxes): # after this step only valid boxes, masks are saved, other are filtered out
    min_box_size = 0.05 # relative to image size 
    min_mask_size = 300/(640*480) # relative to image size assume the pixesl should be in range (300, 10000) need to remove them 
    max_mask_size = 10000/(640*480) 
    img_area = masks.shape[1] * masks.shape[2]
    box_areas = box_area(boxes) / img_area
    formatted_values = [f'{value.item():.6f}' for value in box_areas*img_area]
    mask_areas = masks.sum(dim=(1, 2)) / img_area
    keep_idxs = torch.logical_and(
        torch.logical_and(mask_areas > min_mask_size, mask_areas < max_mask_size),
        box_areas > min_box_size**2
    )

    return keep_idxs


def extract_sam_crops_features(crops, dino_model, device):
    rgb_normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    scaled_padded_crops = []
    for crop in crops:

        normalized_crop = rgb_normalize(np.array(crop)/255.0).float()
        # normalized_crop_rgb = torch.tensor(crop_rgb, dtype=torch.float32).permute(2,0,1)

        scaled_padded_crop = resize_and_pad_image(normalized_crop, target_max=224) # Unsqueeze to make it as a stack of proposals - here we use only 1 proposals
        # print("scaled_padded_crop_rgb.shape", scaled_padded_crop_rgb.shape)

        # # Display the crop
        # plt.imshow(scaled_padded_crop_rgb.squeeze(0).permute(1,2,0))
        # plt.axis('off')  # Optional: Turn off the axis
        # plt.show()
        scaled_padded_crops.append(scaled_padded_crop)
    scaled_padded_crops = torch.stack(scaled_padded_crops)

    # Extract features from 18th layer of Dinov2 
    layers_list = list(range(24))
    torch.cuda.empty_cache()
    with torch.no_grad(): 
        feature_patches= dino_model.module.get_intermediate_layers(
            scaled_padded_crops.to(device), n=layers_list, return_class_token=True)[23][1].reshape(-1,1024)
    del dino_model

    return feature_patches


def hierarchical_clustering(embeddings, n_clusters=2):
    """
    Perform hierarchical clustering on the input embeddings, plot the dendrogram, 
    visualize the cluster assignments, and calculate silhouette score.

    Parameters:
    embeddings (np.ndarray): Input data (e.g., embeddings), shape (num_samples, num_features).
    n_clusters (int): Number of clusters for hierarchical clustering.
    """
    
    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and plot the dendrogram
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    if isinstance(embeddings, torch.Tensor):
        if embeddings.is_cuda:
            logging.info("Moving CUDA tensor to CPU...")
            embeddings = embeddings.cpu().numpy()  # Move to CPU and convert to NumPy
        else:
            embeddings = embeddings.numpy()  # Just convert to NumPy if already on CPU

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(embeddings)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    logging.info(f"The average silhouette score is: {silhouette_avg}")

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(clustering, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    # Plot the first two dimensions of the data with cluster labels
    plt.figure(figsize=(10, 7))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, cmap='viridis')
    plt.title("Hierarchical Clustering Result (First 2 Dimensions)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(label='Cluster Label')
    plt.show()

    # Log cluster labels and cluster size
    logging.info(f"Cluster labels: {cluster_labels}")
    logging.info(f"Number of samples in each cluster: {np.bincount(cluster_labels)}")












