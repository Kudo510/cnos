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
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

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


def analyze_tsne_with_svm_2(tsne_results, n_samples_first_class, total_samples):
    """
    Analyze t-SNE results using One-Class SVM classification.
    
    :param tsne_results: numpy array of shape (total_samples, 2) containing t-SNE results
    :param n_samples_first_class: number of samples that should belong to the first class
    :param total_samples: total number of samples
    :return: tuple containing (other_same_class, accuracy)
    """
    # Use only the first n_samples_first_class points to train the One-Class SVM
    X_train = tsne_results[:n_samples_first_class]

    # Create and train the One-Class SVM model
    # svm_model = svm.OneClassSVM(kernel='rbf', nu=0.1)
    svm_model = svm.OneClassSVM(kernel='poly', degree=3)

    svm_model.fit(X_train)

    # Predict on the entire dataset
    predictions = svm_model.predict(tsne_results)
    
    # In One-Class SVM, 1 indicates points similar to the training set, -1 indicates outliers
    # We'll consider 1 as our "class 0" (similar to the first n_samples_first_class points)
    predictions = (predictions + 1) // 2  # Convert -1 to 0, and 1 to 1

    # Find indices of points classified as similar to the first class
    same_class_indices = np.where(predictions == 1)[0]
    other_same_class = [idx for idx in same_class_indices if idx >= n_samples_first_class]
    print("Indices of other points classified as similar to the first class:")
    print(other_same_class)

    # Create a mesh to plot in
    x_min, x_max = tsne_results[:, 0].min() - 1, tsne_results[:, 0].max() + 1
    y_min, y_max = tsne_results[:, 1].min() - 1, tsne_results[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Obtain labels for each point in mesh
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z + 1) // 2  # Convert -1 to 0, and 1 to 1
    Z = Z.reshape(xx.shape)

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

    # Plot all points
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=predictions, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Highlight the first n_samples_first_class points
    plt.scatter(tsne_results[:n_samples_first_class, 0], tsne_results[:n_samples_first_class, 1], 
                c='green', s=100, label=f'First {n_samples_first_class} points', edgecolor='black')

    # Highlight the other points classified as similar
    plt.scatter(tsne_results[other_same_class, 0], tsne_results[other_same_class, 1],
                c='yellow', s=100, label='Other similar points', edgecolor='black')

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('One-Class SVM Classification of t-SNE results')
    plt.colorbar(scatter)
    plt.legend()

    # Add annotations for class sizes
    print(f"Additional points classified as similar: {len(other_same_class)}")
    print(f"Total points classified as similar: {n_samples_first_class + len(other_same_class)}")
    print(f"Points classified as different: {total_samples - (n_samples_first_class + len(other_same_class))}")

    plt.show()

    # Calculate the fraction of points classified as similar to the first class
    similarity_fraction = (n_samples_first_class + len(other_same_class)) / total_samples
    print(f"Fraction of points classified as similar: {similarity_fraction:.2f}")

    return same_class_indices, similarity_fraction


import hdbscan
from sklearn.metrics import silhouette_score
from collections import Counter

def analyze_tsne_with_hdbscan(tsne_results, n_samples_first_class, total_samples):
    """
    Analyze t-SNE results using HDBSCAN clustering, handling noise points.
    
    :param tsne_results: numpy array of shape (total_samples, 2) containing t-SNE results
    :param n_samples_first_class: number of samples that should belong to the first class
    :param total_samples: total number of samples
    :return: tuple containing (other_same_class, silhouette_score)
    """
    # Create and fit the HDBSCAN model
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(tsne_results)

    # Count the occurrences of each label in the first n_samples_first_class points
    first_class_label_counts = Counter(cluster_labels[:n_samples_first_class])
    
    # Identify the most common non-noise cluster in the first class
    first_class_cluster = max((label for label in first_class_label_counts if label != -1), 
                              key=first_class_label_counts.get, default=-1)

    # If all first class points are noise, set first_class_cluster to -1
    if first_class_cluster == -1:
        print("Warning: All points in the first class are classified as noise.")
        other_same_class = []
    else:
        # Find indices of other points in the same cluster as the first class
        same_cluster_indices = np.where(cluster_labels == first_class_cluster)[0]
        other_same_class = [idx for idx in same_cluster_indices if idx >= n_samples_first_class]
        other_same_class = [i - n_samples_first_class for i in other_same_class]

    print("Indices of other points in the same cluster as the first class:")
    print(other_same_class)

    # Calculate silhouette score (excluding noise points)
    non_noise_mask = cluster_labels != -1
    if np.sum(non_noise_mask) > 1:  # Ensure we have at least 2 non-noise points
        silhouette_avg = silhouette_score(tsne_results[non_noise_mask], 
                                          cluster_labels[non_noise_mask])
    else:
        silhouette_avg = 0  # Default value if we can't compute silhouette score
        print("Warning: Not enough non-noise points to compute silhouette score.")

    # Plot the results
    plt.figure(figsize=(12, 10))

    # Plot all points
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                          c=cluster_labels, cmap='viridis', s=50, alpha=0.7)

    # Highlight the first n_samples_first_class points
    plt.scatter(tsne_results[:n_samples_first_class, 0], tsne_results[:n_samples_first_class, 1], 
                c='red', s=100, label=f'First {n_samples_first_class} points', edgecolor='black')

    # Highlight the other points in the same cluster (if any)
    if other_same_class:
        plt.scatter(tsne_results[other_same_class, 0], tsne_results[other_same_class, 1],
                    c='yellow', s=100, label='Other points in same cluster', edgecolor='black')

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('HDBSCAN Clustering of t-SNE results')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()

    # Add annotations for cluster sizes
    print(f"Points in the first class: {n_samples_first_class}")
    print(f"Additional points in the same cluster: {len(other_same_class)}")
    print(f"Total points in the same cluster: {n_samples_first_class + len(other_same_class)}")
    print(f"Points in other clusters or noise: {total_samples - (n_samples_first_class + len(other_same_class))}")
    print(f"Number of noise points: {np.sum(cluster_labels == -1)}")

    plt.show()

    # Print silhouette score
    print(f"Silhouette Score: {silhouette_avg:.2f}")

    return other_same_class, silhouette_avg
    

def analyze_tsne_with_svm(tsne_results, n_samples_first_class, total_samples):
    """
    Analyze t-SNE results using SVM classification.
    
    :param tsne_results: numpy array of shape (total_samples, 2) containing t-SNE results
    :param n_samples_first_class: number of samples that should belong to the first class
    :param total_samples: total number of samples
    :return: tuple containing (other_same_class, accuracy)
    """
    # Create labels: first n_samples_first_class points are class 0, rest are class 1
    labels = np.zeros(total_samples, dtype=int)
    labels[n_samples_first_class:] = 1

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tsne_results, labels, test_size=0.2, random_state=42)

    # Create and train the SVM model
    svm_model = svm.SVC(kernel='poly', degree=3, random_state=42)


    svm_model.fit(X_train, y_train)

    # Predict on the entire dataset
    predictions = svm_model.predict(tsne_results)

    # Find indices of points classified in the same class as the first n_samples_first_class
    same_class_indices = np.where(predictions == 0)[0]
    other_same_class = [idx - n_samples_first_class for idx in same_class_indices if idx > n_samples_first_class]
    print("Indices of other points classified in the same class as the first class:")
    print(other_same_class)
    # Create a mesh to plot in
    x_min, x_max = tsne_results[:, 0].min() - 1, tsne_results[:, 0].max() + 1
    y_min, y_max = tsne_results[:, 1].min() - 1, tsne_results[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Obtain labels for each point in mesh
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

    # Plot all points
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=predictions, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Highlight the first n_samples_first_class points
    plt.scatter(tsne_results[:n_samples_first_class, 0], tsne_results[:n_samples_first_class, 1], 
                c='green', s=100, label=f'First {n_samples_first_class} points', edgecolor='black')

    # Highlight the other points classified in the same class
    plt.scatter(tsne_results[same_class_indices[same_class_indices >= n_samples_first_class], 0], 
                tsne_results[same_class_indices[same_class_indices >= n_samples_first_class], 1],
                c='yellow', s=100, label='Other points in same class', edgecolor='black')

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('SVM Classification of t-SNE results')
    plt.colorbar(scatter)
    plt.legend()

    # Add annotations for class sizes
    print(f"Points in class 0: {other_same_class}")
    # print(f"Points in class 1: {total_samples - n_samples_first_class}")

    plt.show()

    # Calculate and return the accuracy
    accuracy = svm_model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    return same_class_indices, accuracy


def analyze_tsne_with_svm_old(tsne_results, n_samples_first_class, total_samples):
    """
    Analyze t-SNE results using SVM classification.
    
    :param tsne_results: numpy array of shape (total_samples, 2) containing t-SNE results
    :param n_samples_first_class: number of samples that should belong to the first class
    :param total_samples: total number of samples
    :return: tuple containing (other_same_class, accuracy)
    """
    # Create labels: first n_samples_first_class points are class 0, rest are class 1
    labels = np.ones(total_samples, dtype=int)
    labels[:n_samples_first_class] = 0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tsne_results, labels, test_size=0.2, random_state=42, stratify=labels)

    # Create and train the SVM model
    svm_model = svm.SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict on the entire dataset
    predictions = svm_model.predict(tsne_results)

    # Count the number of points in each class
    class_counts = np.bincount(predictions)
    print(f"Points in class 0: {class_counts[0]}")
    print(f"Points in class 1: {class_counts[1]}")

    # Check if all first n_samples_first_class points are classified as expected
    first_class_predictions = predictions[:n_samples_first_class]
    if np.all(first_class_predictions == 0):
        print(f"All first {n_samples_first_class} points are classified as class 0 as expected.")
    else:
        misclassified = np.sum(first_class_predictions != 0)
        print(f"Warning: {misclassified} out of the first {n_samples_first_class} points were not classified as class 0.")

    # Find indices of other points classified in the same class as the first n_samples_first_class
    same_class_indices = np.where(predictions == 0)[0]
    other_same_class = [idx - n_samples_first_class for idx in same_class_indices if idx > n_samples_first_class]
    print("Indices of other points classified in the same class as the first class:")
    print(other_same_class)

    # Create a mesh to plot in
    x_min, x_max = tsne_results[:, 0].min() - 1, tsne_results[:, 0].max() + 1
    y_min, y_max = tsne_results[:, 1].min() - 1, tsne_results[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Obtain labels for each point in mesh
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

    # Plot all points
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=predictions, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Highlight the first n_samples_first_class points
    plt.scatter(tsne_results[:n_samples_first_class, 0], tsne_results[:n_samples_first_class, 1], 
                c='green', s=100, label=f'First {n_samples_first_class} points', edgecolor='black')

    # Highlight the other points classified in the same class
    plt.scatter(tsne_results[other_same_class, 0], tsne_results[other_same_class, 1], 
                c='yellow', s=100, label='Other points in same class', edgecolor='black')

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('SVM Classification of t-SNE results')
    plt.colorbar(scatter)
    plt.legend()

    # Add annotations for class sizes
    for i, count in enumerate(class_counts):
        plt.annotate(f'Class {i}: {count} points', 
                     xy=(0.05, 0.95 - i*0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     verticalalignment='top')

    plt.show()

    # Calculate and return the accuracy
    accuracy = svm_model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    return other_same_class, accuracy


def analyze_tsne_with_svm_3d(tsne_results, n_samples_first_class, total_samples):
    """
    Analyze 3D t-SNE results using SVM classification.
    
    :param tsne_results: numpy array of shape (total_samples, 3) containing 3D t-SNE results
    :param n_samples_first_class: number of samples that should belong to the first class
    :param total_samples: total number of samples
    :return: tuple containing (other_same_class, accuracy)
    """
    # Create labels: first n_samples_first_class points are class 0, rest are class 1
    labels = np.zeros(total_samples, dtype=int)
    labels[n_samples_first_class:] = 1

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tsne_results, labels, test_size=0.2, random_state=42)

    # Create and train the SVM model
    svm_model = svm.SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict on the entire dataset
    predictions = svm_model.predict(tsne_results)

    # Find indices of points classified in the same class as the first n_samples_first_class
    same_class_indices = np.where(predictions == 0)[0]

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], 
                         c=predictions, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Highlight the first n_samples_first_class points
    ax.scatter(tsne_results[:n_samples_first_class, 0], 
               tsne_results[:n_samples_first_class, 1], 
               tsne_results[:n_samples_first_class, 2], 
               c='green', s=100, label=f'First {n_samples_first_class} points', edgecolor='black')

    # Highlight the other points classified in the same class
    other_same_class = same_class_indices[same_class_indices >= n_samples_first_class]
    ax.scatter(tsne_results[other_same_class, 0], 
               tsne_results[other_same_class, 1],
               tsne_results[other_same_class, 2],
               c='yellow', s=100, label='Other points in same class', edgecolor='black')

    ax.set_xlabel('t-SNE feature 1')
    ax.set_ylabel('t-SNE feature 2')
    ax.set_zlabel('t-SNE feature 3')
    ax.set_title('SVM Classification of 3D t-SNE results')
    plt.colorbar(scatter)
    ax.legend()

    # Add annotations for class sizes
    print(f"Points in class 0: {np.sum(predictions == 0)}")
    print(f"Points in class 1: {np.sum(predictions == 1)}")

    plt.show()

    # Calculate and return the accuracy
    accuracy = svm_model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    return same_class_indices, accuracy
    

def analyze_tsne_with_kmeans(tsne_results, n_samples_first_class, total_samples, n_clusters=2):
    """
    Analyze t-SNE results using K-Means clustering.
    
    :param tsne_results: numpy array of shape (total_samples, 2) containing t-SNE results
    :param n_samples_first_class: number of samples that should belong to the first class
    :param total_samples: total number of samples
    :param n_clusters: number of clusters to use in K-Means
    :return: tuple containing (other_same_class, cluster_labels)
    """
    # Convert tsne_results to double precision
    tsne_results = tsne_results.astype(np.float64)

    # Create labels: first n_samples_first_class points are class 0, rest are randomly assigned
    labels = np.zeros(total_samples, dtype=int)
    labels[n_samples_first_class:] = np.random.choice(np.arange(1, n_clusters), size=total_samples - n_samples_first_class)

    # Create and fit the K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tsne_results)

    # Count the number of points in each cluster
    class_counts = np.bincount(cluster_labels)
    print(f"Points in cluster 0: {class_counts[0]}")
    for i in range(1, n_clusters):
        print(f"Points in cluster {i}: {class_counts[i]}")

    # Check if all first n_samples_first_class points are classified as expected
    first_class_labels = cluster_labels[:n_samples_first_class]
    if np.all(first_class_labels == 0):
        print(f"All first {n_samples_first_class} points are classified in cluster 0 as expected.")
    else:
        misclassified = np.sum(first_class_labels != 0)
        print(f"Warning: {misclassified} out of the first {n_samples_first_class} points were not classified in cluster 0.")

    # Find indices of other points classified in the same cluster as the first n_samples_first_class
    same_class_indices = np.where(cluster_labels == 0)[0]
    other_same_class = same_class_indices[same_class_indices >= n_samples_first_class]
    print("Indices of other points classified in the same cluster as the first class:")
    print(other_same_class)

    # Create a mesh to plot in
    x_min, x_max = tsne_results[:, 0].min() - 1, tsne_results[:, 0].max() + 1
    y_min, y_max = tsne_results[:, 1].min() - 1, tsne_results[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Obtain labels for each point in mesh
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

    # Plot all points
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Highlight the first n_samples_first_class points
    plt.scatter(tsne_results[:n_samples_first_class, 0], tsne_results[:n_samples_first_class, 1], 
                c='green', s=100, label=f'First {n_samples_first_class} points', edgecolor='black')

    # Highlight the other points classified in the same cluster
    plt.scatter(tsne_results[other_same_class, 0], tsne_results[other_same_class, 1], 
                c='yellow', s=100, label='Other points in same cluster', edgecolor='black')

    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title(f'K-Means Clustering of t-SNE results ({n_clusters} clusters)')
    plt.colorbar(scatter)
    plt.legend()

    # Add annotations for cluster sizes
    for i, count in enumerate(class_counts):
        plt.annotate(f'Cluster {i}: {count} points', 
                     xy=(0.05, 0.95 - i*0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     verticalalignment='top')

    plt.show()

    return other_same_class, cluster_labels