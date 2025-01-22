# Standard library imports
import json
from pathlib import Path

# Third party imports
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Local imports
from cnn_features import extract_features
from created_dataset import get_pieces_per_image

def get_features_dir():
    """Get the path to the features directory"""
    return Path(__file__).parent.parent / "data_project" / "features"

def load_feature_files(features_dir, num_images=None):
    """Get list of feature files to process"""
    feature_files = list(features_dir.glob("*_features.json"))
    if num_images is not None:
        feature_files = feature_files[:num_images]
    return feature_files

def extract_piece_data(piece, image_name):
    """Extract features and label data for a single piece"""
    features = np.array(piece['features'])
    label = {
        'image_name': image_name,
        'position': piece['position'],
        'rotation': piece['rotation']
    }
    return features, label

def process_image_file(feature_file):
    """Process a single image feature file"""
    with open(feature_file) as f:
        data = json.load(f)
        
    image_features = []
    image_labels = []
    for piece in data['pieces']:
        features, label = extract_piece_data(piece, data['image_name'])
        image_features.append(features)
        image_labels.append(label)
        
    return np.array(image_features), image_labels

def load_features(num_images=None, n_components=50):
    """Load features from JSON files and reduce dimensionality with PCA"""
    features_dir = get_features_dir()
    
    features_per_image = []
    labels_per_image = []
    all_features = []
    
    print("\nLoading features from JSON files...")
    feature_files = load_feature_files(features_dir, num_images)
        
    # First pass: collect all features
    for feature_file in tqdm(feature_files, desc="Loading features"):
        image_features, image_labels = process_image_file(feature_file)
        features_per_image.append(image_features)
        labels_per_image.append(image_labels)
        all_features.extend(image_features)
            
    # Convert to numpy array and apply PCA
    all_features = np.array(all_features)
    print(f"\nTotal feature array shape before PCA: {all_features.shape}")
    
    pca = PCA(n_components=n_components)
    all_features_reduced = pca.fit_transform(all_features)
    print(f"Total feature array shape after PCA: {all_features_reduced.shape}")
    print(f"Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # Split reduced features back into per-image arrays
    start_idx = 0
    features_per_image_reduced = []
    for features in features_per_image:
        end_idx = start_idx + len(features)
        features_per_image_reduced.append(all_features_reduced[start_idx:end_idx])
        start_idx = end_idx
    
    print(f"\nLoaded and processed {len(features_per_image_reduced)} images")
    return features_per_image_reduced, labels_per_image, pca

def print_piece_details(feature, label, piece_idx):
    """Print details for a single piece"""
    print(f"\nPiece {piece_idx}:")
    print(f"  Image name: {label['image_name']}")
    print(f"  Position: {label['position']}")
    print(f"  Rotation: {label['rotation']}°")
    print(f"  Feature vector shape: {feature.shape}")

def print_image_details(features, labels, image_idx):
    """Print details for a single image"""
    print(f"\nImage {image_idx}:")
    print(f"Features shape: {features.shape}")
    print(f"Number of pieces: {len(labels)}")
    print("\nPiece details:")
    for j, (feature, label) in enumerate(zip(features, labels)):
        print_piece_details(feature, label, j)

def prepare_features(features_per_image, n_components):
    """Combine features and apply PCA and scaling"""
    all_features = np.vstack(features_per_image)
    
    # Apply PCA first
    pca = PCA(n_components=n_components)
    all_features_pca = pca.fit_transform(all_features)
    print(f"Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # Then apply standard scaling
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features_pca)

    print(f"Scaled feature statistics:")
    print(f"Mean feature values: mean={np.mean(all_features_scaled):.3f}, std={np.std(all_features_scaled):.3f}")
    
    return all_features_scaled, pca, scaler

def create_true_labels(features_per_image):
    """Create true labels based on source image index"""
    true_labels = []
    for i in range(len(features_per_image)):
        true_labels.extend([i] * len(features_per_image[i]))
    return np.array(true_labels)

def perform_clustering(features_scaled, eps, min_samples):
    """Perform DBSCAN clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    predicted_labels = dbscan.fit_predict(features_scaled)
    
    # Pretty print cluster information
    unique_labels = set(predicted_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(predicted_labels).count(-1)
    
    print("\nDBSCAN Clustering Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    for label in sorted(unique_labels):
        if label == -1:
            print(f"\nNoise points: {n_noise}")
        else:
            n_points = list(predicted_labels).count(label)
            print(f"\nCluster {label}:")
            print(f"  Number of points: {n_points}")
            
    return predicted_labels

def print_clustering_metrics(predicted_labels, true_labels, features_per_image):
    """Print clustering evaluation metrics"""
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    ami_score = adjusted_mutual_info_score(true_labels, predicted_labels)
    
    print(f"\nClustering Results:")
    print(f"Number of true image sources: {len(features_per_image)}")
    print(f"Number of clusters found: {len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)}")
    print(f"Number of noise points: {sum(1 for label in predicted_labels if label == -1)}")
    print(f"Adjusted Rand Index: {ari_score:.3f} (1.0 means perfect clustering)")
    print(f"Adjusted Mutual Information: {ami_score:.3f} (1.0 means perfect clustering)")

def plot_image_clusters(predicted_labels, pieces_per_image):
    """Plot clustering results"""
    all_pieces = []
    for pieces in pieces_per_image:
        all_pieces.extend(pieces)
        
    unique_clusters = sorted(set(predicted_labels))
    num_clusters = len(unique_clusters)
    
    # Calculate number of rows and columns for a more square layout
    num_cols = min(4, num_clusters)
    num_rows = (num_clusters + num_cols - 1) // num_cols
    
    fig = plt.figure(figsize=(3*num_cols, 3*num_rows))
    
    for i, cluster_id in enumerate(unique_clusters):
        plt.subplot(num_rows, num_cols, i + 1)
        cluster_pieces = [piece for j, piece in enumerate(all_pieces) if predicted_labels[j] == cluster_id]
        
        if cluster_pieces:
            # Limit number of pieces shown per cluster to avoid memory issues
            max_pieces = min(100, len(cluster_pieces))
            reduced_cluster_pieces = cluster_pieces[:max_pieces]
            
            grid_size = int(np.ceil(np.sqrt(max_pieces)))
            piece_height = cluster_pieces[0].shape[0] // 2
            piece_width = cluster_pieces[0].shape[1] // 2
            cluster_grid = np.zeros((grid_size * piece_height,
                                   grid_size * piece_width, 3))
            
            for idx, piece in enumerate(reduced_cluster_pieces):
                row = idx // grid_size
                col = idx % grid_size
                h_start = row * piece_height
                h_end = (row + 1) * piece_height
                w_start = col * piece_width
                w_end = (col + 1) * piece_width
                # Resize piece while preserving aspect ratio
                resized_piece = cv2.resize(piece, (piece_width, piece_height))
                cluster_grid[h_start:h_end, w_start:w_end] = resized_piece
                
            plt.imshow(cv2.cvtColor(cluster_grid.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.title(f"Cluster {cluster_id}\n({len(reduced_cluster_pieces)} pieces shown out of {len(cluster_pieces)})")
        plt.axis('off')
    
    plt.tight_layout()
    return fig

def plot_2d_clusters(features_scaled, predicted_labels):
    """Plot 2D clustering results if features are 2-dimensional"""
    if features_scaled.shape[1] != 2:
        print("Cannot plot 2D visualization - features are not 2-dimensional")
        return None
        
    fig = plt.figure(figsize=(10, 8))
    
    # Plot points for each cluster
    unique_labels = np.unique(predicted_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = predicted_labels == label
        if label == -1:
            # Plot noise points in black
            plt.scatter(features_scaled[mask, 0], features_scaled[mask, 1], 
                       c='black', label='Noise', alpha=0.5, s=50)
        else:
            plt.scatter(features_scaled[mask, 0], features_scaled[mask, 1], 
                       c=[color], label=f'Cluster {label}', alpha=0.7, s=50)
    
    plt.title('2D Visualization of Clusters')
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_clustering_results(features_scaled, predicted_labels, pieces_per_image):
    """Plot both clustering visualizations side by side"""
    fig1 = plot_image_clusters(predicted_labels, pieces_per_image)
    
    if features_scaled.shape[1] == 2:
        fig2 = plot_2d_clusters(features_scaled, predicted_labels)
        if fig2 is not None:
            plt.show()
    else:
        plt.show()

def evaluate_clustering(features_per_image, labels_per_image, pieces_per_image, n_components=2, eps=0.5, min_samples=2):
    """Evaluate clustering performance using DBSCAN"""
    print("\nEvaluating clustering performance...")
    
    # Prepare features with PCA and scaling
    all_features_scaled, pca, scaler = prepare_features(features_per_image, n_components)

    # Create true labels and perform clustering
    true_labels = create_true_labels(features_per_image)
    predicted_labels = perform_clustering(all_features_scaled, eps, min_samples)
    
    # Print metrics and plot results
    print_clustering_metrics(predicted_labels, true_labels, features_per_image)
    plot_clustering_results(all_features_scaled, predicted_labels, pieces_per_image)
    
    return predicted_labels

if __name__ == "__main__":
    num_images_to_load = 3  # Load only first 3 images for testing
    n_components = 2
    eps = 0.5
    min_samples = 2
    print("\nClustering Settings:")
    print(f"Number of images: {num_images_to_load}")
    print(f"PCA components:   {n_components}")
    print(f"DBSCAN eps:       {eps}")
    print(f"DBSCAN min_samples: {min_samples}")
    print()
    
    pieces_per_image, labels_per_image = get_pieces_per_image(num_images_to_load)
    features_per_image = extract_features(pieces_per_image)
    
    #plot_pca_visualization(features_per_image)
    
    #features_per_image, labels_per_image, pca = load_features(num_images_to_load)
    predicted_labels = evaluate_clustering(features_per_image, labels_per_image, pieces_per_image, n_components, eps, min_samples)
