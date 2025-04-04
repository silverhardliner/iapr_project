# Standard library imports
import json
from pathlib import Path
import itertools

# Third party imports
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from sklearn.cluster import KMeans

# Local imports
from cnn_features import extract_features, get_solution_features
from created_dataset import get_pieces_per_image
from solution_to_pieces import get_solution_pieces

VALID_CLUSTER_SIZES = {9, 12, 16}

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
    print(f"Mean feature values: mean={np.mean(all_features_pca):.3f}, std={np.std(all_features_pca):.3f}")
    
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

    # Print true vs predicted label distribution
    print("\nLabel Distribution:")
    print("True Label -> Predicted Labels (counts):")
    
    # Create a mapping of true labels to their predicted labels
    true_to_pred = {}
    for true, pred in zip(true_labels, predicted_labels):
        if true not in true_to_pred:
            true_to_pred[true] = {}
        pred_count = true_to_pred[true].get(pred, 0)
        true_to_pred[true][pred] = pred_count + 1
    
    # Print the mapping in a readable format
    for true_label in sorted(set(true_labels)):
        pred_counts = true_to_pred[true_label]
        pred_str = ", ".join([f"cluster {pred}:{count}" for pred, count in sorted(pred_counts.items())])
        print(f"Image {true_label:2d} -> {pred_str}")
    
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

def plot_2d_clusters(features_scaled, predicted_labels, pca=None, true_labels=None, centers=None):
    """Plot 2D clustering results using PCA if needed"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert features to 2D based on number of dimensions
    if features_scaled.shape[1] == 1:
        # For 1D data, plot feature values on x-axis and zeros on y-axis
        features_2d = np.column_stack((features_scaled, np.zeros_like(features_scaled)))
    elif features_scaled.shape[1] == 2:
        # For 2D data, use features directly
        features_2d = features_scaled
    else:
        # For 3D+ data, use PCA to reduce to 2D
        if pca is None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_scaled)
        else:
            # Use first 2 components of existing PCA
            features_2d = pca.transform(features_scaled)[:, :2]
    
    # Plot predicted clusters
    unique_labels = np.unique(predicted_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = predicted_labels == label
        if label == -1:
            # Plot noise points in black
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c='black', label='Noise', alpha=0.5, s=50)
        else:
            ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[color], label=f'Cluster {label}', alpha=0.7, s=50)
            # Plot cluster center with same color if centers provided
            if centers is not None:
                if centers.shape[1] > 2:
                    centers_2d = pca.transform(centers)[:, :2]
                else:
                    centers_2d = centers
                ax1.scatter(centers_2d[label, 0], centers_2d[label, 1],
                          c=[color], marker='x', s=200, linewidths=3)
    
    ax1.set_title('Predicted Clusters')
    if features_scaled.shape[1] == 1:
        ax1.set_xlabel('Feature Value')
        ax1.set_ylabel('Dimension 2 (zero)')
    else:
        ax1.set_xlabel('First Component')
        ax1.set_ylabel('Second Component')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot true labels if provided
    if true_labels is not None:
        unique_true_labels = np.unique(true_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_true_labels)))
        
        for label, color in zip(unique_true_labels, colors):
            mask = true_labels == label
            if label == -1:
                ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c='black', label='Noise', alpha=0.5, s=50)
            else:
                ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[color], label=f'Puzzle {label}', alpha=0.7, s=50)
        
        ax2.set_title('True Labels')
        if features_scaled.shape[1] == 1:
            ax2.set_xlabel('Feature Value')
            ax2.set_ylabel('Dimension 2 (zero)')
        else:
            ax2.set_xlabel('First Component')
            ax2.set_ylabel('Second Component')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_clustering_results(features_scaled, predicted_labels, pieces_per_image, pca=None):
    """Plot both clustering visualizations side by side"""
    fig1 = plot_image_clusters(predicted_labels, pieces_per_image)
    fig2 = plot_2d_clusters(features_scaled, predicted_labels, pca)
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

def plot_solution_clustering(solution_pieces, predicted_labels):
    """Plot solution pieces with their cluster labels"""
    # Create figure with subplots in a grid
    n_pieces = len(solution_pieces)
    n_cols = 8  # Increased number of columns
    n_rows = (n_pieces + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(12, 1.5*n_rows))  # Reduced height per row
    
    # Get unique labels for color mapping
    unique_labels = np.unique(predicted_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    for i, (piece, label) in enumerate(zip(solution_pieces, predicted_labels)):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Display the piece at full resolution
        ax.imshow(piece[...,::-1])  # Convert BGR to RGB
        
        # Add colored border based on cluster label
        if label == -1:
            color = 'black'
            label_text = 'Noise'
        else:
            color = label_to_color[label]
            label_text = f'C{label}'  # Shortened label text
            
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)  # Slightly thinner border
            
        ax.set_title(label_text, fontsize=8)  # Smaller font
        ax.axis('off')
    
    plt.tight_layout(pad=0.3)  # Reduced padding
    return fig

def calculate_clustering_fitness(n_clusters, n_noise, cluster_sizes):
    """Calculate fitness score for clustering results (lower is better)"""
    fitness = 0.0
    
    # First priority: Check number of clusters (must be 2 or 3)
    if n_clusters < 2 or n_clusters > 3:
        return 1000.0  # Immediate large penalty for wrong number of clusters
        
    # Second priority: Check valid cluster sizes (must be 9, 12, or 16)
    for size in cluster_sizes:
        if size not in VALID_CLUSTER_SIZES:
            return 500.0  # Large penalty for any invalid cluster size
            
    # Third priority: Penalize for each noise point
    fitness += n_noise * 100.0  # Penalty of 100 per noise point
    return fitness

def evaluate_solution_clustering(predicted_labels):
    """Evaluate clustering performance for the solution"""
    print("\nEvaluating clustering performance for the solution...")
    # Count number of clusters and noise points
    unique_labels = np.unique(predicted_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(predicted_labels).count(-1)
    
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of invalid noise points: {n_noise}")  # Updated message
    
    # Print size of each cluster and check if sizes match expectations
    non_compliant = []
    cluster_sizes = []
    
    for label in sorted(unique_labels):
        if label == -1:
            print(f"Invalid noise points: {n_noise}")  # Updated message
        else:
            n_points = list(predicted_labels).count(label)
            print(f"Cluster {label}: {n_points} points")
            cluster_sizes.append(n_points)
            
            if n_points not in VALID_CLUSTER_SIZES:
                non_compliant.append((label, n_points))
    
    if not non_compliant:
        print("\nAll clusters comply with expected sizes (9, 12, or 16 pieces)")
    else:
        print("\nThe following clusters do not comply with expected sizes:")
        for label, size in non_compliant:
            print(f"Cluster {label}: has {size} pieces (expected 9, 12, or 16)")
            
    # Calculate and print fitness score
    fitness = calculate_clustering_fitness(n_clusters, n_noise, cluster_sizes)
    print(f"\nFitness score (lower is better): {fitness}")
    return fitness

def load_solution_params(solution_index):
    """Load clustering parameters for a specific solution from JSON file"""
    params_path = Path(__file__).parent.parent / "results" / "clustering_parameters.json"
    with open(params_path) as f:
        params = json.load(f)
    
    solution_params = params[f"solution_{solution_index}"]
    return solution_params["eps"], solution_params["min_samples"], solution_params["n_components"]

def perform_clustering_kmeans(features_scaled):
    """Perform K-means clustering with different numbers of clusters and evaluate"""
    from sklearn.cluster import KMeans
    
    # Try different numbers of clusters
    k_range = [2, 3]  # Try 2 to 3 clusters
    best_k = None
    best_model = None
    best_score = float('inf')
    
    print("\nEvaluating K-means with different cluster counts:")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        predicted_labels = kmeans.predict(features_scaled)
        
        # Calculate score based on maximum difference between cluster sizes and valid sizes
        score = 0
        for cluster_id in range(k):
            cluster_size = np.sum(predicted_labels == cluster_id)
            # Find distance to closest valid size
            min_distance = min(abs(cluster_size - valid_size) for valid_size in VALID_CLUSTER_SIZES)
            # Update score if this distance is larger than current score
            score = max(score, min_distance)
            
        print(f"K={k}: Max size difference={score} (lower is better)")
        
        if score < best_score:
            best_score = score
            best_k = k
            best_model = kmeans
    
    print(f"\nBest K-means model:")
    print(f"Number of clusters: {best_k}")
    print(f"Max size difference: {best_score}")
    
    # Get initial predictions and centers
    predicted_labels = best_model.predict(features_scaled)
    centers = best_model.cluster_centers_
    
    # Adjust cluster sizes to match valid sizes
    for cluster_id in range(best_k):
        # Get points in this cluster
        cluster_mask = predicted_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_points = features_scaled[cluster_mask]
        
        # Find closest valid size
        valid_size = min(VALID_CLUSTER_SIZES, key=lambda x: abs(x - cluster_size))
        
        if cluster_size != valid_size:
            # Calculate distances to cluster center
            center = centers[cluster_id]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            # Sort points by distance
            sorted_indices = np.argsort(distances)
            
            if cluster_size > valid_size:
                # Need to remove points (mark as outliers)
                points_to_remove = cluster_size - valid_size
                # Get indices of points to mark as outliers
                outlier_mask = np.zeros_like(predicted_labels, dtype=bool)
                cluster_indices = np.where(cluster_mask)[0]
                outlier_indices = cluster_indices[sorted_indices[-points_to_remove:]]
                predicted_labels[outlier_indices] = -1
                print(f"Cluster {cluster_id}: Removed {points_to_remove} points to reach size {valid_size}")
            else:
                print(f"Cluster {cluster_id}: Size {cluster_size} smaller than minimum valid size")
    
    return predicted_labels, centers

def evaluate_kmeans_results(predicted_labels, true_labels):
    """Evaluate K-means clustering results against true labels"""
    # Calculate metrics
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    ami_score = adjusted_mutual_info_score(true_labels, predicted_labels)
    
    # Print evaluation results
    print("\nClustering Evaluation Results:")
    print("------------------------------")
    print(f"Number of pieces: {len(solution_pieces)}")
    
    # Print cluster sizes
    unique_pred_labels = sorted(set(predicted_labels))
    print("\nPredicted cluster sizes:")
    for label in unique_pred_labels:
        size = np.sum(predicted_labels == label)
        if label == -1:
            print(f"Noise points: {size}")
        else:
            print(f"Cluster {label}: {size} pieces")
    
    # Print true cluster sizes
    unique_true_labels = sorted(set(true_labels))
    print("\nTrue cluster sizes:")
    for label in unique_true_labels:
        size = np.sum(true_labels == label)
        if label == -1:
            print(f"Noise points: {size}")
        else:
            print(f"Puzzle {label}: {size} pieces")
    
    # Print confusion matrix-like information
    print("\nCluster assignments:")
    for true_label in unique_true_labels:
        if true_label == -1:
            continue
        true_mask = true_labels == true_label
        counts = []
        for pred_label in unique_pred_labels:
            pred_mask = predicted_labels == pred_label
            overlap = np.sum(true_mask & pred_mask)
            if overlap > 0:
                counts.append(f"C{pred_label}:{overlap}")
        print(f"Puzzle {true_label} pieces -> {', '.join(counts)}")
    
    print("\nMetrics:")
    print(f"Adjusted Rand Index: {ari_score:.3f} (1.0 is perfect clustering)")
    print(f"Adjusted Mutual Info: {ami_score:.3f} (1.0 is perfect clustering)")
    
    return true_labels

if __name__ == "__main__":
    
    #num_images_to_load = 10  # Load only first 3 images for testing
    solution_index = 1
    #eps, min_samples, n_components = load_solution_params(solution_index)
    #print("\nClustering Settings:")
    #print(f"Number of images: {num_images_to_load}")
    #print(f"PCA components:   {n_components}")
    #print(f"DBSCAN eps:       {eps}")
    #print(f"DBSCAN min_samples: {min_samples}")
    #print()

    solution_pieces, solution_labels = get_solution_pieces(solution_index)
    solution_features = get_solution_features(solution_pieces)
    # Prepare features with PCA and scaling
    n_components = 2
    all_features_scaled, pca, scaler = prepare_features(solution_features, n_components)
    print(f"all_features_scaled.shape: {all_features_scaled.shape}")
    predicted_labels, centers = perform_clustering_kmeans(all_features_scaled)
    true_labels = evaluate_kmeans_results(predicted_labels, solution_labels)
    #evaluate_solution_clustering(predicted_labels)
    plot_solution_clustering(solution_pieces, predicted_labels)
    plot_2d_clusters(all_features_scaled, predicted_labels, true_labels=solution_labels, centers=centers)
    plt.show()
    quit()
    
    pieces_per_image, labels_per_image = get_pieces_per_image(num_images_to_load)
    features_per_image = extract_features(pieces_per_image)
    
    #plot_pca_visualization(features_per_image)
    
    #features_per_image, labels_per_image, pca = load_features(num_images_to_load)
    predicted_labels = evaluate_clustering(features_per_image, labels_per_image, pieces_per_image, n_components, eps, min_samples)

    """
    For all images:
    Optimization Results:
    Best parameters found:
    eps: 1.001
    min_samples: 2    
    n_components: 16
    Best ARI score: 0.917
    """
