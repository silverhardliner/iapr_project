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

# Local imports
from cnn_features import extract_features, get_solution_features
from created_dataset import get_pieces_per_image
from solution_to_pieces import get_solution_pieces

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

def plot_2d_clusters(features_scaled, predicted_labels, pca=None):
    """Plot 2D clustering results using PCA if needed"""
    if features_scaled.shape[1] != 2:
        if pca is None:
            # Create new PCA if not provided
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_scaled)
        else:
            # Use first 2 components of existing PCA
            features_2d = features_scaled[:, :2]
    else:
        features_2d = features_scaled
        
    fig = plt.figure(figsize=(10, 8))
    
    # Plot points for each cluster
    unique_labels = np.unique(predicted_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = predicted_labels == label
        if label == -1:
            # Plot noise points in black
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c='black', label='Noise', alpha=0.5, s=50)
        else:
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[color], label=f'Cluster {label}', alpha=0.7, s=50)
    
    plt.title('2D Visualization of Clusters (First Two Principal Components)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
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
    valid_sizes = {9, 12, 16}
    for size in cluster_sizes:
        if size not in valid_sizes:
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
    valid_sizes = {9, 12, 16}
    non_compliant = []
    cluster_sizes = []
    
    for label in sorted(unique_labels):
        if label == -1:
            print(f"Invalid noise points: {n_noise}")  # Updated message
        else:
            n_points = list(predicted_labels).count(label)
            print(f"Cluster {label}: {n_points} points")
            cluster_sizes.append(n_points)
            
            if n_points not in valid_sizes:
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

def flexible_constrained_kmeans(X, max_noise=3, max_iters=100):
    """
    Perform K-means clustering with flexible constraints:
    - 2-3 clusters of sizes 9, 12, or 16
    - Up to max_noise points as noise
    
    Args:
        X: feature matrix (n_samples, n_features)
        max_noise: maximum number of noise points allowed (default 3)
        max_iters: maximum number of iterations
    
    Returns:
        labels: cluster assignments with -1 for noise points
        fitness: score indicating how well constraints are met (lower is better)
    """
    n_samples = X.shape[0]
    valid_sizes = {9, 12, 16}
    
    print(f"\nStarting flexible K-means clustering with {n_samples} samples")
    print(f"Max noise points allowed: {max_noise}")
    
    # Try both 2 and 3 clusters
    best_labels = None
    best_fitness = float('inf')
    
    for n_clusters in [2, 3]:
        print(f"\nTrying {n_clusters} clusters...")
        
        # Initialize centroids using k-means++
        centroids = np.zeros((n_clusters, X.shape[1]))
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Choose remaining centroids
        for k in range(1, n_clusters):
            # Compute distances to closest centroid for each point
            distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids[:k]], axis=0)
            # Choose next centroid with probability proportional to distance squared
            probs = distances / distances.sum()
            centroids[k] = X[np.random.choice(n_samples, p=probs)]
        
        # First check if any valid size combinations exist
        remaining_points = n_samples - max_noise
        valid_combinations = []
        for sizes in itertools.combinations_with_replacement(valid_sizes, n_clusters):
            if sum(sizes) == remaining_points:
                valid_combinations.append(sizes)
        
        if not valid_combinations:
            print(f"  No valid size combinations possible for {n_clusters} clusters")
            continue  # Skip to next number of clusters
            
        print(f"  Valid size combinations found: {len(valid_combinations)}")
        
        for iter_num in range(max_iters):
            # Calculate distances to centroids
            distances = euclidean_distances(X, centroids)
            
            # Find potential noise points (furthest from all centroids)
            min_distances = np.min(distances, axis=1)
            noise_candidates = np.argsort(min_distances)[-max_noise:]
            
            # Initialize labels
            labels = np.full(n_samples, -1)
            
            # Remove noise points from consideration
            valid_indices = list(set(range(n_samples)) - set(noise_candidates))
            
            # Try different valid size combinations
            valid_combinations = []
            remaining_points = len(valid_indices)
            
            print(f"  Iteration {iter_num + 1}")
            print(f"  Points after removing noise: {remaining_points}")
            
            # Generate valid size combinations
            for sizes in itertools.combinations_with_replacement(valid_sizes, n_clusters):
                if sum(sizes) == remaining_points:
                    valid_combinations.append(sizes)
            
            print(f"  Valid size combinations found: {len(valid_combinations)}")
            
            for sizes in valid_combinations:
                print(f"  Trying cluster sizes: {sizes}")
                temp_labels = np.full(n_samples, -1)
                remaining = set(valid_indices)
                
                # Assign points to clusters while respecting size constraints
                for k in range(n_clusters):
                    size = sizes[k]
                    cluster_distances = distances[:, k]
                    available = list(remaining)
                    closest = sorted(available, 
                                  key=lambda i: cluster_distances[i])[:size]
                    
                    temp_labels[closest] = k
                    remaining -= set(closest)
                
                # Calculate fitness for this arrangement
                fitness = calculate_clustering_fitness(
                    n_clusters=n_clusters,
                    n_noise=max_noise,
                    cluster_sizes=[np.sum(temp_labels == k) for k in range(n_clusters)]
                )
                
                print(f"  Fitness score: {fitness}")
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_labels = temp_labels.copy()
                    print(f"  New best fitness found: {fitness}")
            
            # Update centroids (excluding noise points)
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = best_labels == k
                if np.any(mask):  # Only update if cluster is not empty
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]  # Keep old centroid if cluster is empty
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                print("  Converged!")
                break
                
            centroids = new_centroids
    
    print(f"\nFinal best fitness: {best_fitness}")
    print(f"Final cluster sizes: {[np.sum(best_labels == k) for k in range(n_clusters)]}")
    print(f"Noise points: {np.sum(best_labels == -1)}")
    
    return best_labels, best_fitness

def perform_clustering_v2(features_scaled, eps=None, min_samples=None):
    """Updated clustering function using flexible constrained k-means"""
    predicted_labels, fitness = flexible_constrained_kmeans(features_scaled)
    
    # Pretty print cluster information
    unique_labels = set(predicted_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(predicted_labels).count(-1)
    
    print("\nClustering Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Fitness score: {fitness}")
    
    for label in sorted(unique_labels):
        if label == -1:
            print(f"\nNoise points: {n_noise}")
        else:
            n_points = list(predicted_labels).count(label)
            print(f"\nCluster {label}:")
            print(f"  Number of points: {n_points}")
            
    return predicted_labels


if __name__ == "__main__":
    
    num_images_to_load = 10  # Load only first 3 images for testing
    solution_index = 0
    eps, min_samples, n_components = load_solution_params(solution_index)
    print("\nClustering Settings:")
    print(f"Number of images: {num_images_to_load}")
    print(f"PCA components:   {n_components}")
    print(f"DBSCAN eps:       {eps}")
    print(f"DBSCAN min_samples: {min_samples}")
    print()

    solution_pieces = get_solution_pieces(solution_index)
    solution_features = get_solution_features(solution_pieces)
    # Prepare features with PCA and scaling
    n_components = 3
    all_features_scaled, pca, scaler = prepare_features(solution_features, n_components)
    print(f"all_features_scaled.shape: {all_features_scaled.shape}")
    predicted_labels = perform_clustering_v2(all_features_scaled)
    evaluate_solution_clustering(predicted_labels)
    plot_solution_clustering(solution_pieces, predicted_labels)
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
