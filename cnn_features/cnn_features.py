from created_dataset import get_pieces_per_image
from solution_to_pieces import get_solution_pieces
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
import numpy as np
from tqdm import tqdm
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_pca_to_features(features_per_image):
    """Apply PCA to reduce features to 2D"""
    # Combine all features into one array
    all_features = np.vstack([np.array(feat) for img_feat in features_per_image for feat in img_feat])
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    
    return features_2d

def plot_pca_visualization(features_2d, num_images=None, ax=None):
    """Plot PCA visualization of features"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
    
    # If features_2d is a list of arrays, convert to single array
    if isinstance(features_2d, list):
        features_2d = np.vstack([np.array(feat) for img_feat in features_2d for feat in img_feat])
        # Apply PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_2d)
        
    if num_images is None:
        # Just plot all points in one color
        ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
    else:
        # Create color map for different images
        colors = plt.cm.rainbow(np.linspace(0, 1, num_images))
        
        # Plot points with different colors for each image
        points_per_image = len(features_2d) // num_images
        for i in range(num_images):
            start_idx = i * points_per_image
            end_idx = (i + 1) * points_per_image
            ax.scatter(features_2d[start_idx:end_idx, 0], 
                      features_2d[start_idx:end_idx, 1],
                      c=[colors[i]], 
                      label=f'Image {i+1}',
                      alpha=0.6)
            ax.legend()
    
    ax.set_title('PCA Visualization of Image Features')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    return ax

def plot_feature_vectors(features_per_image, ax=None):
    """Plot mean feature vector comparison"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
    for i, img_features in enumerate(features_per_image):
        # Plot mean feature vector for each image
        mean_features = np.mean(np.array(img_features), axis=0)
        ax.plot(mean_features, label=f'Image {i+1} (mean)', alpha=0.8)
    ax.set_title('Mean Feature Vectors Comparison')
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Feature Value')
    ax.legend()
    return ax

def plot_feature_distribution(features_per_image, ax=None):
    """Plot distribution of feature values"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
    for i, img_features in enumerate(features_per_image):
        ax.hist(np.array(img_features).flatten(), bins=50, alpha=0.3,
               label=f'Image {i+1}', density=True)
    ax.set_title('Feature Value Distribution')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Density')
    ax.legend()
    return ax

def plot_example_pieces(features_per_image, pieces_per_image, ax=None):
    """Plot example pieces and their feature vectors"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
    colors = plt.cm.rainbow(np.linspace(0, 1, len(features_per_image)))
    num_pieces = min(3, len(pieces_per_image[0]))
    
    for i in range(num_pieces):
        for j in range(len(pieces_per_image)):
            piece = pieces_per_image[j][i]
            feature = np.array(features_per_image[j][i])
            
            # Create small subplot for piece
            inset_ax = ax.inset_axes([0.05 + j*0.3, 0.7 - i*0.3, 0.25, 0.25])
            inset_ax.imshow(piece[...,::-1])  # Convert BGR to RGB
            inset_ax.axis('off')
            inset_ax.set_title(f'Image {j+1}\nPiece {i+1}')
            
            # Plot corresponding feature vector
            inset_ax = ax.inset_axes([0.05 + j*0.3, 0.6 - i*0.3, 0.25, 0.1])
            inset_ax.plot(feature, c=colors[j])
            inset_ax.set_xticks([])
    ax.axis('off')
    ax.set_title('Example Pieces and Their Feature Vectors')
    return ax

def visualize_features(features_per_image, pieces_per_image=None):
    """Visualize features using PCA and feature vector comparisons"""
    print("\nVisualizing features...")
    
    # Create separate figures for each visualization
    pca_fig = plt.figure(figsize=(10, 8))
    vector_fig = plt.figure(figsize=(10, 8)) 
    dist_fig = plt.figure(figsize=(10, 8))
    examples_fig = plt.figure(figsize=(10, 8))
    
    # Generate each visualization
    plot_pca_visualization(features_per_image, len(features_per_image), pca_fig.add_subplot(111))
    plot_feature_vectors(features_per_image, vector_fig.add_subplot(111))
    plot_feature_distribution(features_per_image, dist_fig.add_subplot(111))
    if pieces_per_image is not None:
        plot_example_pieces(features_per_image, pieces_per_image, examples_fig.add_subplot(111))
    
    # Show all figures
    plt.show()

def get_feature_extractor():
    """Initialize and return the feature extractor model and preprocessing transforms"""
    print("\nInitializing feature extraction...")
    # Use MobileNetV2 which is one of the smallest pretrained models
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    # Remove the last fully connected layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    print("Loading preprocessing transforms...")
    # Get the preprocessing transforms directly from the model weights
    preprocess = weights.transforms()
    
    return feature_extractor, preprocess

def extract_features_from_pieces(pieces, feature_extractor, preprocess):
    """Extract features from a list of pieces using the provided model"""
    features = []
    with torch.no_grad():
        for piece in tqdm(pieces, desc="Processing pieces", leave=False):
            # Transform piece using the model's preprocessing
            # Add .copy() to handle negative strides in numpy array
            piece_tensor = preprocess(torch.from_numpy(piece.copy()).permute(2, 0, 1) / 255.0).unsqueeze(0)
            
            # Extract features
            piece_features = feature_extractor(piece_tensor)
            # Flatten features
            piece_features = piece_features.squeeze().flatten().numpy()
            features.append(piece_features)
            
    return features

def extract_features(pieces_per_image):
    """Extract features from multiple images' worth of pieces"""
    feature_extractor, preprocess = get_feature_extractor()

    features_per_image = []
    total_pieces = sum(len(pieces) for pieces in pieces_per_image)
    print(f"\nProcessing {len(pieces_per_image)} images with total {total_pieces} pieces...")
    
    start_time = time.time()
    for i, pieces in enumerate(tqdm(pieces_per_image, desc="Processing images")):
        image_features = extract_features_from_pieces(pieces, feature_extractor, preprocess)
        features_per_image.append(image_features)
    
    elapsed_time = time.time() - start_time
    print(f"\nFeature extraction completed in {elapsed_time:.2f} seconds")
    return features_per_image

def save_features(features_per_image, labels_per_image):
    output_dir = Path(__file__).parent.parent / "data_project" / "features"
    output_dir.mkdir(exist_ok=True)
    
    print("\nSaving features to JSON files...")
    for features, label_info in tqdm(zip(features_per_image, labels_per_image), desc="Saving features"):
        image_name = label_info['image_name']
        
        # Combine features with their corresponding labels
        pieces_data = []
        for feature_vector, piece_label in zip(features, label_info['piece_labels']):
            piece_data = {
                'features': feature_vector.tolist(),  # Convert numpy array to list here
                'position': piece_label['position'],
                'rotation': piece_label['rotation']
            }
            pieces_data.append(piece_data)
            
        output_data = {
            'image_name': image_name,
            'pieces': pieces_data
        }
        
        output_path = output_dir / f"{image_name.rsplit('.', 1)[0]}_features.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f)

def get_solution_features(solution_pieces):
    feature_extractor, preprocess = get_feature_extractor()
    pieces_features = extract_features_from_pieces(solution_pieces, feature_extractor, preprocess)
    return pieces_features

def main():
    solution_index = 0
    solution_pieces = get_solution_pieces(solution_index)
    solution_features = get_solution_features(solution_pieces)
    print(solution_features)
    visualize_features([solution_features], None)
    quit()
    
    print("Loading image pieces...")
    num_images_to_load = 3  # Load only first 3 images for testing
    pieces_per_image, labels_per_image = get_pieces_per_image(num_images_to_load)
    features_per_image = extract_features(pieces_per_image)
    print(f"Extracted features shape for first piece: {features_per_image[0][0].shape}")
    visualize_features(features_per_image, pieces_per_image)
    #save_features(features_per_image, labels_per_image)

if __name__ == "__main__":
    main()
