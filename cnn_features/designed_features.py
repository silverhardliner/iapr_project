import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
# Local imports
from solution_to_pieces import get_solution_pieces

def get_mode_color(piece):
    """Calculate mode color for each channel of a piece."""
    # Flatten spatial dimensions, keep channels
    flat_piece = piece.reshape(-1, 3)
    
    # Find mode for each channel independently
    b = np.bincount(flat_piece[:,0].astype(int)).argmax()
    g = np.bincount(flat_piece[:,1].astype(int)).argmax()
    r = np.bincount(flat_piece[:,2].astype(int)).argmax()
    
    return np.array([b, g, r])

def detect_circles(piece):
    """Detect circles in the piece and return number of circles found."""
    # Convert to grayscale
    gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    
    # Detect circles with relaxed parameters
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,  # Allow very close circles
        param1=50,   # Edge detection parameter
        param2=15,   # Very relaxed circle detection
        minRadius=15,
        maxRadius=60
    )
    
    # Return number of circles found (0 if none found)
    return len(circles[0]) if circles is not None else 0

def get_edge_density(piece):
    """Calculate the density of edges in the piece."""
    # Convert to grayscale
    gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    # Calculate percentage of pixels that are edges
    edge_density = np.count_nonzero(edges) / edges.size
    
    return edge_density

def get_designed_features(pieces):
    features = []
    for piece in pieces:
        # Get mode color features
        mode_color = get_mode_color(piece)
        # Get edge density feature
        edge_density = get_edge_density(piece)
        
        # Combine features
        piece_features = np.append(mode_color, edge_density)
        features.append(piece_features)
    
    # Convert to numpy array and standardize each feature
    features = np.array(features)
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    
    return standardized_features

def plot_pieces_and_colors(pieces):
    # Calculate number of rows and columns needed for the grid
    n = len(pieces)
    grid_size = int(np.ceil(np.sqrt(n)))
    
    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(grid_size*6, grid_size*2))
    
    for i, piece in enumerate(pieces):
        # Get mode color and edge density
        mode_color = get_mode_color(piece)
        edge_density = get_edge_density(piece)
        
        # Convert to grayscale and get edges for visualization
        gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Original piece
        plt.subplot(grid_size, grid_size*3, i*3 + 1)
        plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
        plt.title(f'#{i+1}')
        plt.axis('off')
        
        # Edge detection
        plt.subplot(grid_size, grid_size*3, i*3 + 2)
        plt.imshow(edges, cmap='gray')
        plt.title(f'Edges: {edge_density:.3f}')
        plt.axis('off')
        
        # Mode color
        plt.subplot(grid_size, grid_size*3, i*3 + 3)
        color_patch = np.full((100,100,3), mode_color, dtype=np.uint8)
        plt.imshow(cv2.cvtColor(color_patch, cv2.COLOR_BGR2RGB))
        plt.title(f'Color #{i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def single_piece_analysis(piece):
    # Flatten piece and separate channels
    flat_piece = piece.reshape(-1, 3)  # Flatten spatial dimensions, keep channels
    b, g, r = flat_piece[:,0], flat_piece[:,1], flat_piece[:,2]
    
    # Create figure with 3 subplots for RGB histograms
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    
    # Plot histograms
    ax1.hist(r, bins=50, color='red', alpha=0.7)
    ax1.set_title('Red Channel Histogram')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Count')
    
    ax2.hist(g, bins=50, color='green', alpha=0.7)
    ax2.set_title('Green Channel Histogram') 
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Count')
    
    ax3.hist(b, bins=50, color='blue', alpha=0.7)
    ax3.set_title('Blue Channel Histogram')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def single_piece_circle_analysis(piece):
    """Visualize circle detection on a single piece."""
    # Convert to grayscale
    gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect circles with relaxed parameters
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=15,
        minRadius=15,
        maxRadius=60
    )
    
    # Draw detected circles
    piece_with_circles = piece.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Draw the outer circle
            cv2.circle(piece_with_circles, center, radius, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(piece_with_circles, center, 2, (0, 0, 255), 3)
    
    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
    plt.title('Original Piece')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(piece_with_circles, cv2.COLOR_BGR2RGB))
    plt.title(f'Has Circles: {circles is not None}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def single_piece_edge_analysis(piece):
    """Visualize edge detection on a single piece."""
    # Convert to grayscale
    gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = get_edge_density(piece)
    
    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
    plt.title('Original Piece')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Edge Detection\nDensity: {edge_density:.3f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pieces, labels = get_solution_pieces(9)
    #single_piece_analysis(pieces[0])
    #for piece in pieces:
    #    single_piece_circle_analysis(piece)
    print(len(pieces))
    features = get_designed_features(pieces)
    print(features)
    plot_pieces_and_colors(pieces)
