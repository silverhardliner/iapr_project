from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

def split_image(img, size=128):
    if img.shape[0] % size != 0 or img.shape[1] % size != 0:
        raise ValueError(f"Image size is wrong for piece size {size}x{size}.")
    
    piece_vert = img.shape[0]//size
    piece_hori = img.shape[1]//size

    pieces = []
    piece_positions = []  # Store position labels
    for i in range(piece_vert):
        for j in range(piece_hori):
            new_piece = img[i*size:(i+1)*size, j*size:(j+1)*size,:]
            pieces.append(new_piece)
            piece_positions.append((i, j))  # Position in original image

    return pieces, piece_positions

def rotate_pieces(pieces, piece_positions):
    rotated_pieces = []
    piece_labels = []  # Store position and rotation labels
    for piece, pos in zip(pieces, piece_positions):
        # Add original piece
        rotated_pieces.append(piece)
        piece_labels.append({'position': pos, 'rotation': 0})
        # Add 90 degree rotations
        for k in range(1, 4):
            rotated_pieces.append(np.rot90(piece, k))
            piece_labels.append({'position': pos, 'rotation': k * 90})
    return rotated_pieces, piece_labels

def get_data_path():
    return Path(__file__).parent.parent / "data_project" / "created_dataset"

def get_pieces_per_image(num_images=None):
    data_path = get_data_path()
    pieces_per_image = []
    labels_per_image = []  # Store labels for each image's pieces
    total_pieces = 0
    
    if not data_path.exists():
        print(f"Directory not found: {data_path}")
        return [], []
        
    for i, file in enumerate(data_path.iterdir()):
        if num_images is not None and i >= num_images:
            break
            
        if file.name.endswith('.png'):
            # Read image
            img = cv2.imread(str(file))
            if img is None:
                print(f"Failed to load image: {file}")
                continue
                
            # Split image into pieces and rotate each piece
            pieces, positions = split_image(img)
            rotated_pieces, piece_labels = rotate_pieces(pieces, positions)
            pieces_per_image.append(rotated_pieces)
            labels_per_image.append({
                'image_name': file.name,
                'piece_labels': piece_labels
            })
            
            print(f"Split {file.name} into {len(pieces)} pieces, created {len(rotated_pieces)} pieces with rotations")
            total_pieces += len(rotated_pieces)
    
    print(f"\nTotal number of pieces including rotations: {total_pieces}")
    return pieces_per_image, labels_per_image

def plot_pieces(pieces_per_image, labels_per_image):
    if not pieces_per_image:
        return
        
    total_pieces = sum(len(pieces) for pieces in pieces_per_image)
    
    # Plot all pieces
    plt.figure(figsize=(20, 20))
    current_idx = 1
    
    # Calculate overall grid size based on total pieces
    total_rows = int(np.sqrt(total_pieces)) + 1
    total_cols = (total_pieces + total_rows - 1) // total_rows
    
    for img_pieces, img_labels in zip(pieces_per_image, labels_per_image):
        for piece, label in zip(img_pieces, img_labels['piece_labels']):
            plt.subplot(total_rows, total_cols, current_idx)
            plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
            plt.title(f"Pos:{label['position']}\nRot:{label['rotation']}°", fontsize=8)
            plt.axis('off')
            current_idx += 1
            
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def main():
    pieces_per_image, labels_per_image = get_pieces_per_image()
    #print(len(labels_per_image[0]['piece_labels']))
    #plot_pieces(pieces_per_image, labels_per_image)

if __name__ == "__main__":
    main()
