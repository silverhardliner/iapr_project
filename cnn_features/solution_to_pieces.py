from created_dataset import split_image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_solution_pieces(solution_index):
    path = Path(__file__).parent.parent / "data_project" / "train2_solutions"
    all_pieces = []
    
    # First collect all pieces
    for file in path.glob("*.png"):
        file_name = file.name
        parts = file_name.split("_")
        
        if int(parts[1]) == solution_index:
            print(f"Processing {file_name}")
            img = cv2.imread(str(file))
            if img is None:
                print(f"Failed to load image: {file}")
                continue
                
            # Split image into pieces
            pieces, positions = split_image(img)
            all_pieces.extend(pieces)
            
    return all_pieces

def plot_solution_images(solution_index):
    all_pieces = get_solution_pieces(solution_index)
            
    # Plot all pieces
    if all_pieces:
        total_pieces = len(all_pieces)
        rows = int(np.sqrt(total_pieces)) + 1
        cols = (total_pieces + rows - 1) // rows
        
        fig = plt.figure(figsize=(20, 20))
        for idx, piece in enumerate(all_pieces, 1):
            plt.subplot(rows, cols, idx)
            plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
            plt.title(f"Piece {idx}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    solution_index = 0
    plot_solution_images(solution_index)
