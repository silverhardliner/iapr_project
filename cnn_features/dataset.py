"""Dataset creation"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_data_paths():
    """Get paths to data directories"""
    data_path = Path(__file__).parent.parent / "data_project"
    solutions_path = Path(data_path, "train_solutions")
    solutions2_path = Path(data_path, "train2_solutions")
    return solutions_path, solutions2_path

def images_are_similar(img1, img2):
    """Helper function to compare images using numpy array comparison"""
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Compare arrays directly
    return np.array_equal(img1, img2)

def collect_images(solutions_path, solutions2_path, image_prefix="solution"):
    """Collect all images from both paths"""
    all_images = []  # Temporary list to store all images before deduplication
    all_titles = []  # Temporary list to store all titles
    
    for solution_path in [solutions_path, solutions2_path]:
        for file in solution_path.iterdir():
            if file.name.startswith(image_prefix):
                image = cv2.imread(str(file))
                if image is None:
                    print(f"Failed to load image: {file}")
                    continue
                all_images.append(image)
                title = f"{solution_path.name}/{file.name}"
                all_titles.append(title)
                
    return all_images, all_titles

def find_unique_images(all_images, all_titles):
    """Find and remove duplicate images"""
    images = []
    image_titles = []
    duplicate_indices = set()
    
    for i in range(len(all_images)):
        if i in duplicate_indices:  # Skip if already marked as duplicate
            continue
            
        for j in range(i + 1, len(all_images)):
            if j not in duplicate_indices and images_are_similar(all_images[i], all_images[j]):
                print(f"Found similar images: {all_titles[j]} matches with {all_titles[i]}")
                duplicate_indices.add(j)  # Mark the duplicate
                
        # Add non-duplicate images to final lists
        if i not in duplicate_indices:
            images.append(all_images[i])
            image_titles.append(all_titles[i])
            
    return images, image_titles, duplicate_indices

def plot_images(images, image_titles, duplicate_indices):
    """Plot all collected images in a grid"""
    num_images = len(images)
    rows = int(np.sqrt(num_images))
    cols = (num_images + rows - 1) // rows  # Ceiling division

    plt.figure(figsize=(15, 15))
    for idx, (img, title) in enumerate(zip(images, image_titles)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if idx in duplicate_indices:
            plt.title(f"{title}\nShape: {img.shape}\n(has duplicate)", fontsize=8, color='red')
        else:
            plt.title(f"{title}\nShape: {img.shape}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the image processing pipeline"""
    solutions_path, solutions2_path = get_data_paths()
    all_images, all_titles = collect_images(solutions_path, solutions2_path)
    images, image_titles, duplicate_indices = find_unique_images(all_images, all_titles)
    plot_images(images, image_titles, duplicate_indices)

if __name__ == "__main__":
    main()
