import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from scipy.stats import skew, mode, chisquare, kurtosis
from PIL import Image, ImageFilter, ImageDraw
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.morphology import closing, disk, remove_small_holes
from skimage.util import invert
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from sklearn.decomposition import PCA

class Segmentation:
    def __init__(self):
        self.name = "Segmentation"
        self.squares = None

    def load_input_image_(self, image_index, flag='train', folder="train2", path ="data_project"):
        if flag == 'solution':
            filename = f'solution_{image_index}.png'
        else:
            filename = "train_{:02}.png".format(int(image_index))

        path_solution = os.path.join(path, folder , filename)
        
        im= Image.open(path_solution).convert('RGB')
        im = np.array(im)
        return im
    
    def edge_morphology_(self, single_channel_img, gl_std, pad_width, disk_size=3, holes_size=40000):
        gauss_lap = gaussian_laplace(single_channel_img, gl_std)
        gauss_lap_pad = np.pad(gauss_lap, pad_width, mode='constant')
        closed_laplace = closing(gauss_lap_pad, disk(disk_size))
        remove_laplace = remove_small_holes(closed_laplace, holes_size).astype('float')
        return remove_laplace
    
    def remove_pad_(self, image, pad_width):
        out = image[pad_width:-pad_width, pad_width:-pad_width]
        return out
    
    def circ_filter_(self, size = 128):
        x = np.arange(0, size)
        y = np.arange(0, size)
        arr = np.zeros((y.size, x.size))

        cx = size//2.
        cy = size//2.
        r = size//2

        # The two lines below could be merged, but I stored the mask
        # for code clarity.
        mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
        arr[mask] = 1.

        return arr
    
    def find_squares_(self, img_2d):
        # Make the objects to 1 and background to 0
        magic_constant = 2
        values, counts = np.unique(img_2d, return_counts=True)
        if len(counts) > 1 and counts[0] < counts[1]:
            img_2d = invert(img_2d.astype(int)).astype(float)+magic_constant
        
        # Square filter
        square_side = 128
        the_square = np.ones((square_side, square_side))

        # Circ filter
        circ_dia = 128
        the_circle = self.circ_filter_(circ_dia)
        
        # Rotation 0 degrees
        locations = cv2.filter2D(src=img_2d, kernel=the_circle, ddepth=-1)
        return locations
    
    def norm_data_(self, data):
        """
        normalize data to have mean=0 and standard_deviation=1
        """
        mean_data=np.mean(data)
        std_data=np.std(data, ddof=1)
        #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
        return (data-mean_data)/(std_data)
    
    def find_biggest_neighbor_(self, image, loc):
        ret = loc
        new_max = image[loc[0], loc[1]]
        loc_add = np.array([(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)])
        for i in range(len(loc_add)):
            new_loc = loc + loc_add[i]
            if 0 <= new_loc[0] < image.shape[0] and 0 <= new_loc[1] < image.shape[1]:
                if image[new_loc[0], new_loc[1]] > new_max:
                    new_max = image[new_loc[0], new_loc[1]]
                    ret = new_loc

        return ret
    
    def close_to_border_(self, peak, small_sq_size, big_im_size):
        keep = True
        sq_half = small_sq_size//2
        cond1 = peak[0] <= sq_half
        cond2 = peak[1] <= sq_half
        cond3 = peak[0] >= big_im_size[0] - sq_half
        cond4 = peak[1] >= big_im_size[1] - sq_half
        if cond1 or cond2 or cond3 or cond4:
            keep = False

        return keep
    
    def travel_to_peaks_(self, norm_loc, small_sq_size = 128):
        norm_loc_comp = np.mean(norm_loc)
        #norm_loc_comp = np.max(norm_loc)*0.6
        peaks = []
        for i in range(100, norm_loc.shape[0], 100):
            for j in range(100, norm_loc.shape[1], 100):
                loc = np.array([i,j])
                new_loc = loc
                
                k = 0
                flag = True
                while(flag):
                    new_loc = self.find_biggest_neighbor_(norm_loc, new_loc)
                    if new_loc[0] == loc[0] and new_loc[1] == loc[1]:
                        flag = False
                    else:
                        loc = new_loc
                    k+=1
                    if k%100 == 0:
                        pass
                value = norm_loc[new_loc[0], new_loc[1]]
                if k > 1 and value > norm_loc_comp:
                    keep = self.close_to_border_(new_loc, small_sq_size, norm_loc.shape)
                    if keep:
                        peaks.append(new_loc)

        peaks = np.array(peaks)
        unique_peaks = np.unique(peaks, axis=0)
        return unique_peaks
    
    def my_dist_(self, line, point):
        line = np.asarray(line)
        point = np.asarray(point)
        new_point = np.array([point[1], point[0]])
        p0, p1 = line
        line_centr = (p0+p1)/2
        d = np.linalg.norm(line_centr-new_point)
        return d
    
    def get_angles_(self):
        num = -1
        for sq in self.squares:
            num += 1
            angles = []
            for ln in sq.lines:
                one_angle = self.line_angle_(ln, idx=num)
                angles.append(one_angle)
            # Remove one element to fix median calculation
            if len(angles) % 2 == 0:
                angles.pop()
            sq.angle = np.median(angles)

    def line_angle_(self, line, idx=None):
        p0 = line[0]
        p1 = line[1]
        d0 = p0[0] - p1[0]
        d1 = p0[1] - p1[1]
        angle_rad = np.arctan2(d1,d0)
        
        if angle_rad < 0:
            angle_rad += np.pi
        if angle_rad > np.pi/2:
            angle_rad -= np.pi/2

        return angle_rad

    def cluster_lines_(self, centroids, lines, debug=False):
        
        for i in range(len(lines)):
            index = 0
            d_min = float('inf')
            if debug:
                print(f"Line {lines[i]}")
            for j in range(len(centroids)):
                if debug:
                    print(f"Centroid {centroids[j]}")
                one_line = lines[i]
                one_center = centroids[j]
                d = self.my_dist_(one_line, one_center)
                if debug:
                    print(f"Distance {d}")
                if d < d_min:
                    if debug:
                        print(f"Yes")
                    d_min = d
                    index = j
                else:
                    if debug:
                        print("No")

            self.squares[index].lines.append(lines[i])

    def get_square_points_(self, sq, angle=0, side_lenght = 128):

        h = side_lenght / np.sqrt(2)  # half-diagonal length
        cx = sq.centroid[1]
        cy = sq.centroid[0]
        
        # Calculate the coordinates of the four points
        dx = h * np.cos(angle + np.pi / 4)
        dy = h * np.sin(angle + np.pi / 4)

        point1 = (cx - dx, cy - dy)
        point2 = (cx + dy, cy - dx)
        point3 = (cx + dx, cy + dy)
        point4 = (cx - dy, cy + dx)

        return np.array([point1, point2, point3, point4])
    
    def get_squares_(self, centroids, lines):
        self.squares = [self.Square(centroids[i]) for i in range(len(centroids))]
        
        self.cluster_lines_(centroids, lines, False)
        self.get_angles_()
        for sq in self.squares:
            sq._square_points = self.get_square_points_(sq, angle=sq.angle)

    def create_perfect_mask_(self, size=2000):
        output = np.zeros((size,size))
        for sq in self.squares:
            square_points = np.array([sq.square_points], dtype=np.int32)
            cv2.fillPoly(output, pts = [square_points], color=(1))
        return output
    
    def edge_image_fix_(self, image, flag='edge'):

        edge = 3

        # Split the image into individual color channels
        channels = cv2.split(image)

        if flag == 'inpaint':
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[:edge, :] = 1
            mask[-edge:, :] = 1
            mask[:, :edge] = 1
            mask[:,-edge:] = 1

            # Apply inpainting to each color channel separately
            inpainted_channels = []
            for channel in channels:
                # Apply inpainting to the channel
                inpainted_channel = cv2.inpaint(channel, mask, 3, cv2.INPAINT_TELEA)
                inpainted_channels.append(inpainted_channel)

            # Merge the inpainted color channels back into an RGB image
            out = cv2.merge(inpainted_channels)
        else:
            mirrored_channels = []
            for channel in channels:
                cropped_channel = channel[edge:-edge, edge:-edge]
                padded_channels = np.pad(cropped_channel, edge, mode='edge')
                mirrored_channels.append(padded_channels)
            out = cv2.merge(mirrored_channels)
            #print(out.shape)

        return out

    def crop_square_(self, image_path, points, angle, save_path=None, image_size=(128,128)):
        # Open the image
        image = Image.open(image_path)

        # Create a new image with an RGBA mode and transparent background
        cropped_image = Image.new("RGBA", image.size, (0, 0, 0, 0))

        # Create a mask based on the given points
        mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
        mask_points = [tuple(point) for point in points]
        
        mask_points.append(mask_points[0])
        ImageDraw.Draw(mask).polygon(mask_points, outline=(255, 255, 255, 255), fill=(255, 255, 255, 255))

        # Apply the mask to the original image
        cropped_image.paste(image, mask=mask)

        # Crop the image to the minimum bounding rectangle
        cropped_image = cropped_image.crop(mask.getbbox())

        # Rotate image
        cropped_image = cropped_image.rotate(np.rad2deg(angle))

        # Find bounding box
        image_gray = cropped_image.convert("L")

        # Find the bounding box of the non-white pixels
        bbox = image_gray.getbbox()

        # Crop
        trimmed_image = cropped_image.crop(bbox)

        # Resize to 128x128
        output_image = trimmed_image.resize(image_size)

        # Inpaint edges to get rid of background artefacts
        output_image = self.edge_image_fix_(np.array(output_image), flag='edge')
        #print(edge_fixed.dtype)

        if save_path:
            # Save the inpainted image
            Image.fromarray(output_image).save(save_path)
            #trimmed_image.save(save_path)

        return output_image
    
    def plot_images(images, title="Cropped images", cmap=None):
        num_images = len(images)
        rows = (num_images + 7) // 8  # Calculate the number of rows based on the number of images

        fig, axes = plt.subplots(rows, 8, figsize=(12, rows * 1.5))
        fig.suptitle(title)

        # Turn off the axes for all subplots
        for ax in axes.flatten():
            ax.axis("off")

        # Plot the images
        for i, image in enumerate(images):
            row_idx = i // 8  # Row index
            col_idx = i % 8   # Column index

            # Plot the image in the corresponding subplot
            axes[row_idx, col_idx].imshow(image, cmap=cmap)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        plt.show()
    
    def squares_from_image(self, input_num):
        # Load images
        img = self.load_input_image_(input_num)#[100:380,:350]
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        im_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        value = im_hsv[:,:,2]

        # Padding, edge and morphology
        pad_width = 50
        std_gray = 0.8
        gray_mask = self.edge_morphology_(img_gray, std_gray, pad_width)
        std_value = 1
        value_mask = self.edge_morphology_(value, std_value, pad_width)
        ultimate_combine = np.logical_or(gray_mask, value_mask).astype(float)
        
        #Combine unpad
        combine_no_pad = self.remove_pad_(ultimate_combine, pad_width)
        
        # Square locations
        locations = self.find_squares_(combine_no_pad)

        # Find peaks
        smooth_std = 10
        smooth_loc= gaussian_filter(locations, smooth_std)
        locations_norm = self.norm_data_(smooth_loc)
        peaks = self.travel_to_peaks_(locations_norm)
        
        # Finding lines
        canny_std = 3
        edge_close_th = canny(combine_no_pad, canny_std)
        lines = probabilistic_hough_line(edge_close_th, threshold=10, line_length=80, line_gap=20)

        # Making square objects
        self.get_squares_(peaks, lines)

        '''FIRST OUTPUT'''
        perfect_mask = self.create_perfect_mask_()

        # Cutting out images
        output_squares = []
        whole_im_path = f"data_project/train2/train_{input_num:02}.png"
        for i in range(len(self.squares)):
            sq = self.squares[i]
            out_path = f"output/output_{input_num:02}_{i:02}.png"
            cropped_image = self.crop_square_(whole_im_path, sq.square_points, sq.angle)
            output_squares.append(cropped_image)
        
        return output_squares, perfect_mask
    
    class Square:

        def __init__(self, centroid):
            self._centroid = np.asarray(centroid)
            self._lines = []
            self._angle = 0
            self._square_points = []

        # Centroid
        @property
        def centroid(self):
            """Returns cenntroid of square."""
            return self._centroid

        @centroid.setter
        def centroid(self, centroid):
            """Sets centroid of card."""
            self._centroid = np.asarray(centroid)

        # Lines
        @property
        def lines(self):
            return self._lines
        
        @lines.setter
        def lines(self, lines):
            self._lines = np.asarray(lines)

        # Angle
        @property
        def angle(self):
            return self._angle
        
        @angle.setter
        def angle(self, angle):
            self._angle = angle

        # Square points
        @property
        def square_points(self):
            return self._square_points
        
        @square_points.setter
        def square_points(self, square_points):
            self._square_points = square_points

    
    
    
class Classification:
    def __init__(self):
        pass
    
    def generate_feature_vectors_(self, pieces, n_features):
        idx_piece = 0
        idx_feature = 0
        features_vectors = np.zeros((len(pieces), n_features))

        for i in range(len(pieces)):
            features_vectors[idx_piece, :] = self.extract_feature_of_interest_(pieces[i], idx_piece, n_features)
            idx_piece += 1
            
        return features_vectors
        
    def extract_feature_of_interest_(self,patch, idx_piece, n_features):
        """
        Extract the feature of interest from the given patch image.

        Parameters:
            patch (numpy.ndarray): The patch image to extract the feature from.
            idx_piece (int): The index of the puzzle piece.
            n_features (int): The number of features to extract.
        Returns:
            numpy.ndarray: The flattened feature vector.
        """

        features_vector = np.zeros((n_features, 1))
    
        rgb_piece = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        hsv_piece = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        idx_feature = 0

        ksize = 5
        phi = 0

        for theta in np.linspace(0, np.pi, 5, endpoint = False):
            for sigma in np.arange(0.5, 3, 0.5): #range(1, 9, 2):
                for lamda in np.arange(0.1, 0.5, 0.1): #np.linspace(0.05, 0.5, 5)
                    for gamma in np.arange(0.5, 1, 0.5):

                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype = cv2.CV_32F)
                        fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                        fimgr = cv2.filter2D(rgb_piece[:,:,0], cv2.CV_8UC3, kernel)
                        fimgg = cv2.filter2D(rgb_piece[:,:,1], cv2.CV_8UC3, kernel)
                        fimgb = cv2.filter2D(rgb_piece[:,:,2], cv2.CV_8UC3, kernel)

                        features_vector[idx_feature] = np.mean(fimg)
                        features_vector[idx_feature+1] = np.mean(fimgr)
                        features_vector[idx_feature+2] = np.mean(fimgg)
                        features_vector[idx_feature+3] = np.mean(fimgb)

                        features_vector[idx_feature+4] = skew(fimg.flatten())
                        features_vector[idx_feature+5] = skew(fimgr.flatten())
                        features_vector[idx_feature+6] = skew(fimgg.flatten())
                        features_vector[idx_feature+7] = skew(fimgb.flatten())

                        features_vector[idx_feature+8] = mode(fimg.flatten())[0]

                        features_vector[idx_feature+9] = np.nan_to_num(chisquare(fimg.flatten())[0])
                        features_vector[idx_feature+10] = np.nan_to_num(kurtosis(fimg.flatten()))
                        features_vector[idx_feature+11] = np.std(fimg)

                        idx_feature += 12

        return features_vector.flatten()

    
    def normalize_features_(self, features_vectors):
        """
        Normalize the feature vectors.

        Parameters:
            features_vectors (numpy.ndarray): The feature vectors to be normalized.
        Returns:
            numpy.ndarray: The normalized feature vectors.
        """

        n_features = features_vectors.shape[1]  # Number of features in each vector

        for i in range(n_features):
            min_ = np.min(features_vectors[:, i])  # Minimum value of the current feature
            max_ = np.max(features_vectors[:, i])  # Maximum value of the current feature

            if (max_ - min_) != 0:
                # Normalize the feature vector if the range is non-zero
                normalized_vector = (features_vectors[:, i] - min_) / (max_ - min_)
                features_vectors[:, i] = normalized_vector

            elif max_ == 0 and min_ == 0:
                # Handle the case when all values in the feature vector are zero
                features_vectors[:, i] = features_vectors[:, i]

            elif max_ == 255 and min_ == 255:
                # Handle the case when all values in the feature vector are 255 (for grayscale images)
                features_vectors[:, i] = features_vectors[:, i] / 255

        return features_vectors

    
    def compute_distance_(self, features_vectors):
        """
        Compute the distance matrix between feature vectors.

        Parameters:
            features_vectors (numpy.ndarray): The feature vectors.
        Returns:
            numpy.ndarray: The distance matrix.
        """
        X = features_vectors
        dist = distance_matrix(X, X)

        return dist
    
    def reduce_dimension_(self, features_vectors):
        
        # Calculate the mean along the columns
        column_means = np.mean(features_vectors, axis=0)
        # Find the indices of the columns with a mean value of 1
        columns_to_delete = np.where(column_means == 1)[0]

        features_vectors = np.delete(features_vectors, columns_to_delete, axis=1)
        
        # Calculate the mean along the columns
        column_means = np.mean(features_vectors, axis=0)
        # Find the indices of the columns with a mean value of 0
        columns_to_delete = np.where(column_means == 0)[0]

        features_vectors = np.delete(features_vectors, columns_to_delete, axis=1)
        
        #Apply PCA
        # Create an instance of PCA
        pca = PCA(n_components=2)

        # Fit PCA on your features_vectors array
        pca.fit(features_vectors)

        # Apply dimensionality reduction to your features_vectors
        final_features_vectors = pca.transform(features_vectors)
        
        return final_features_vectors
        
    
    def extract_combinations_(self, dist):
        """
        Extract combinations of indices based on distance matrix.

        Parameters:
            dist (numpy.ndarray): The distance matrix.
        Returns:
            list: List of combinations of indices.
        """
        combinations = []

        for i in range(len(dist[:,0])):
    
            # Compute histogram
            hist, bins = np.histogram(dist[i,:], bins=30, density=True)

            # Filter the non-zero bins and their corresponding values
            non_zero_bins = bins[:-1][hist != 0]

            # Create and fit the DBSCAN model
            dbscan = DBSCAN(eps=1.5, min_samples=1)
            labels = dbscan.fit_predict(non_zero_bins.reshape(-1, 1))

            last_zero_idx = np.where(labels == 0)[0][-1]
            first_one_idx = np.where(labels == 1)[0][0]

            threshold = np.mean([non_zero_bins[last_zero_idx], non_zero_bins[first_one_idx]])

            # Check condition and get indices
            indices = np.where(dist[i,:] < threshold)[0]

            combinations.append(indices)

        return combinations
       
    def compute_combination_proba_(self, combinations):
        """
        Compute the most probable combinations from the given combinations.

        Parameters:
            combinations (list): List of combinations.
        Returns:
            list: List of the most probable combinations.
        """
        final_combinations = []
        scores = []
        n_combinations = 0

        for i in range(len(combinations)):
            if i == 0:
                final_combinations.append(np.sort(combinations[i]))
                if (len(combinations[i]) == 9 or len(combinations[i]) == 12 or len(combinations[i]) == 16):
                    scores.append(3)
                else:
                    scores.append(1)
                n_combinations += 1
            else:
                is_different = False
                iz = 0
                for j in range(n_combinations):
                    if np.array_equal(np.array(combinations[i]), final_combinations[j]):
                        scores[j] += 1
                        is_different = False
                        iz = 1
                    else:
                        if iz == 0:
                            is_different = True

                if is_different:
                    final_combinations.append(np.sort(combinations[i]))
                    if (len(combinations[i]) == 9 or len(combinations[i]) == 12 or len(combinations[i]) == 16):
                        scores.append(3)
                    else:
                        scores.append(1)
                    n_combinations += 1

        return final_combinations, scores
    
    def compute_most_probable_combinations_(self, combinations, scores):
        """
        Compute the most probable combinations from the given combinations based on scores.

        Parameters:
            combinations (list): List of combinations.
            scores (list): List of scores corresponding to the combinations.
        Returns:
            list: List of the most probable combinations.
        """
        # Sort the indices based on scores in descending order
        sorted_indices = np.argsort(scores)[::-1]

        # Rearrange the combinations based on the sorted indices
        rearranged_list = [combinations[i] for i in sorted_indices]
        final_combinations = []
        zi = 0

        for i in range(len(rearranged_list)):
            if i == 0:
                final_combinations.append(rearranged_list[i])
            else:
                for j in range(len(final_combinations)):
                    # Check if the current combination is not present in any of the final combinations
                    is_not_present = np.isin(np.array(rearranged_list[i]), final_combinations[j], invert=True)

                    if ~np.all(is_not_present):
                        zi = 1

                if zi == 0:
                    final_combinations.append(rearranged_list[i])

            zi = 0

        return final_combinations
    
    def classify(self, pieces):
        n_features = 1200
        #Extract the features:
        features_vectors = self.generate_feature_vectors_(pieces, n_features)
        #Normalize the features
        features_vectors = self.normalize_features_(features_vectors)
        #Reduce dimentionality of features vectors with PCA
        final_features_vectors = self.reduce_dimension_(features_vectors)
        #Distance computation
        dist = self.compute_distance_(final_features_vectors)
        #Extract the combinations for each piece using k-means
        combinations = self.extract_combinations_(dist)
        #Compute the probability scores
        combinations, scores = self.compute_combination_proba_(combinations)
        #Compute the classification results
        final_combinations = self.compute_most_probable_combinations_(combinations, scores)
        
        return final_combinations, dist
        
class Puzzle:
        
    def __init__(self):
        pass
    
    def calculate_ssim_(self, image1, image2):
        """
        Calculate the structural similarity index (SSIM) between two images.
        SSIM is a metric for measuring the similarity between two images based on human perception.

        Parameters:
            image1 (PIL.Image.Image): The first image.
            image2 (PIL.Image.Image): The second image.
        Returns:
            float: The SSIM value between the two images.
        """

        # Convert images to grayscale for SSIM calculation
        grayscale_image1 = image1.convert('L')
        grayscale_image2 = image2.convert('L')

        # Convert grayscale images to numpy arrays
        array1 = np.array(grayscale_image1)
        array2 = np.array(grayscale_image2)

        # Calculate SSIM between the two images
        similarity = ssim(array1, array2)

        return similarity


    def calculate_edge_similarity_(self,image1, image2):
        """
        Calculate the edge similarity between two images using structural similarity index (SSIM).
        Edge similarity measures the similarity of the edges present in the images.

        Parameters:
            image1 (PIL.Image.Image): The first image.
            image2 (PIL.Image.Image): The second image.
        Returns:
            float: The edge similarity value between the two images.
        """

        # Extract edges from the images
        edges1 = image1.filter(ImageFilter.FIND_EDGES)
        edges2 = image2.filter(ImageFilter.FIND_EDGES)

        # Calculate SSIM between the edge images
        similarity = self.calculate_ssim_(edges1, edges2)

        return similarity

    def rotate_image_(self,image, angle):
        """
        Rotate the given image by the specified angle.

        Parameters:
            image (PIL.Image.Image): The image to rotate.
            angle (float): The angle to rotate the image by, in degrees.
        Returns:
            PIL.Image.Image: The rotated image.
        """

        # Rotate the image by the specified angle
        rotated_image = image.rotate(angle, expand=True)
        return rotated_image


    def flip_image_(self,image, flip_horizontal, flip_vertical):
        """
        Flip the given image horizontally or vertically if needed.

        Parameters:
            image (PIL.Image.Image): The image to flip.
            flip_horizontal (bool): Flag indicating whether to flip horizontally.
            flip_vertical (bool): Flag indicating whether to flip vertically.
        Returns:
            PIL.Image.Image: The flipped image.
        """

        # Flip the image horizontally or vertically if needed
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT) if flip_horizontal else image
        flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM) if flip_vertical else flipped_image
        return flipped_image


    def group_images(self,subpart_images):
        """
        Group the subpart images to form a big image based on similarity and orientation analysis.

        Parameters:
            subpart_images (List[PIL.Image.Image]): List of subpart images.
        Returns:
            PIL.Image.Image: The final big image formed by grouping the subpart images.
        """

        # Calculate the number of rows and columns
        rows = int(len(subpart_images) ** 0.5)
        columns = rows

        # Sort subpart images based on similarity to the first image
        sorted_images = [subpart_images[0]]
        subpart_images.pop(0)

        while subpart_images:
            best_similarity = 0
            best_index = None
            best_orientation = (0, False, False)  # (angle, flip_horizontal, flip_vertical)

            for i, image in enumerate(subpart_images):
                for angle in range(0, 360, 90):
                    rotated_image = self.rotate_image_(image, angle)

                    for flip_horizontal in [False, True]:
                        for flip_vertical in [False, True]:
                            flipped_image = self.flip_image_(rotated_image, flip_horizontal, flip_vertical)
                            edge_similarity = self.calculate_edge_similarity_(sorted_images[-1], flipped_image)

                            if edge_similarity > best_similarity:
                                best_similarity = edge_similarity
                                best_index = i
                                best_orientation = (angle, flip_horizontal, flip_vertical)

            best_image = subpart_images[best_index]
            sorted_images.append(best_image)
            subpart_images.pop(best_index)

        # Calculate the dimensions of the big image
        width = sorted_images[0].width * columns
        height = sorted_images[0].height * rows

        # Create a new image with the dimensions calculated
        grouped_pieces = Image.new('RGB', (width, height))

        for i, image in enumerate(sorted_images):
            # Calculate the position of the current subpart image
            row = i // columns
            col = i % columns

            # Calculate the coordinates to paste the subpart image
            x = col * image.width
            y = row * image.height

            # Paste the subpart image onto the big image
            grouped_pieces.paste(image, (x, y))

        # Return the final big image
        return grouped_pieces


