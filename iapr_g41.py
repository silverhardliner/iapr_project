import numpy as np
import cv2 as cv
import os 
from PIL import Image
import skimage
import scipy

import scipy.spatial.distance as distance

from skimage.feature import canny
from skimage.morphology import closing, disk, remove_small_holes

from PIL import ImageDraw

class IaprG41:

    def __init__(self):
        self.name = "G41"
        self.squares = None

    def load_input_image(image_index , im_name,  folder ="train2" , path = "data_project"):
        
        filename = f"{im_name}_{image_index:02}.png"
        path_solution = os.path.join(path,folder , filename )
        
        im= Image.open(os.path.join(path,folder,filename)).convert('RGB')
        im = np.array(im)
        return im

    def save_solution_puzzles(image_index , solved_puzzles, outliers  , folder ="train2" , path = "data_project"  ,group_id = 0):
        
        path_solution = os.path.join(path,folder + "_solution_{}".format(str(group_id).zfill(2)))
        if not  os.path.isdir(path_solution):
            os.mkdir(path_solution)

        print(path_solution)
        for i, puzzle in enumerate(solved_puzzles):
            filename =os.path.join(path_solution, "solution_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
            Image.fromarray(puzzle).save(filename)

        for i , outlier in enumerate(outliers):
            filename =os.path.join(path_solution, "outlier_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
            Image.fromarray(outlier).save(filename)
    
    def threshold(self, img, th1, th2):
        """
        This function threshold an image based on two given thresholds
        @Inputs:
        - img: input image
        - th1: lower threshold
        - th2: Higher threshold
        @Outputs:
        - ret: thresholded image
        """
        # Find the maximum intensity value in the image
        i_max = np.max(img)
        # Make a copy of the input image
        masked_img = np.zeros_like(img).astype(bool)
        # Check that thresholds are valid and within the range of the intensity values
        if (th1<th2) and (th1>=0) and (th2 <= i_max):
            # Create masks for values greater than or equal to th1 and less than or equal to th2
            mask1 = img>=th1
            mask2 = img<=th2
            # Combine the masks to create a final mask
            mask = mask1 & mask2
            # Set the values within the mask to 1 and those outside to 0
            masked_img[mask] = False
            masked_img[~mask] = True

            #_, counts = np.unique(masked_img, return_counts=True)
            #if len(counts) > 1 and counts[0] < counts[1]:
            masked_img = ~masked_img
            
            # Return the masked image
            ret = masked_img
        else:
            # Print an error message and return -1 if the thresholds are invalid
            print("error")
            ret = -1
        return ret
    
    # FUNCTION FOR FINDING CENTROIDS
    def circ_filter(self, size = 128):
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

    def find_squares(self, img_2d):
        # Make the objects to 1 and background to 0
        magic_constant = 2
        values, counts = np.unique(img_2d, return_counts=True)
        if len(counts) > 1 and counts[0] < counts[1]:
            img_2d = skimage.util.invert(img_2d.astype(int)).astype(float)+magic_constant
        
        # Square filter
        square_side = 128
        the_square = np.ones((square_side, square_side))

        # Circ filter
        circ_dia = 128
        the_circle = self.circ_filter(circ_dia)
        
        # Rotation 0 degrees
        locations = cv.filter2D(src=img_2d, kernel=the_circle, ddepth=-1)
        return locations
    
    def norm_data(data):
        """
        normalize data to have mean=0 and standard_deviation=1
        """
        mean_data=np.mean(data)
        std_data=np.std(data, ddof=1)
        #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
        return (data-mean_data)/(std_data)
    
    def find_biggest_neighbor(self, image, loc):
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
    
    def close_to_border(self, peak, small_sq_size, big_im_size):
        keep = True
        sq_half = small_sq_size//2
        cond1 = peak[0] <= sq_half
        cond2 = peak[1] <= sq_half
        cond3 = peak[0] >= big_im_size[0] - sq_half
        cond4 = peak[1] >= big_im_size[1] - sq_half
        if cond1 or cond2 or cond3 or cond4:
            keep = False

        return keep
    
    def travel_to_peaks(self, norm_loc, small_sq_size = 128):
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
                    new_loc = self.find_biggest_neighbor(norm_loc, new_loc)
                    if new_loc[0] == loc[0] and new_loc[1] == loc[1]:
                        flag = False
                    else:
                        loc = new_loc
                    k+=1
                    if k%100 == 0:
                        pass
                        #print(new_loc)
                value = norm_loc[new_loc[0], new_loc[1]]
                if k > 1 and value > norm_loc_comp:
                    keep = self.close_to_border(new_loc, small_sq_size, norm_loc.shape)
                    if keep:
                        peaks.append(new_loc)
                #print(f"{np.array([i,j])}, {new_loc}")

        peaks = np.array(peaks)
        unique_peaks = np.unique(peaks, axis=0)
        return unique_peaks


    
    # FUNCTIONS TO FIND ROTATIONS
    def my_dist(self, line, point):
        line = np.asarray(line)
        point = np.asarray(point)
        new_point = np.array([point[1], point[0]])
        p0, p1 = line
        line_centr = (p0+p1)/2
        #print(line_centr)
        d = np.linalg.norm(line_centr-new_point)
        return d
    
    def line_angle(self, line, idx=None):
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
    
    def get_angles(self):
        num = -1
        for sq in self.squares:
            num += 1
            angles = []
            for ln in sq.lines:
                one_angle = self.line_angle(ln, idx=num)
                angles.append(one_angle)

            #print(angles)
            sq.all_angles = angles
            # Remove one element to fix median calculation
            if len(angles) % 2 == 0:
                angles.pop()
            sq.angle = np.median(angles)

    
    def cluster_lines(self, centroids, lines, debug=False):
        
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
                d = self.my_dist(one_line, one_center)
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

    def generate_points_between_two_points(self, x1, y1, x2, y2, n=10):
        # Calculate distance between points
        delta_x = (x2 - x1) / (n + 1)
        delta_y = (y2 - y1) / (n + 1)

        # Generate intermediate points
        intermediate_points = []
        for i in range(1, n + 1):
            x_intermediate = x1 + i * delta_x
            y_intermediate = y1 + i * delta_y
            intermediate_points.append(np.array([x_intermediate, y_intermediate]))

        return intermediate_points
    
    def get_points_from_edges(self):
        for sq in self.squares:
            points_for_square = []
            for ln in sq.lines:
                p0, p1 = ln
                x0 = p0[0]
                y0 = p0[1]
                x1 = p1[0]
                y1 = p1[1]
                new_points = self.generate_points_between_two_points(x0, y0, x1, y1)
                points_for_square.append(new_points)
            sq.points_from_edges = np.concatenate(points_for_square)
            

    def get_square_points(self, sq, angle=0, side_lenght = 128):

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

    def add_contours(self, cont_centers, cont_clusters):
        
        other_centers = [np.array([sq.centroid[1], sq.centroid[0]]) for sq in self.squares]
        for i in range(len(cont_centers)):
            cluster_center = cont_centers[i]
            dist = distance.cdist([cluster_center], other_centers)
            closest_center_idx = np.argmin(dist)
            #print(f"Cluster index {i} to square index {closest_center_idx}")
            self.squares[closest_center_idx].contour_center = cluster_center
            self.squares[closest_center_idx].contour_cluster = cont_clusters[i]
            self.squares[closest_center_idx].all_contour_points = np.squeeze(np.concatenate(cont_clusters[i]))
    
    def get_squares(self, centroids, lines, cont_centers=None, cont_clusters=None):
        self.squares = [self.Square(centroids[i]) for i in range(len(centroids))]
        
        if cont_centers and cont_clusters:
            self.add_contours(cont_centers, cont_clusters)
        
        self.cluster_lines(centroids, lines, False)
        self.get_angles()
        for sq in self.squares:
            sq._square_points = self.get_square_points(sq, angle=sq.angle)
            sq.bigger_square_points = self.get_square_points(sq, side_lenght=200)
        self.get_points_from_edges()


        return self.squares
    
    def edge_morphology(single_channel_img, gl_std, pad_width, disk_size=3, holes_size=40000):
        gauss_lap = scipy.ndimage.gaussian_laplace(single_channel_img, gl_std)
        gauss_lap_pad = np.pad(gauss_lap, pad_width, mode='constant')
        closed_laplace = closing(gauss_lap_pad, disk(disk_size))
        remove_laplace = remove_small_holes(closed_laplace, holes_size).astype('float')
        return remove_laplace
    
    def remove_pad(image, pad_width):
        out = image[pad_width:-pad_width, pad_width:-pad_width]
        return out

    def create_perfect_mask(self, size=2000):
        output = np.zeros((size,size))
        for sq in self.squares:
            square_points = np.array([sq.square_points], dtype=np.int32)
            cv.fillPoly(output, pts = [square_points], color=(1))
        return output
    
    def edge_image_fix(self, image, flag='edge'):

        edge = 3

        # Split the image into individual color channels
        channels = cv.split(image)

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
                inpainted_channel = cv.inpaint(channel, mask, 3, cv.INPAINT_TELEA)
                inpainted_channels.append(inpainted_channel)

            # Merge the inpainted color channels back into an RGB image
            out = cv.merge(inpainted_channels)
        else:
            mirrored_channels = []
            for channel in channels:
                cropped_channel = channel[edge:-edge, edge:-edge]
                padded_channels = np.pad(cropped_channel, edge, mode='edge')
                mirrored_channels.append(padded_channels)
            out = cv.merge(mirrored_channels)
            #print(out.shape)

        return out

    def crop_square(self, image_path, points, angle, save_path=None, image_size=(128,128)):
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
        trimmed_image = trimmed_image.resize(image_size)

        # Inpaint edges to get rid of background artefacts
        edge_fixed = self.edge_image_fix(np.array(trimmed_image), flag='s')
        #print(edge_fixed.dtype)

        if save_path:
            # Save the inpainted image
            Image.fromarray(edge_fixed).save(save_path)
            #trimmed_image.save(save_path)

        return edge_fixed
        
    class Square:

        def __init__(self, centroid):
            self._centroid = np.asarray(centroid)
            self._lines = []
            self._angle = 0
            self.all_angles = []
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




    
if __name__ == "__main__":
    im_path = "data_project/train/train_00.png"

#plt.imshow(locations)
#fig = px.imshow(locations)
#fig.show()
#plt.imshow(circ_filter())