from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import itertools
from skimage.filters import threshold_otsu
from skimage import data, filters, measure, morphology, color
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.spatial.distance import pdist, squareform
from skimage.io import imread
import math
import time

def Import_New(Path, array_input):
    """
    Input: Path to folder containing '.tif' image files (no other file types are read).
    Output: Dictionary with integer keys (for iteration in analysis) and the filename and image ([0], [1] respectively) stored alonside.
    """
    image_list = []
    if array_input != None:
        for i in Path:
            min_dist = 6
            analysis_footprint = 3
            image_list.append({'file': None ,'array': i,'Min Dist': min_dist, 'footprint': analysis_footprint, 'Min Size': 10})
        return image_list
    
    filelist = os.listdir(Path)
    for i in range(len(filelist)):
        x = filelist[i]
        if x.endswith('.tif'):
            # im_test = imread("Ch2.tif", as_gray=True) # might want to update image input route!?!?!?!??!?!
            image = Image.open(str(Path + '/' + x))
            image_array = np.asarray(image, dtype = np.float64)
            min_dist = 6
            analysis_footprint = 3
            image_list.append({'file': x ,'array': image_array,'Min Dist': min_dist, 'footprint': analysis_footprint, 'Min Size': 10})
    return image_list


def Analysis_new( binary_image_dict,  pos_diag, eccentricity_threshold):
    image_Objects = binary_image_dict['Objects']
    props = measure.regionprops( image_Objects )
    num_objects = len(props)
    
    ecc = np.array([ prop.eccentricity for prop in props ])
    ecc_good = ecc > eccentricity_threshold                  # filtering objects with low eccentricity out.
    
    num_objects_ecc_threshold = np.sum(ecc_good)

    centroids = np.array([ prop.centroid for prop in props ])
    centroids_good = centroids[ecc_good]
    # get distances between centroids
    distances = pdist(centroids_good)     
    # make a matrix of distances between centroids corresponding to all object interactions
    distance_matrix = squareform(distances)

    # orientations are treated as unit vectors and decomposed into (x,y) elements
    angles = np.array([prop.orientation for prop in props])
    angles = angles[ecc_good]
    angles[angles < 0] = abs(angles[angles < 0])
    if np.all(angles >= 0) == np.False_:
        raise "Not all angles are posative"

    mean_angle = np.mean(angles)
    av_director = np.array([np.sin(mean_angle), np.cos(mean_angle)])
    
    orient = np.array([[ np.sin(angle), np.cos(angle)] for angle in angles ])
    angles_rel = np.dot(orient, orient.T)

    mean_director_interaction = np.array([np.dot(av_director, vector.T) for vector in orient])
    mean_director_order = ((2*(mean_director_interaction)**2)-1)
    
    
    # Calculate pairwise order parameter from orientation vectors
    orientation_dots = ((2*(np.dot(orient, orient.T))**2)-1)
    
    assert np.round(np.trace(orientation_dots), 2) == num_objects_ecc_threshold , "Sum of Orientation Matrix diagonal is inconsitent with # of objects" 
    assert np.round(np.trace(distance_matrix), 2) == 0,  "Sum of diagonal elements of Orientation Matrix is not 0"

    # take only the lower triangular section of order parameter matrix (it should be symmetric about its diagonal, this is done to prevent double counting)
    triag_s = np.tril( orientation_dots, k=pos_diag )
    triag_s[triag_s == 0] = np.nan 
    # flatten the triangular elements into a 1D array
    s_dataset = triag_s
    
    triag_d = np.tril( distance_matrix, k=pos_diag )
    triag_d[triag_d == 0] = np.nan
    d_dataset = triag_d
        
    return d_dataset, s_dataset, num_objects, orientation_dots, mean_director_order, num_objects_ecc_threshold 



def Image_Proccess( binary_image_list, imaging, index, pos_diag ):
    """
    Input: 
    """    
    if index != -1:
        lower = index
        upper = index +1
    else:
        lower = 0
        upper = len(binary_image_list)
    
    for i in range(lower, upper):
        binary_image = binary_image_list[i]['array'] > 0
        min_dist = binary_image_list[i]['Min Dist']
        foot = binary_image_list[i]['footprint']
        Min_size = binary_image_list[i]['Min Size']
        
        distance = ndi.distance_transform_edt(binary_image)
        
        # Find local maxima (these will be the markers)
        local_maxi = peak_local_max(distance, min_distance = min_dist, footprint=np.ones((foot, foot)), labels=binary_image)
        
        mask = np.zeros(distance.shape, dtype=bool) # changed from distance.shape
        mask[tuple(local_maxi.T)] = True
        
        # Label the markers
        markers, num = ndi.label(mask)   #, structure=distance)
        labels = watershed(-distance, markers, mask=binary_image)

        objects = measure.label(labels)
        accepted_objects = morphology.remove_small_objects(objects, min_size= Min_size)
        
        if imaging == 1:
            fig, ax = plt.subplots(1, 2, figsize=(24, 12))
            ax[0].imshow(binary_image, cmap='gray')
            ax[0].set_title(str(binary_image_list[i]['file']) + ' ; Key = ' + str(i))
            
            rgb_obj = color.label2rgb(accepted_objects, bg_label=0)
            ax[1].imshow(rgb_obj)
            ax[1].set_title('Watershed Segmentation; min_dist = '+str(min_dist)+' , footprint = (' +str(foot)+','+str(foot)+')')
            #plt.show()
        
        binary_image_list[i].update({'Binary Image': binary_image, 'Objects': accepted_objects})
    return  binary_image_list



def Bin(distances, orientation_dots, max_distance, bin_width):
# Maximum distance to consider for correlation
    if max_distance is None:
        max_distance = np.nanmax(distances)
   
    # Define bins for distances
    bins = np.arange(0, max_distance + bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
   
    # Calculate the average dot product for each bin
    C_r = np.zeros(len(bin_centers))
    C_r_err = np.zeros(len(bin_centers))
    C_r_standard_error = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        in_bin = (distances >= bins[i]) & (distances < bins[i + 1])
        if np.any(in_bin):
            C_r[i] = np.nanmean(orientation_dots[in_bin])
            C_r_err[i] = np.nanstd(orientation_dots[in_bin])
            C_r_standard_error[i] = np.nanstd(orientation_dots[in_bin])/np.sqrt(np.sum(in_bin))
        else:
            C_r[i] = np.nan  # Set to NaN if no pairs are in the bin
   
    return bin_centers, C_r, C_r_err, C_r_standard_error

def Bin_Fin(distances, orientation_dots, s_err, max_distance, bin_width):
# Maximum distance to consider for correlation
    if max_distance is None:
        max_distance = np.nanmax(distances)
   
    # Define bins for distances
    bins = np.arange(0, max_distance + bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
   
    # Calculate the average dot product for each bin
    C_r = np.zeros(len(bin_centers))
    C_r_err = np.zeros(len(bin_centers))
    C_r_standard_error = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        in_bin = (distances >= bins[i]) & (distances < bins[i + 1])
        if np.any(in_bin):
            C_r[i] = np.nanmean(orientation_dots[in_bin])
            C_r_err[i] = np.nanstd(orientation_dots[in_bin])
            C_r_standard_error[i] = np.sqrt( np.sum(s_err[in_bin]**2))/np.sum(in_bin)
        else:
            C_r[i] = np.nan  # Set to NaN if no pairs are in the bin
   
    return bin_centers, C_r, C_r_err, C_r_standard_error
