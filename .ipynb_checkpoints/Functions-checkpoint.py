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
from skimage.io import imread
import math
import time

def Import_New(Path):
    """
    Input: Path to folder containing '.tif' image files (no other file types are read).
    Output: Dictionary with integer keys (for iteration in analysis) and the filename and image ([0], [1] respectively) stored alonside.
    """
    filelist = os.listdir(Path)
    image_list = []
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


#def Get_Props(image_list): # MAYBE DEPRECIATED AND NEED DELETED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    """
#    Input: Dictionary containing Images in form of numpy arrays (located in the [1] element under a key)
#    Output: Results of SKimage 'regionsprops' run on image arrays stored in new element of dictionary under the same key as #input
#    """
#    for i in range(len(image_list)):
#        image_array = image_list[i]['array']
#        labels = measure.label( image_array )
#        props = measure.regionprops(labels, image_array )
#        image_dict.update({i: (image_dict[i][0], image_array, props)})
#    return image_dict

def alignment_cent_calc( i, ax ):
    labels = i['Objects']
    props = measure.regionprops(labels, i['Binary Image'])
        # 'props' and 'labels'are of different shapes!!! - careful when handling  
    area = []
    for k in props:
        area.append(k.area)
    biggest_area = np.max(area)
          
    orient = np.ones(len(props))
    centroid = np.ones((len(props),2))
    plt.figure(1)
    for j in range(len(props)):
        if props[j].eccentricity > 0.7 and props[j].area <= biggest_area*0.95:
            orient[j] = props[j].orientation
            centroid[j][1], centroid[j][0] = props[j].centroid
            ax[1].scatter(props[j].centroid[1], props[j].centroid[0], s=0.5, alpha=1, color='black')
        else:
            continue
    
    #ax = plt.gca()
    #ax.set_ylim(ax.get_ylim()[::-1])   
    #plt.imshow(i['Binary Image'], cmap='gray')
    plt.show()
    i.update({'Objects_ECC': labels, 'Centroid': centroid, 'Orientation': orient, 'Obj Number': len(props)})  
    return i


def Image_Proccess( binary_image_list, imaging, index):
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
        
        mask = np.zeros(distance.shape, dtype=bool)
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
        binary_image_list[i] = alignment_cent_calc( binary_image_list[i], ax )
    return  binary_image_list


    
def relative_alignment(image_dict, graph):
    """
    yap - current version
    """
    d_list = []
    S_list = []
    S_weight_list = []
    S_err_list = []
    S_nn_list = []
    for j in image_dict:
        image = j
        for i in range(len(image['Orientation'])):
            Orient = image['Orientation']
            Dist = image['Centroid']
        
            relative_orient = np.dot(Orient[i], Orient)
            
            centroid_dist_x = (Dist[:,0] - Dist[i][0])
            centroid_dist_x = np.delete(centroid_dist_x, i)
            centroid_dist_y = (Dist[:,1] - Dist[i][1])
            centroid_dist_y = np.delete(centroid_dist_y, i) # delete point used as origin!
        
            relative_orient = np.delete(relative_orient, i)
            
            distance = np.linalg.norm([centroid_dist_x, centroid_dist_y], axis=0)  
            #===========NN==================
            distance_nn = 100 > distance 
            S_nn = np.cos(2*relative_orient[distance_nn])

            if any(distance_nn) == True and np.sum(distance[distance_nn]) != 0:
                S_nn_weight = np.average(S_nn, weights=distance[distance_nn])
                S_nn_list.append(S_nn_weight)
            else:
                print('object '+str(i)+' in '+str(image['file'])+' cannot compute; 1= '+str(any(distance_nn) == True)+' 2= '+str(np.sum(distance[distance_nn]) != 0))
            #============NN=================
            
            S = np.cos(2*relative_orient)

            if graph ==1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
                ax1.scatter(distance, S, s=1, label='All')
                ax1.scatter(distance_nn, S_nn, s=1, label='NN')
                ax1.set_ylim(-1, 1)
                ax1.set_ylabel('S = Cos^2(theta)')
                ax1.set_xlabel('Distance (centroid seperation) [Pixels]')
                ax1.legend()
                
                ax2.imshow(image['Binary Image'])
                ax2.scatter(Dist[i][0], Dist[i][1])
                ax2.imshow(color.label2rgb(image['Objects'], bg_label=0))
                plt.show()
                
            S_av = np.mean(S)
            S_list.append(S_av)
            S_weight = np.average(S, weights=distance)
            S_weight_list.append(S_weight)
            S_w_std = np.std(S_weight)
            S_err_list.append(S_w_std)
    return S_list, S_weight_list, S_err_list, S_nn_list
        
        
        

#===============================================================================================================================
def Relative_Orientation(image_dict):
    """
    Input: Dictionary of images and pre-determined properties (where the result to measure.regionprops() is stored in element 2).
    Output: Dictionary with key corresponding to input containing position and alignment data for objects in analysed objects.
    """
    analysis_dict = {} 
    for x in range(len(image_dict)+1):
        start = time.time()
        regions = image_dict[x][2]

        theta = []
        cosine_theta = []
        position = []
        for i in range(len(regions)):
            for j in range(len(regions)):
                if i == j:
                    break
                y0, x0 = regions[i].centroid
                r1, r0 = regions[j].centroid
                pos_diff = np.sqrt((x0 - r0)**2 + (y0 - r1)**2)
                position.append(pos_diff)

                orient1 = regions[i].orientation  
                orient2 = regions[j].orientation

                if abs(orient1) == orient1:
                    theta1 = 90 - orient1*180/np.pi
                else:
                    a = -90 - orient1*180/np.pi
                    theta1 = 180 + a
                                
                if abs(orient2) == orient2:
                    theta2 = 90 - orient2*180/np.pi
                else:
                    a = -90 - orient2*180/np.pi
                    theta2 = 180 + a
            
                if theta1 >= theta2:
                    alignment = theta1 - theta2
                    theta.append(alignment)
                    cosine_theta.append(np.cos(alignment))
                elif theta2 > theta1:
                    alignment = theta2 - theta1
                    theta.append(alignment)
                    cosine_theta.append(np.cos(alignment))
                else:
                    break
                    
                #x2 = x0 - math.sin(orient1) * 0.5 * regions[i].axis_major_length
                #y2 = y0 - math.cos(orient1) * 0.5 * regions[i].axis_major_length

                #diff_x = x2 - x0
                #diff_y = y2 - y0
                #vector = np.array([diff_x, diff_y])
        
                #x_iplus = r0 - math.sin(orient2) * 0.5 * regions[j].axis_major_length
                #y_iplus = r1 - math.cos(orient2) * 0.5 * regions[j].axis_major_length
        
                #diff_xiplus = x_iplus - r0
                #diff_yiplus = y_iplus - r1
                #vector_iplus = np.array([diff_xiplus, diff_yiplus])
                
                #rel_align = np.arccos(vector.dot(vector_iplus)/(np.linalg.norm(vector)*np.linalg.norm(vector_iplus)))
                #theta.append(rel_align)
                #cosine_rel_align = (vector.dot(vector_iplus)/(np.linalg.norm(vector)*np.linalg.norm(vector_iplus)))
                #cosine_theta.append(cosine_rel_align)

        theta = np.asarray([theta])
        theta[np.isnan(theta)] = 0
        cosine_theta = np.asarray([cosine_theta])
        cosine_theta[np.isnan(cosine_theta)] = 0
        position = np.asarray([position])
        analysis_dict.update({x: ( position, position**2, theta, cosine_theta )})
        print(str(x)+'/'+str(len(image_dict)) + ' , time taken = ' +str(time.time() - start))
    return analysis_dict




def Relative_Orientation_vector(image_dict):
    """
    Input: Dictionary of images and pre-determined properties (where the result to measure.regionprops() is stored in element 2).
    Output: Dictionary with key corresponding to input containing position and alignment data for objects in analysed objects.
    """
    analysis_dict = {} 
    for x in range(len(image_dict)+1):
        start = time.time()
        regions = image_dict[x][2]

        theta = []
        cosine_theta = []
        position = []
        for i in range(len(regions)):
            for j in range(len(regions)):
                if i == j:
                    break
                y0, x0 = regions[i].centroid
                r1, r0 = regions[j].centroid
                pos_diff = np.sqrt((x0 - r0)**2 + (y0 - r1)**2)
                position.append(pos_diff)

                orient1 = regions[i].orientation  
                orient2 = regions[j].orientation

                    
                x2 = x0 - math.sin(orient1) * 0.5 * regions[i].axis_major_length
                y2 = y0 - math.cos(orient1) * 0.5 * regions[i].axis_major_length

                diff_x = x2 - x0
                diff_y = y2 - y0
                vector = np.array([diff_x, diff_y])
        
                x_iplus = r0 - math.sin(orient2) * 0.5 * regions[j].axis_major_length
                y_iplus = r1 - math.cos(orient2) * 0.5 * regions[j].axis_major_length
        
                diff_xiplus = x_iplus - r0
                diff_yiplus = y_iplus - r1
                vector_iplus = np.array([diff_xiplus, diff_yiplus])
                
                rel_align = np.arccos(vector.dot(vector_iplus)/(np.linalg.norm(vector)*np.linalg.norm(vector_iplus)))
                theta.append(rel_align)
                cosine_rel_align = (vector.dot(vector_iplus)/(np.linalg.norm(vector)*np.linalg.norm(vector_iplus)))
                cosine_theta.append(cosine_rel_align)

        theta = np.asarray([theta])
        theta[np.isnan(theta)] = 0
        cosine_theta = np.asarray([cosine_theta])
        cosine_theta[np.isnan(cosine_theta)] = 0
        position = np.asarray([position])
        analysis_dict.update({x: ( position, position**2, theta, cosine_theta )})
        print(str(x)+'/'+str(len(image_dict)) + ' , time taken = ' +str(time.time() - start))
    return analysis_dict
                
        