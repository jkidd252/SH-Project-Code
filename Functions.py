from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import re
import os
import itertools
from skimage.filters import threshold_otsu
from skimage import data, filters, measure, morphology
import math

def Import_New(Path):
    """
    Input: Path to folder containing '.tif' image files (no other file types are read).
    Output: Dictionary with integer keys (for iteration in analysis) and the filename and image ([0], [1] respectively) stored alonside.
    """
    filelist = os.listdir(Path)
    image_dict = {}
    for i in range(len(filelist)):
        x = filelist[i]
        if x.endswith('.tif'):
            image = Image.open(str(Path + '/' + x))
            image_array = np.asarray(image, dtype = np.float64)
            image_dict.update({i : (x, image_array)})
    return image_dict

def Get_Props(image_dict):
    """
    Input: Dictionary containing Images in form of numpy arrays (located in the [1] element under a key)
    Output: Results of SKimage 'regionsprops' run on image arrays stored in new element of dictionary under the same key as input
    """
    for i in range(len(image_dict)):
        image_array = image_dict[i][1]
        labels = measure.label( image_array )
        props = measure.regionprops(labels, image_array )
        image_dict.update({i: (image_dict[i][0], image_array, props)})
    return image_dict