""" 
File: background.py
Author: Rhys Shaw
Date: 23/12/2023
Version: v1.0
Description: Functions for calculating the background of an image.

"""

import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt




def radio_background_map(cutout : np.ndarray, box_size : int):

    '''

    This function takes an image and a box size and calculates the radio_background() for each box to create a map of local background.
    
    '''
    
    step_size = box_size//2
    
    # initialize the map
    map_shape = (cutout.shape[0]//step_size, cutout.shape[1]//step_size)
    bg_map = np.zeros(map_shape)
    mean_bg_map = np.zeros(map_shape)
    
    # box
    box = np.ones((box_size, box_size))
    
    for i in range(0, cutout.shape[0], step_size):
        for j in range(0, cutout.shape[1], step_size):
            # get the box
            box_image = cutout[i:i+box_size, j:j+box_size]
            # calculate the radio background
            local_bg = mad_std(box_image, ignore_nan=True)
            mean_bg = np.nanmedian(box_image)
            # set the value in the map
            bg_map[i//step_size, j//step_size] = local_bg
            mean_bg_map[i//step_size, j//step_size] = mean_bg
            

    # now upsample the map to the original image size
    bg_map = np.repeat(bg_map, step_size, axis=0)
    bg_map = np.repeat(bg_map, step_size, axis=1)
    
    mean_bg_map = np.repeat(mean_bg_map, step_size, axis=0)
    mean_bg_map = np.repeat(mean_bg_map, step_size, axis=1)
    # shift the map to the correct position
        
    return bg_map, mean_bg_map
        
    
def calculate_background_map(image,box_size,mode='mad_std'):
    
    # do a sliding box to calculate the background
    # calculate the background std and mean for each box.
    # then upsample the map to the original image size.
    
    image_height, image_width = len(image), len(image[0])
    print('Image size: {},{}'.format(image_height,image_width))
    print('Number of Pixels in Image: {}'.format(image_height*image_width))
    box_sum = 0
    box_mean_bg = np.zeros((image_height//box_size + 1, image_width//box_size + 1))
    box_std_bg = np.zeros((image_height//box_size + 1, image_width//box_size + 1))
    # estimate the dimensions of the background map
    print('Number of boxes to calculate: ', (image_height//box_size + 1) * (image_width//box_size + 1))
    for i in range(image_height//box_size + 1):
        #print(i)
        for j in range(image_width//box_size + 1):
            # Extract the subarray (box) from the image
            xmin = i*box_size
            ymin = j*box_size 
            subarray = image[xmin:xmin+box_size, ymin:ymin+box_size]
            #print(subarray)
            # Perform calculations on the subarray
            box_mean_bg[i,j], box_std_bg[i,j] = radio_background(subarray,metric=mode)  
            
    #plt.imshow(box_mean_bg)
    #plt.savefig('box_mean_bg.png')
    return box_mean_bg, box_std_bg
            
    

def get_bg_value_from_result_image(original_location_in_full_image, box_size, bg_map):
    i, j = original_location_in_full_image
    result_value = bg_map[i//box_size + 1, j//box_size + 1]
    return result_value



def calculate_background(image, mode='mad_std'):
    '''
    This function calculates the background of an image.
    
    Args:
        image (np.ndarray): The image.
        mode (str, optional): Can choose from mad_std or rms.
    
    Returns:
        background (float): The background of the image.
    '''
    
    if mode == 'mad_std' or mode == 'rms':
        
    
        local_bg, mean_bg = radio_background(image, metric=mode)

    else:
        # some other method can be added here.
        raise ValueError('mode not recognised. Please use mad_std or rms')
    
    
    return local_bg, mean_bg


        
        
def radio_background(image : np.ndarray, metric : str ='mad_std'):
    """
    
    Returns the local background of an image.

    Args:
        image (np.ndarray): The image.
        metric (str, optional): Can Choose from mad_std or rms.

    Raises:
        ValueError: If metric is not recognised.

    Returns:
    
        local_bg (float): The local background of the image.
    
    """
    
    if metric == 'mad_std':
    
        local_bg = mad_std(image,ignore_nan=True)
        
    elif metric == 'rms':
        
        local_bg = np.sqrt(np.nanmean(image**2))
        
    else:
        raise ValueError('metric not recognised. Please use mad_std or rms')
    
    mean_bg = np.nanmedian(image)
    
    return local_bg, mean_bg










def optical_background(nsamples : int, image : np.ndarray):

        '''

        Returns the mean and standard deviation of a random sample of pixels in the image.
        This function also resamples the mean by doing a 3-sigma clip.
        
        Args:
            nsamples (int): The number of samples to take.
            
        Returns:
            mean_bg (float): The mean background value.
            std_bg (float): The standard deviation of the background values.
        
        '''

        sample_box = np.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]])
        
        image_shape = image.shape
        min_x = 2
        max_x = image_shape[0]-2
        min_y = 2
        max_y = image_shape[1]-2

        # list of random x and y coordinates
        x = np.random.randint(min_x, max_x, nsamples)
        y = np.random.randint(min_y, max_y, nsamples)

        background_values = []

        for i in range(len(x)):
            sample_box = np.array([[1,1,1],
                                    [1,1,1],
                                    [1,1,1]])

            sample_box = sample_box*image[x[i]-1:x[i]+2,y[i]-1:y[i]+2]
            background_values += sample_box.flatten().tolist()


        background_values = np.array(background_values)
        background_values = background_values[background_values < np.mean(background_values) + 3*np.std(background_values)]
        background_values = background_values[background_values > np.mean(background_values) - 3*np.std(background_values)]

        mean_bg = np.mean(background_values)
        std_bg = np.std(background_values)
        
        
        return mean_bg, std_bg




