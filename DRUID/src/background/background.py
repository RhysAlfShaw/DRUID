""" 
File: background.py
Author: Rhys Shaw
Date: 23/12/2023
Version: v1.0
Description: Functions for calculating the background of an image.

"""

import numpy as np
from astropy.stats import mad_std





def radio_background_map(cutout : np.ndarray, box_size : int):

    '''

    This function takes an image and a box size and calculates the radio_background() for each box to create a map of local background.
    
    '''
    
    step_size = box_size//2
    
    # initialize the map
    map_shape = (cutout.shape[0]//step_size, cutout.shape[1]//step_size)
    bg_map = np.zeros(map_shape)
    
    # box
    box = np.ones((box_size, box_size))
    
    for i in range(0, cutout.shape[0], step_size):
        for j in range(0, cutout.shape[1], step_size):
            # get the box
            box_image = cutout[i:i+box_size, j:j+box_size]
            # calculate the radio background
            local_bg = mad_std(box_image, ignore_nan=True)
            # set the value in the map
            bg_map[i//step_size, j//step_size] = local_bg

    # now upsample the map to the original image size
    bg_map = np.repeat(bg_map, step_size, axis=0)
    bg_map = np.repeat(bg_map, step_size, axis=1)
    
    # shift the map to the correct position
        
    return bg_map
        
        






        
        
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

    return local_bg










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




