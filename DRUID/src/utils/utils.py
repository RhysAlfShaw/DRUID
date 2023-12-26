"""
File: utils.py
Author: Rhys Shaw
Date: 23/12/2023
Version: v1.0
Description: Utility functions for DRUID

"""



from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
from scipy.ndimage import label
from astropy.wcs import WCS


try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as cupy_label

except:
    
    pass




def open_image(PATH : str):
    """
    
    Function to open fits image and return image and header.


    Args:
        PATH (str): Path to the fits image.
        
    Returns:
        image (np.ndarray): The image data.
        header (astropy.io.fits.header.Header): The image header.
    
    
    """
    
    hdul = fits.open(PATH)
    image = hdul[0].data
    header = hdul[0].header
    hdul.close()
    
    image_shape = image.shape
    
    if len(image_shape) == 4:

        image = np.squeeze(image, axis=(0,1))
    
    if len(image_shape) == 3:

        image = np.squeeze(image, axis=(0))
    
    return image, header








def cut_image(size : int, image : np.ndarray):
    
    """Cuts an image into smaller images of the specified size (square).
    
    Args:
        size (int): The size of the cutouts.
        image (np.ndarray): The image to be cut.

    Returns:
        cutouts (list): A list of the cutout images.
        coords (list): A list of the coordinates of the cutouts.
        
    """

    cutouts = []
    coords = []
    
    for i in range(0, image.shape[0], size):
        
        for j in range(0, image.shape[1], size):
            
            cutout = image[i:i+size, j:j+size]
            
            cutouts.append(cutout)
            
            coords.append([i, j])
            

    return cutouts, coords








def smoothing(image : np.ndarray, sigma : float):
    
    """Smooths an image using a gaussian filter.
    
    Args:
        image (np.ndarray): The image to be smoothed.
        sigma (float): The sigma value for the gaussian filter.

    Returns:
        smoothed_image (np.ndarray): The smoothed image.
        
    """
    
    
    smoothed_image = gaussian_filter(image, sigma=sigma)
    
    return smoothed_image








def calculate_beam(header : fits.header.Header):
    '''

    Calculates the beam of the image in pixels.

    Args:
        header (astropy.io.fits.header.Header): The header of the image.

    returns:
        beam (float): The beam correction factor.
        BMAJ (float): The BMAJ value in pixels.
        BMIN (float): The BMIN value in pixels.
        BPA (float): The BPA value in pixels.
        
    '''

    # get beam info from header

    BMAJ = header['BMAJ']
    BMIN = header['BMIN']
    # convert to pixels
    arcseconds_per_pixel = header['CDELT1']*3600*-1
    beam_size_arcseconds = header['BMAJ']*3600
    BMAJ_oversampled_spacial_width = (BMAJ**2 + beam_size_arcseconds**2)**0.5
    BMAJ = BMAJ_oversampled_spacial_width/arcseconds_per_pixel
    
    beam_size_arcseconds = header['BMIN']*3600
    BMIN_oversampled_spacial_width = (BMIN**2 + beam_size_arcseconds**2)**0.5
    BMIN = BMIN_oversampled_spacial_width/arcseconds_per_pixel
    
    try:
        BPA = header['BPA']
        
    except KeyError:
        BPA = 0
        
    return np.pi * (BMAJ)*(BMIN) / (4*np.log(2)), BMAJ, BMIN, BPA









def props_to_dict(regionprops):

        '''

        Converts the regionprops to a dictionary.
        # slow function when all are called.
        '''

        dict = {
            'area': regionprops.area,
            'bbox': regionprops.bbox,
            #'bbox_area': regionprops.bbox_area,
            'centroid': regionprops.centroid,
            #'convex_area': regionprops.convex_area,
            'eccentricity': regionprops.eccentricity,
            #'equivalent_diameter': regionprops.equivalent_diameter,
            #'euler_number': regionprops.euler_number,
            #'extent': regionprops.extent,
            #'filled_area': regionprops.filled_area,
            'major_axis_length': regionprops.major_axis_length,
            'minor_axis_length': regionprops.minor_axis_length,
            #'moments': regionprops.moments,
            #'perimeter': regionprops.perimeter,
            #'solidity': regionprops.solidity,
            'orientation': regionprops.orientation,
            'max_intensity':regionprops.max_intensity,
        }

        return dict
    







def get_region_props(mask,image):
    
    region = measure.regionprops(mask,image)
    
    return region









def model_beam_func(peak_flux,shape,x,y,bmaj,bmin,bpa):
    model_beam = np.zeros(shape)
    model_beam = generate_2d_gaussian(peak_flux,shape,(x,y),bmaj,bmin,bpa,norm=False)
    return model_beam








def flux_correction_factor(mask,Model_Beam):

    # calculate the correction factor
    
    model_beam_flux = np.sum(Model_Beam)
    masked_beam_flux = np.sum(mask*Model_Beam)
    
    correction_factor = model_beam_flux/masked_beam_flux
    
    return correction_factor







def generate_2d_gaussian(A,shape, center, sigma_x, sigma_y, angle_deg=0,norm=True):
    """
    
    Generate a 2D elliptical Gaussian distribution on a 2D array.

    Parameters:
    
        shape (tuple): Shape of the output array (height, width).
        center (tuple): Center of the Gaussian distribution (x, y).
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        angle_deg (float): Rotation angle in degrees (default is 0).

    Returns:
    
        ndarray: 2D array containing the Gaussian distribution.
    
    """
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x_c, y_c = center
    angle_rad = np.radians(angle_deg)

    # Rotate coordinates
    
    x_rot = (x - x_c) * np.cos(angle_rad) - (y - y_c) * np.sin(angle_rad)
    y_rot = (x - x_c) * np.sin(angle_rad) + (y - y_c) * np.cos(angle_rad)

    # Calculate Gaussian values
    
    gaussian = A *np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))

    if norm:
        return gaussian / (2 * np.pi * sigma_x * sigma_y) 
    else:
        return gaussian

        





def get_enclosing_mask_gpu(x, y, mask):
    """Returns the connected components inside the mask starting from the point (x, y) using the GPU.
    

    Args:
        x (int): x location of the source in the mask.
        y (int): y location of the source in the mask.
        mask (cp.ndarray): unfiltered mask of source (on GPU memory).

    Returns:
        component_mask(np.ndarray): Mask for the source in the provided coordinates.
    """

    labeled_mask, num_features = cupy_label(mask)

    # Check if the specified pixel is within the mask
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        label_at_pixel = labeled_mask[y, x]
        
        if label_at_pixel != 0:
            # Extract the connected component containing the specified pixel
            component_mask = (labeled_mask == label_at_pixel)
            return cp.asnumpy(component_mask)
        else:
            return None
    else:
        return None






def get_mask_CPU(row, img):
    
    """Get mask for a single row uses the CPU
    
    Args:
        row (pd.Series): row we want to get the mask for.
        img (np.ndarray): image that the source is in.

    Returns:
        mask_enclosed(np.ndarray): Array of the mask

    """
    
    mask = np.zeros(img.shape)
    mask = np.logical_or(mask,np.logical_and(img <= row.Birth,img > row.Death))
    mask_enclosed = get_enclosing_mask_CPU(int(row.y1),int(row.x1),mask)
    
    return mask_enclosed






def get_enclosing_mask_CPU(x, y, mask):
    '''
    Returns the connected components inside the mask starting from the point (x, y).
    '''
    labeled_mask, num_features = label(mask)
    
    # Check if the specified pixel is within the mask
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        label_at_pixel = labeled_mask[y, x]
        
        if label_at_pixel != 0:
            # Extract the connected component containing the specified pixel
            component_mask = (labeled_mask == label_at_pixel)
            return component_mask
        else:
            # The specified pixel is not part of any connected component
            return None
    else:
        # The specified pixel is outside the mask
        return None






def get_mask_GPU(Birth,Death,row,img):
    
    """Gets mask for a single row using the GPU (requires cupy)

    Args:
        Birth (float): Birth value of the row of interest.
        Death (float): Death value of the row of interest.
        row (pd.series): row of interest.
        img (cp.ndarray): Image containing the source (in the GPUs memory)

    Returns:
        mask_enclosed (np.ndarray): The mask returned in normal memory.

    """

    mask = cp.zeros(img.shape,dtype=cp.float64)
    mask = cp.logical_or(mask,cp.logical_and(img <= Birth,img > Death))
    mask_enclosed = get_enclosing_mask_gpu(int(row.y1),int(row.x1),mask)
    
    return mask_enclosed    


def _get_polygons_gpu(x1 : int ,y1 : int ,birth : float ,
                      death : float ,image):
    '''
    Returns the polygon of the enclosed area of the point (x,y) in the mask.
    '''
    # is the image on the GPU memory?
    #if not cp.is_cuda_array(self.image):
    #    self.image = cp.asarray(self.image, dtype=cp.float64)
        
    mask = cp.zeros(image.shape)
    mask = cp.logical_or(mask,cp.logical_and(image <= birth,image > death))
    mask = get_enclosing_mask_GPU(int(y1),int(x1),mask)
    contour = measure.find_contours(mask,0)[0]
    
    return contour 


def _get_polygons_CPU(x1,y1,birth,death, image : np.ndarray):

    '''
    Returns the polygon of the enclosed area of the point (x,y) in the mask.
    '''

    mask = np.zeros(image.shape)
    mask = np.logical_or(mask,np.logical_and(image <= birth, image > death))
    mask = get_enclosing_mask_CPU(int(y1),int(x1),mask)
    contour = measure.find_contours(mask,0)[0]

    return contour


def _get_polygons_in_bbox(Xmin,Xmax,Ymin,Ymax,x1,y1,birth,death,mask,pad=1):
        
        
    mask = np.pad(mask, pad, mode='constant', constant_values=0)    
    contour = measure.find_contours(mask, 0)[0]
    contour = measure.find_contours(mask, 0)[0]
    
    # remove the border
    contour[:,0] -= pad
    contour[:,1] -= pad
    
    # correct the coordinates to the original image
    contour[:,0] += Ymin
    contour[:,1] += Xmin
    
    return contour



def xy_to_RaDec(x,y,header,mode):
        
        """
        
        Convert an X and Y coordinate to RA and Dec using a header file with astropy.

        Parameters:
        x (float): The X coordinate.
        y (float): The Y coordinate.
        header_file (str): The path to the FITS header file.
        stokes (int): The Stokes dimension.
        freq (int): The frequency dimension.

        Returns:
        tuple: A tuple containing the RA and Dec in degrees.
        
        """ 
        wcs = WCS(header)
            
        if mode == 'Radio': 
            
            stokes = 0  # stokes and freq are not used in this function.
            freq = 0    # stokes and freq are not used in this function.
            ra, dec, _, _ = wcs.all_pix2world(x, y, stokes, freq, 0)
            
        elif mode == 'optical':
            # image is 2d so no stokes or freq
            ra, dec = wcs.all_pix2world(x, y, 0)
        
        return ra, dec
