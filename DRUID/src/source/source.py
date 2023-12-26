
"""
File: src/source.py
Author: Rhys Shaw
Date: 27/12/2023
Version: v1.0
Description: Functions for calculating source properties for sources.

"""

from ..utils import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label


try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as cupy_label

except:
    
    pass




def create_params_df(cutup : bool, params : list):

        '''

        Creates a pandas dataframe from the parameters.

        '''

        params = pd.DataFrame(params,columns=['ID',
                                                  'Birth',
                                                  'Death',
                                                  'x1',
                                                  'y1',
                                                  'x2',
                                                  'y2',
                                                  'Flux_total',
                                                  'Flux_peak',
                                                  'Corr_f',
                                                  'Area',
                                                  'Xc',
                                                  'Yc',
                                                  'bbox1',
                                                  'bbox2',
                                                  'bbox3',
                                                  'bbox4',
                                                  'Maj',
                                                  'Min',
                                                  'Pa',
                                                  'parent_tag',
                                                  'Class',
                                                  'SNR',
                                                  'Noise'])
    
        return params
    









def radio_characteristing(catalogue : pd.DataFrame, sigma : float, cutout : np.ndarray = None, cutup : bool = False,
                          pb_PATH : str = None, pb_image : np.ndarray = None, local_bg : np.ndarray = None,
                          cutout_pb : np.ndarray = None, background_map : np.ndarray = None):
    
    # get beam in pixels

    if cutout is None:
        cutup = False
        Beam, BMAJp, BMINp, BPA = utils.calculate_beam()
        background_map = local_bg


    else:
        Beam, BMAJ, BMIN, BPA = utils.calculate_beam()
        image = cutout
        cutup = True
        if pb_PATH is not None:
            pb_image = cutout_pb
        
        background_map = background_map


    # for each source in the catalogue create mask and measure properties. prephorm source flux correction.

    flux_correction_list = []

    params = []

    for i, source in tqdm(catalogue.iterrows(),total=len(catalogue),desc='Calculating Source Properties..',disable=not self.output):
         
        try:

            mask = utils.get_mask(row=source,image=image)
    
        except:

            continue

        source_props = utils.get_region_props(mask,image=image)

        source_props = utils.props_to_dict(source_props[0])

        peak_coords = np.where(image == source_props['max_intensity'])

        y_peak_loc = peak_coords[0][0]
        x_peak_loc = peak_coords[1][0]

        Model_Beam = utils.model_beam_func(source_props['max_intensity'],image.shape,
                                           x_peak_loc,y_peak_loc,BMAJp/2,
                                           BMINp/2,BPA)
        Flux_correction_factor = utils.flux_correction_factor(mask, Model_Beam)

        flux_correction_list.append(Flux_correction_factor)

        # calculate the flux of the source with option for pb correction.
            
        if pb_PATH is not None:
            
            background_mask = mask*background_map/sigma                    # fixed problem with slight offset.
            Flux_total = np.nansum(mask*image/pb_image - background_mask)/Beam   # may need to be altered for universality.
            Flux_peak = np.nanmax(mask*image/pb_image) - background_mask[y_peak_loc,x_peak_loc]
        
            # may need to be altered for universality.
            # get location of peak in the image
            
            Flux_peak_loc = np.where(image == Flux_peak)
            Flux_peak = Flux_peak - background_mask[Flux_peak_loc[0][0],Flux_peak_loc[1][0]]
            
        else:
            
            background_mask = mask*background_map/sigma                 # fixed problem with slight offset.
            #print(self.Beam)
            Flux_total = np.nansum(mask*image - background_mask)/Beam   # may need to be altered for universality.
            Flux_peak = np.nanmax(mask*image) - background_mask[y_peak_loc,x_peak_loc]
        
        #pdb.set_trace() # for debugging
        background_mask = np.where(background_mask == 0, np.nan, background_mask) # set background mask to nan where there is no background.
        Noise = np.nanmean(background_mask)
       
        #print('Noise: ',Noise)
        Flux_total = Flux_total*Flux_correction_factor
        
        Area = np.sum(mask)
        SNR = Flux_total/Noise
        Xc = source_props['centroid'][1]
        Yc = source_props['centroid'][0]

        bbox1 = source_props['bbox'][0]
        bbox2 = source_props['bbox'][1]
        bbox3 = source_props['bbox'][2]
        bbox4 = source_props['bbox'][3]

        Maj = source_props['major_axis_length']
        Min = source_props['minor_axis_length']
        Pa = source_props['orientation']

        params.append([source.name,
                        source.Birth,
                        source.Death,
                        source.x1,
                        source.y1,
                        source.x2,
                        source.y2,
                        Flux_total,
                        Flux_peak,
                        Flux_correction_factor,
                        Area,
                        Xc,
                        Yc,
                        bbox1,
                        bbox2,
                        bbox3,
                        bbox4,
                        Maj,
                        Min,
                        Pa,
                        source.parent_tag,
                        source.Class,
                        SNR,
                        Noise])
    
    return create_params_df(cutup,params)
    
    
    



    
    
def large_mask_red_image_procc_GPU(Birth,Death,x1,y1,image):
    '''
    Does all gpu processing for the large mask.
    
    return the red_image and red_mask and the bounding box.
    
    '''
    
    
    mask = cp.zeros(image.shape,dtype=cp.bool_)
    mask = cp.logical_or(mask,cp.logical_and(image <= Birth, image > Death))
    # mask_enclosed = self.get_enclosing_mask_gpu(y1,x1,mask)
    labeled_mask, num_features = cupy_label(mask)
    
    # Check if the specified pixel is within the mask
    if 0 <= y1 < mask.shape[1] and 0 <= x1 < mask.shape[0]:
        label_at_pixel = labeled_mask[x1, y1]
        
        if label_at_pixel != 0:
            # Extract the connected component containing the specified pixel
            component_mask = (labeled_mask == label_at_pixel)
            #plt.imshow(component_mask.get())
            #plt.show()
            #pdb.set_trace()
            non_zero_indices = cp.nonzero(component_mask)

            # Extract minimum and maximum coordinates
            xmin = cp.min(non_zero_indices[1])
            ymin = cp.min(non_zero_indices[0])
            xmax = cp.max(non_zero_indices[1])
            ymax = cp.max(non_zero_indices[0])
            
            # images are not being cropped?
            
            red_image = image[ymin:ymax+1, xmin:xmax+1]
            red_mask = component_mask[ymin:ymax+1, xmin:xmax+1]
            
            return red_image, red_mask, xmin, xmax, ymin, ymax









def large_mask_red_image_procc_CPU(Birth,Death,x1,y1,image):
    '''
    Does all gpu processing for the large mask.
    
    return the red_image and red_mask and the bounding box.
    
    '''
    
    
    mask = np.zeros(image.shape)
    mask = np.logical_or(mask,np.logical_and(image <= Birth, image > Death))
    # mask_enclosed = self.get_enclosing_mask_gpu(y1,x1,mask)
    labeled_mask, num_features = label(mask)
    
    # Check if the specified pixel is within the mask
    if 0 <= y1 < mask.shape[1] and 0 <= x1 < mask.shape[0]:
        label_at_pixel = labeled_mask[x1, y1]
        
        if label_at_pixel != 0:
            # Extract the connected component containing the specified pixel
            component_mask = (labeled_mask == label_at_pixel)
            #plt.imshow(component_mask.get())
            #plt.show()
            #pdb.set_trace()
            non_zero_indices = np.nonzero(component_mask)

            # Extract minimum and maximum coordinates
            xmin = np.min(non_zero_indices[1])
            ymin = np.min(non_zero_indices[0])
            xmax = np.max(non_zero_indices[1])
            ymax = np.max(non_zero_indices[0])
            
            # images are not being cropped?
            
            red_image = image[ymin:ymax+1, xmin:xmax+1]
            red_mask = component_mask[ymin:ymax+1, xmin:xmax+1]
            
            return red_image, red_mask, xmin, xmax, ymin, ymax

    



    
    
    
def optical_characteristing(use_gpu,catalogue=None,cutout=None,background_map=None,output=None):
    '''
    
    Characterising the Source Assuming the input image of of the format of a optical astronomical image.
    
    # needs to work on cutup images.
    # work on the whole image.

    '''
    
    if use_gpu:
        
        try:
        
            import cupy as cp
        
        except:
        
            raise ImportError('cupy not installed. GPU acceleration not possible.')
        
    image = cutout
    #background_map = local_bg


    if use_gpu:
        image_gpu = cp.asarray(image, dtype=cp.float64)
    
    x1 = catalogue['x1'].to_numpy()
    y1 = catalogue['y1'].to_numpy()
    x2 = catalogue['x2'].to_numpy()
    y2 = catalogue['y2'].to_numpy()
    Birth = catalogue['Birth'].to_numpy()
    Death = catalogue['Death'].to_numpy()
    parent_tag = catalogue['parent_tag'].to_numpy()
    Class = catalogue['Class'].to_numpy()
    
    print(len(catalogue))
    print(len(Birth))
    
    params = []
    polygons = []


    for i, source in tqdm(enumerate(Birth),total=len(Birth),desc='Calculating Source Properties..',disable=not output):
        
        if use_gpu == True:
            red_image, red_mask, xmin, xmax, ymin, ymax = large_mask_red_image_procc_GPU(Birth[i],Death[i],x1[i],y1[i],image_gpu)
            red_mask = red_mask.astype(int)
            
            red_image = red_image.get()
            red_mask = red_mask.get()
            xmin = xmin.get()
            xmax = xmax.get()
            ymin = ymin.get()
            ymax = ymax.get()
        
        else:
            red_image, red_mask, xmin, xmax, ymin, ymax = large_mask_red_image_procc_CPU(Birth[i],Death[i],int(x1[i]),int(y1[i]),image)
            red_mask = red_mask.astype(int)
        
        contour = utils._get_polygons_in_bbox(xmin,xmax,ymin,ymax,x1[i],y1[i],Birth[i],Death[i],red_mask)
        
        source_props = utils.get_region_props(red_mask,image=red_image)
        source_props = utils.props_to_dict(source_props[0])
        red_background_mask = np.where(red_mask == 0, np.nan, red_mask*background_map)
        Noise = np.nanmean(red_background_mask)
        Flux_total = np.nansum(red_mask*red_image - red_background_mask)
        Area = source_props['area']
        SNR = Flux_total/Noise
        Xc = source_props['centroid'][1]
        Yc = source_props['centroid'][0]
        
        bbox1 = source_props['bbox'][0]
        bbox2 = source_props['bbox'][1]
        bbox3 = source_props['bbox'][2]
        bbox4 = source_props['bbox'][3]
        
        Maj = source_props['major_axis_length']
        Min = source_props['minor_axis_length']
        Pa = source_props['orientation']
        
        params.append([i,
                        Birth[i],
                        Death[i],
                        x1[i],
                        y1[i],
                        x2[i],
                        y2[i],
                        Flux_total,
                        Flux_total,
                        1,
                        Area,
                        Xc,
                        Yc,
                        bbox1,
                        bbox2,
                        bbox3,
                        bbox4,
                        Maj,
                        Min,
                        Pa,
                        parent_tag[i],
                        Class[i],
                        SNR,
                        Noise])
        polygons.append(contour)
    #print(params)
    #print(len(params[0]))
    return create_params_df(False,params), polygons

