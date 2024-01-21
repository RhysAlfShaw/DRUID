
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
import pdb

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
                                                'Flux_correction_factor',
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
                                                'Noise',
                                                'X0_cutout',
                                                'Y0_cutout',
                                                'mean_bg',
                                                'Edge_flag',
                                                'contour',
                                                'enclosed_i'])
    
        return params
    






    
    
def large_mask_red_image_procc_GPU(Birth,Death,x1,y1,image,X0,Y0):
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
        #print(x1)
        #print(y1)
        #print(label_at_pixel)
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
            
            # correct the bounding box for the cutout.
            xmin = xmin #+ Y0
            xmax = xmax #+ Y0
            ymin = ymin #+ X0
            ymax = ymax #+ X0
            
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

    



    
    
    
def measure_source_properties(use_gpu,catalogue=None,cutout=None,background_map=None,output=None,cutupts=None, mode='Optical',header=None):
    '''
    
    Characterising the Source Assuming the input image of of the format of a optical astronomical image.
    
    # needs to work on cutup images.
    # work on the whole image.

    '''
    
    if mode == 'Radio':
        print('Radio mode selected')
        Beam, BMAJ, BMIN, BPA = utils.calculate_beam(header=header)
    
    if use_gpu:
        
        try:
        
            import cupy as cp
        
        except:
        
            raise ImportError('cupy not installed. GPU acceleration not possible.')
    
    image = cutout
    if use_gpu:
        image_gpu = cp.asarray(image, dtype=cp.float64)
       
        
    Birth = catalogue['Birth'].to_numpy()
    Death = catalogue['Death'].to_numpy()
    parent_tag = catalogue['parent_tag'].to_numpy()
    Class = catalogue['Class'].to_numpy()
    bg = catalogue['bg'].to_numpy()
    X0 = catalogue['X0_cutout'].to_numpy()
    Y0 = catalogue['Y0_cutout'].to_numpy()
    mean_bg = catalogue['mean_bg'].to_numpy()
    enclosed_i = catalogue['enclosed_i'].to_numpy()
    IDs = catalogue['ID'].to_numpy()
    Edge_flags = catalogue['edge_flag'].to_numpy()      
    bbox1_og = catalogue['bbox1'].to_list()
    bbox2_og = catalogue['bbox2'].to_list() 
    bbox3_og = catalogue['bbox3'].to_list()
    bbox4_og = catalogue['bbox4'].to_list()
    
    x1 = catalogue['x1'].to_numpy() #- 1 #- X0 
    y1 = catalogue['y1'].to_numpy() #- 1#- X0 
    x2 = catalogue['x2'].to_numpy() #- 1#- Y0 
    y2 = catalogue['y2'].to_numpy() #- 1#- X0
    
    
    # map X0 and Y0 to the cutout number
    params = []
    polygons = []
    
    import matplotlib.pylab as plt

    for i, source in tqdm(enumerate(Birth),total=len(Birth),desc='Calculating Source Properties..',disable=not output):
    
        if use_gpu == True:
            
            cropped_image_gpu = image_gpu[bbox1_og[i]-1:bbox3_og[i]+1,bbox2_og[i]-1:bbox4_og[i]+1]
            
            try:
                red_image, red_mask, xmin, xmax, ymin, ymax = large_mask_red_image_procc_GPU(Birth[i],Death[i],
                                                                                        x1[i]-bbox1_og[i]+1,
                                                                                        y1[i]-bbox2_og[i]+1,
                                                                                        cropped_image_gpu,
                                                                                        X0[i],Y0[i])
            except:
                
                print('Error in GPU processing!')
                print('Source ID: ',IDs[i])
                print('x1:',x1[i]-bbox1_og[i]+1)
                print('y1:',y1[i]-bbox2_og[i]+1)
                print('Birth:',Birth[i])
                print('Death:',Death[i])
                #print('Cropped_image',cropped_image_gpu.get())
                
            red_mask = red_mask.astype(int)
            
            red_image = red_image.get()
            red_mask = red_mask.get()
            xmin = xmin.get() + bbox2_og[i]
            xmax = xmax.get() + bbox2_og[i] 
            ymin = ymin.get() + bbox1_og[i]
            ymax = ymax.get() + bbox1_og[i]
        
        else:
            cropped_image = image[bbox1_og[i]-1:bbox3_og[i]+1,bbox2_og[i]-1:bbox4_og[i]+1]
            try:
                red_image, red_mask, xmin, xmax, ymin, ymax = large_mask_red_image_procc_CPU(Birth[i],
                                                                                            Death[i],
                                                                                            int(x1[i])-int(bbox1_og[i])+1,
                                                                                            int(y1[i])-int(bbox2_og[i])+1,
                                                                                            cropped_image)
            except:
                print("Error in CPU processing!")
                print('Source ID: ',IDs[i])
                print('x1:',x1[i]-bbox1_og[i]+1)
                print('y1:',y1[i]-bbox2_og[i]+1)
                print('Birth:',Birth[i])
                print('Death:',Death[i])
                
            red_mask = red_mask.astype(int)

            xmin = xmin + bbox2_og[i]
            xmax = xmax + bbox2_og[i]
            ymin = ymin + bbox1_og[i]
            ymax = ymax + bbox1_og[i]
            
        #print(red_mask)
        contour = utils._get_polygons_in_bbox(xmin,xmax,ymin,ymax,x1[i],y1[i],Birth[i],Death[i],red_mask,0,0)
        source_props = utils.get_region_props(red_mask,image=red_image)
        source_props = utils.props_to_dict(source_props[0])
        #print(background_map)
        background_map = bg[i]
        #print('Mean bg',mean_bg)
        mean_bg_s = mean_bg[i]
        
        background_map = np.random.normal(mean_bg_s,background_map,red_mask.shape)
        
        red_background_mask = np.where(red_mask == 0, np.nan, red_mask*background_map)
        
        Noise = np.sum(red_background_mask) 
        peak_coords = np.where(red_image == source_props['max_intensity'])

        y_peak_loc = peak_coords[0][0]
        x_peak_loc = peak_coords[1][0]
        
        if mode == 'Radio':
            shape = red_image.shape
            # if shape is smaller than 100 in any direction then we add padding evenly to each side.
            if shape[0] < 100:
                pad = int((100 - shape[0])/2)
                red_image = np.pad(red_image,((pad,pad),(0,0)),mode='constant',constant_values=0)
                red_background_mask = np.pad(red_background_mask,((pad,pad),(0,0)),mode='constant',constant_values=0)
                red_mask = np.pad(red_mask,((pad,pad),(0,0)),mode='constant',constant_values=0)
                shape = red_image.shape
                x = y_peak_loc + pad
                
            else:
                x = y_peak_loc
                y = x_peak_loc
                
            if shape[1] < 100:
                pad = int((100 - shape[1])/2)
                red_image = np.pad(red_image,((0,0),(pad,pad)),mode='constant',constant_values=0)
                red_background_mask = np.pad(red_background_mask,((0,0),(pad,pad)),mode='constant',constant_values=0)
                red_mask = np.pad(red_mask,((0,0),(pad,pad)),mode='constant',constant_values=0)
                shape = red_image.shape
                y = x_peak_loc + pad
            
            else:
                x = y_peak_loc
                y = x_peak_loc
                
           # print(shape)
            
            Model_Beam = utils.model_beam_func(source_props['max_intensity'],shape,
                                        x,y,BMAJ/2,
                                        BMIN/2,BPA)
        
            Flux_total = np.nansum(red_mask*red_image - red_background_mask)/Beam
            Flux_peak = np.nanmax(red_mask*red_image) - red_background_mask[y_peak_loc,x_peak_loc]        

            Flux_correction_factor = utils.flux_correction_factor(red_mask, Model_Beam)
            Flux_total = Flux_total*Flux_correction_factor
            
            # remove the 1pixel padding
            padding = 1
            
        else:
            
            Flux_total = np.nansum(red_mask*red_image - red_background_mask)
            Flux_peak = np.nanmax(red_mask*red_image) - red_background_mask[y_peak_loc,x_peak_loc]
            Flux_correction_factor = np.nan
            padding = 0
            
        Area = source_props['area']
        
        SNR = Flux_total/(Area*bg[i])
        
        Noise = np.std(red_background_mask)
        
        Xc = source_props['centroid'][1] + xmin - padding
        Yc = source_props['centroid'][0] + ymin - padding
        
            
        Xc = Xc #+ X0[i]
        Yc = Yc #+ Y0[i]
        bbox1 = source_props['bbox'][0] + xmin - padding
        bbox2 = source_props['bbox'][1] + ymin - padding
        bbox3 = source_props['bbox'][2] + xmin - padding
        bbox4 = source_props['bbox'][3] + ymin - padding
        
        Maj = source_props['major_axis_length'] 
        Min = source_props['minor_axis_length']
        Pa = source_props['orientation']
        
        if Edge_flags[i] != 1:
        
            params.append([IDs[i],
                            Birth[i],
                            Death[i],
                            x1[i],
                            y1[i],
                            x2[i],
                            y2[i],
                            Flux_total,
                            Flux_peak, # this should be the peak flux.
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
                            parent_tag[i],
                            Class[i],
                            SNR,
                            Noise,
                            X0[i],
                            Y0[i],
                            mean_bg_s,
                            Edge_flags[i],
                            contour,
                            enclosed_i[i]])
        
            polygons.append(contour)
        #except:
        #    print('Error in optical characteristing!')
        #    print('Source ID: ',IDs[i])
        #    print('Skipping source...')
        #    continue
        #print(params)
        #print(len(params[0]))
    
    return create_params_df(False,params), polygons

