
"""
File: src/source.py
Author: Rhys Shaw
Date: 27/12/2023
Version: v1.0
Description: Functions for calculating source properties for sources.

"""

from ..src import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label
import pdb
from scipy.ndimage import binary_dilation
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

#import cv2
#import numpy as np

#def dilate_mask_circular(mask, radius):
    # Create a circular structuring element using cv2.getStructuringElement
#    circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

    # Perform dilation using cv2.dilate
#    dilated_mask = cv2.dilate(mask.astype(np.uint8), circular_kernel)#

#    return dilated_mask
    

def curve_of_growth_dilation(mask,image):
    converged = False
    i = 0
    dilated_mask = np.zeros(image.shape, dtype=bool)
    mask_before = mask
    while converged == False:
        if i == 50:
            #print('max iterations reached')
            return dilated_mask.astype(int)
       # print(i)
        i += 1
        
        dilated_mask = dilate_mask_circular(mask_before,1)

        # check if the mask has converged
        flux = np.sum(image*dilated_mask)
        flux_old = np.sum(image*mask_before)
        diff = (flux - flux_old)
        #print(diff)
        if diff<0:
            converged = True
            dilated_mask = mask_before
            
        else:
            mask_before = dilated_mask
    #print('converged {}'.format(i))
    return dilated_mask.astype(int)

    
def measure_source_properties(use_gpu,catalogue=None,cutout=None,background_map=None,output=None,cutupts=None, mode='optical',header=None):
    '''
    
    Characterising the Source Assuming the input image of of the format of a optical astronomical image.
    
    # needs to work on cutup images.
    # work on the whole image.

    '''
    if header == None:
        mode = 'other'
        
    if mode == 'Radio':
        print('Radio mode selected')
        Beam, BMAJ, BMIN, BPA = utils.calculate_beam(header=header)
        
    if mode == 'optical':
        print('Optical mode selected, modeling the PSF as a gaussian.')
        psf_fwhm_p = utils.get_psf_FWHM(header)
        #EFFRON = utils.get_EFFRON(header)
        #EFFGAIN = utils.get_EFFGAIN(header)
        #EXPTIME = utils.get_EXPTIME(header)
    
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
            
        # DILATION TEST Not Working.
        #print(red_mask)
        
        # #try:
        # if Class[i] == 0:
        #     if xmin > 50 and ymin > 50:
        #         expanx = 50
        #         expany = 50
        #         # increase mask and image size by 100 pixels in each direction.
        #         # this is to ensure that the mask is large enough to capture the entire source.
        #         # if the source is less than 50px away from the image edge then we padd with the distance to the edge -1
        #         red_mask = np.pad(red_mask,pad_width=((expanx,expanx-1),(expany,expany-1)),mode='constant',constant_values=0)
        #                         #,((expanx,expanx),(expany,expany)),mode='constant',constant_values=0)
        #         #plt.imshow(red_mask)
        #         #plt.savefig('red_mask.png')
        #         red_mask = red_mask#[0:-1,0:-1]
        #         #print('red_mask shape',red_mask.shape)
        #         red_image = image[ymin-expanx:ymax+expanx,xmin-expany:xmax+expany]
        #         #c
        #         plt.figure(figsize=(10,10))
        #         plt.imshow(red_image,cmap='gray',origin='lower',vmin=1E-12,vmax=1E-10)
        #         plt.contour(red_mask)
        #         plt.savefig('red_image.png')
                
                
        #         #print('red_image shape',red_image.shape)
        #         # dilate the mask until the flux converges.
        #         # check that the shapes are the same
        #         if red_mask.shape != red_image.shape:
        #             print('red_mask and red_image shapes are not the same.')
        #             print('Skipping source...')
        #             continue
        #         red_mask = curve_of_growth_dilation(red_mask,red_image)
        #         plt.figure(figsize=(10,10))
        #         plt.imshow(red_image,cmap='gray',origin='lower',vmin=1E-12,vmax=1E-10)
        #         plt.contour(red_mask)
        #         plt.savefig('red_image_after.png')
                
        #         pdb.set_trace()
                
        #         #print('red_mask shape',red_mask.shape)
        #         # reduce the mask and recalculate the boundig box.
        #         # plt.imshow(red_mask)
        #         # plt.savefig('red_mask.png')
        #         xminn,yminn,xmaxn,ymaxn = utils.bounding_box_cpu(red_mask)
        #         xmax = xmaxn - expany + xmax
        #         xmin = xminn - expany + xmin
        #         ymin = yminn - expanx + ymin
        #         ymax = ymaxn - expanx  + ymax
        #         red_mask = red_mask[yminn:ymaxn,xminn:xmaxn]
        #         red_image = red_image[yminn:ymaxn,xminn:xmaxn]
                

    
        contour = utils._get_polygons_in_bbox(xmin,xmax,ymin,ymax,x1[i],y1[i],Birth[i],Death[i],red_mask,0,0)
        source_props = utils.get_region_props(red_mask,image=red_image)
        source_props = utils.props_to_dict(source_props[0])
        #print(background_map)
        background_map = bg[i]
        #print('Mean bg',mean_bg)
        mean_bg_s = mean_bg[i]
        
        background_map = np.random.normal(mean_bg_s,background_map,red_mask.shape)
        
        red_background_mask = np.where(red_mask == 0, np.nan, red_mask*background_map)
        
        Noise = abs(np.nansum(red_background_mask))
        #print('Noise',Noise)
        peak_coords = np.where(red_image == source_props['max_intensity'])

        y_peak_loc = peak_coords[0][0]
        x_peak_loc = peak_coords[1][0]
        #print('x_peak_loc',x_peak_loc)
        #print('y_peak_loc',y_peak_loc)
        shape = red_image.shape
        
        Area = source_props['area']
        #print('Class',Class[i])
        
            
        
        if mode == 'Radio' or mode == 'optical':
            shape = red_image.shape
            # if shape is smaller than 100 in any direction then we add padding evenly to each side.
            if shape[0] < 100:
                pad = int((100 - shape[0])/2)
                red_image = np.pad(red_image,((pad,pad),(0,0)),mode='constant',constant_values=0)
                red_background_mask = np.pad(red_background_mask,((pad,pad),(0,0)),mode='constant',constant_values=0)
                red_mask = np.pad(red_mask,((pad,pad),(0,0)),mode='constant',constant_values=0)
                shape = red_image.shape
                y = y_peak_loc + pad
                
            else:
                x = y_peak_loc
                y = x_peak_loc
                
            if shape[1] < 100:
                pad = int((100 - shape[1])/2)
                red_image = np.pad(red_image,((0,0),(pad,pad)),mode='constant',constant_values=0)
                red_background_mask = np.pad(red_background_mask,((0,0),(pad,pad)),mode='constant',constant_values=0)
                red_mask = np.pad(red_mask,((0,0),(pad,pad)),mode='constant',constant_values=0)
                shape = red_image.shape
                x = x_peak_loc + pad
            
            else:
                x = y_peak_loc
                y = x_peak_loc
                
           # print(shape)

            if mode == 'optical':
                
                MAJ = psf_fwhm_p/2.355 # maybe use full expression in future. 
                MIN = MAJ
                BPA = 0 
                        
                Model_Beam = utils.model_beam_func(source_props['max_intensity'],shape,
                                                                           x,y,MAJ,
                                                                            MIN,BPA)
                #plt.figure(figsize=(10,10))
                #plt.imshow(Model_Beam)
                #plt.scatter(x,y,color='red',marker='x',s=10)    
                # plot the contour of the red_mask
                #plt.contour(red_mask)
                #plt.savefig('Model_Beam_test.png')
                #pdb.set_trace()
                
            else:
                BMAJ = BMAJ
                BMIN = BMIN
            
                Model_Beam = utils.model_beam_func(source_props['max_intensity'],shape,
                                                                           x,y,BMAJ/2,
                                                                            BMIN/2,BPA)
                
                
            if mode == 'Radio':
                Flux_total = np.nansum(red_mask*red_image - red_background_mask)/Beam
            else:
            
                Flux_total = np.nansum(red_mask*red_image - red_background_mask)
                
            Flux_peak = np.nanmax(red_mask*red_image) - red_background_mask[y_peak_loc,x_peak_loc]        

            Flux_correction_factor = utils.flux_correction_factor(red_mask, Model_Beam)
            
            if Area < 100:
                # we assume that the flux cannot be corrected
                Flux_total = Flux_total*Flux_correction_factor
            
            # remove the 1pixel padding
            if mode == 'Radio':
                padding = 1
            else:
                padding = 0
            
        else:
            
            Flux_total = np.nansum(red_mask*red_image - red_background_mask)
            Flux_peak = np.nanmax(red_mask*red_image) - red_background_mask[y_peak_loc,x_peak_loc]
            Flux_correction_factor = np.nan
            padding = 0
            
            # add some estimation of the PSF. To account for the loss of flux due to the PSF cutting at low SNR.

        
        SNR = Flux_total/(Area*bg[i])
        
       # Noise = np.std(red_background_mask)
        
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

