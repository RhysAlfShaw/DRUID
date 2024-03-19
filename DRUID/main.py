

'''
File: main.py
Author: Rhys Shaw
Date: 23/12/2023
Version: 0.0
Description: Main file for DRUID
'''


version = '0.0.0'

from .src import utils 
from .src import homology_new as homology
from .src import background
from matplotlib import colors
from .src import source
import numpy as np
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy
import pandas as pd
import setproctitle
from multiprocessing import Pool
import time
import os
from scipy import ndimage
import logging




DRUID_MESSAGE = """   
              
              
#############################################

_______   _______          _________ ______  
(  __  \ (  ____ )|\     /|\__   __/(  __  \ 
| (  \  )| (    )|| )   ( |   ) (   | (  \  )
| |   ) || (____)|| |   | |   | |   | |   ) |
| |   | ||     __)| |   | |   | |   | |   | |
| |   ) || (\ (   | |   | |   | |   | |   ) |
| (__/  )| ) \ \__| (___) |___) (___| (__/  )
(______/ |/   \__/(_______)\_______/(______/ 
        
        
#############################################

Detector of astRonomical soUrces in optIcal and raDio images

Version: {}

For more information see:
https://github.com/RhysAlfShaw/DRUID
        """.format(version)


setproctitle.setproctitle('DRUID')










class sf:

    


    def __init__(self, image : np.ndarray = None, image_path : str = None, mode : str = None, 
                 pb_path : str = None, cutup : bool = False, cutup_size : int = 500, 
                 cutup_buff : int = None, output : bool = True, 
                 area_limit : int = 5, smooth_sigma = 1, nproc : int = 1, GPU : bool = False, 
                 header : astropy.io.fits.header.Header = None, Xoff : int = None, Yoff : int = None,debug_mode=False,remove_edge=True) -> None:
        """Initialise the DRUID, here general parameters can be set..

        Args:
            image (np.ndarray, optional): _description_. Defaults to None.
            image_path (str, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to None.
            pb_path (str, optional): _description_. Defaults to None.
            cutup (bool, optional): _description_. Defaults to False.
            cutup_size (int, optional): _description_. Defaults to 500.
            output (bool, optional): _description_. Defaults to True.
            area_limit (int, optional): _description_. Defaults to 5.
            smooth_sigma (int, optional): _description_. Defaults to 1.
            nproc (int, optional): _description_. Defaults to 1.
            GPU (bool, optional): _description_. Defaults to False.
            header (astropy.io.fits.header.Header, optional): _description_. Defaults to None.
            Xoff (int, optional): _description_. Defaults to None.
            Yoff (int, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        # start up message!
        print(DRUID_MESSAGE)

        if debug_mode:
            logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
            logging.debug('Debug mode enabled')
        else:
            logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
            
        self.cutup = cutup
        self.output = output
        self.image_path = image_path
        self.area_limit = area_limit
        self.smooth_sigma = smooth_sigma
        self.GPU = GPU
        self.Xoff = Xoff
        self.Yoff = Yoff
        self.cutup_buff = cutup_buff
        self.remove_edge = remove_edge
        
        if self.GPU:

            # Lets try importing the GPU stuff, if it fails then we can just use the CPU.

            try:
                import cupy as cp
                from cupyx.scipy.ndimage import label as cp_label
                
            #    num_gpus = cp.cuda.runtime.getDeviceCount()
                #print(f'Found {num_gpus} GPUs')
                
            #    if num_gpus > 0:
            #        #print('GPUs are avalible, GPU functions will now be avalible.')
                   # GPU_AVALIBLE = True
            #    else:
                    #print('No GPUs avalible, using CPU')
            #        GPU_AVALIBLE = False
            except:
                
                #print('Could not import cupy. DRUID GPU functions will not be avalible')
                GPU_AVALIBLE = False
                
        self.nproc = nproc
        
        self.header = header
        
        if self.image_path is None:
            self.image = image
            
        else:
            self.image, self.header = utils.open_image(self.image_path)
        
        if mode == 'Radio':
            # add 1 pixel padding to all sides of the image this causes issues?
            self.image = np.pad(self.image,((1,1),(1,1)),mode='constant',constant_values=0)
            
        
        if self.smooth_sigma !=0:
            self.image = utils.smoothing(self.image,self.smooth_sigma)
            logging.info('Image smoothed with sigma = {}'.format(self.smooth_sigma))
            
        self.mode = mode
        # check if the mode is valid
        
        if self.mode not in ['Radio', 'optical','other']:
            raise ValueError('Mode must be either radio, optical or other.')
        
        
        self.pb_path = pb_path
        
        if self.pb_path is not None:
            self.pb_image , self.pb_header = utils.open_image(self.pb_path)
        
        
        if self.cutup:
            self.cutup_size = cutup_size
            #print('Cutting up image {}x{} into {}x{} cutouts'.format(self.image.shape[0],self.image.shape[1],cutup_size,cutup_size))
            if self.cutup_buff is not None:
                self.cutouts, self.coords = utils.cut_image_buff(self.image,cutup_size, buffer_size=self.cutup_buff)
            else:
                self.cutouts, self.coords = utils.cut_image(cutup_size,self.image)
                        
            if self.pb_path is not None:
                self.pb_cutouts, self.pb_coords = utils.cut_image(cutup_size,self.pb_image)
                
            else:
                self.pb_cutouts = None
                self.pb_coords = None
        else:
            self.cutup_size = None
            self.cutouts = None
            self.coords = None
            self.pb_cutouts = None
            self.pb_coords = None
            
        
            
        
        
        







    def phsf(self, lifetime_limit : float = 0,lifetime_limit_fraction : float = 2,):
        
        """ Performs the persistent homology source finding algorithm.
        
        Args:
        
            lifetime_limit (float): The lifetime limit for the persistent homology algorithm.
        
        Returns:
            
            None. The catalogue is stored in the self.catalogue attribute.
            
        """
        
        
        if self.cutup == True:
            
            catalogue_list = []
            IDoffset = 0
            for i, cutout in enumerate(self.cutouts):
                
                print("Computing for Cutout number :{}/{}".format(i+1, len(self.cutouts)))
                
                catalogue = homology.compute_ph_components(cutout,self.local_bg,analysis_threshold_val=self.analysis_threshold_val,
                                                        lifetime_limit=lifetime_limit,output=self.output,bg_map=self.bg_map,area_limit=self.area_limit,
                                                        GPU=self.GPU,lifetime_limit_fraction=lifetime_limit_fraction,mean_bg=self.mean_bg,
                                                        IDoffset=IDoffset,box_size=self.box_size,detection_threshold=self.sigma)
                if len(catalogue) == 0:
                    continue
                
                IDoffset += len(catalogue)
                
                catalogue['Y0_cutout'] = self.coords[i][0]
                catalogue['X0_cutout'] = self.coords[i][1]
                # corrent the x1 position for the cutout
                catalogue['x1'] = catalogue['x1'] + self.coords[i][0]
                catalogue['x2'] = catalogue['x2'] + self.coords[i][0]
                catalogue['y1'] = catalogue['y1'] + self.coords[i][1]
                catalogue['y2'] = catalogue['y2'] + self.coords[i][1]
                catalogue['bbox1'] = catalogue['bbox1'] + self.coords[i][0]
                catalogue['bbox2'] = catalogue['bbox2'] + self.coords[i][1]
                catalogue['bbox3'] = catalogue['bbox3'] + self.coords[i][0]
                catalogue['bbox4'] = catalogue['bbox4'] + self.coords[i][1]
                catalogue['distance_from_center'] = ((catalogue['x1']-cutout.shape[0]/2)**2 + (catalogue['y1']-cutout.shape[1]/2)**2)**0.5
                catalogue['cutup_number'] = i
                catalogue_list.append(catalogue)
                self.catalogue = pd.concat(catalogue_list)
                # remove duplicates and keep the one closest to its cutout centre.
                #print('before duplicated removal :',len(self.catalogue))
                #self.catalogue = utils.remove_duplicates(self.catalogue)
                # drop any with edge_flag == 1
                # set edge flag False to 0
                self.catalogue['edge_flag'] = self.catalogue['edge_flag'].astype(int)
            
            if self.remove_edge:
                self.catalogue = self.catalogue[self.catalogue.edge_flag != 1]
                self.catalogue = self.catalogue.sort_values(by=['distance_from_center'],ascending=True)
                self.catalogue = self.catalogue.drop_duplicates(subset=['x1','y1','Birth'], keep='first')
            else:
                
                for i, row in catalogue.iterrows():
                    if row.edge_flag == 1:
                        if row.bbox1 == 0:
                            row.bbox1 = 1
                        if row.bbox2 == 0:
                            row.bbox2 = 1
                        if row.bbox3 == self.image.shape[0]:
                            row.bbox3 = self.image.shape[0]-1
                        if row.bbox4 == self.image.shape[1]:
                            row.bbox4 = self.image.shape[1]-1
                            
                        self.catalogue.at[i,'bbox1'] = row.bbox1
                        self.catalogue.at[i,'bbox2'] = row.bbox2
                        self.catalogue.at[i,'bbox3'] = row.bbox3
                        self.catalogue.at[i,'bbox4'] = row.bbox4
                
        else:
            IDoffset = 0
            catalogue = homology.compute_ph_components(self.image,self.local_bg,analysis_threshold_val=self.analysis_threshold_val,
                                                        lifetime_limit=lifetime_limit,output=self.output,bg_map=self.bg_map,area_limit=self.area_limit,
                                                        GPU=self.GPU,lifetime_limit_fraction=lifetime_limit_fraction,mean_bg=self.mean_bg,
                                                        IDoffset=IDoffset,box_size=self.cutup_size,detection_threshold=self.sigma)
            self.catalogue = catalogue
            
            self.catalogue['Y0_cutout'] = 0
            self.catalogue['X0_cutout'] = 0
            self.catalogue['edge_flag'] = self.catalogue['edge_flag'].astype(int)
            if self.remove_edge:
                self.catalogue = self.catalogue[self.catalogue.edge_flag != 1]
            else:
                for i, row in catalogue.iterrows():
                    if row.edge_flag == 1:
                        if row.bbox1 == 0:
                            row.bbox1 = 1
                        if row.bbox2 == 0:
                            row.bbox2 = 1
                        if row.bbox3 == self.image.shape[0]:
                            row.bbox3 = self.image.shape[0]-1
                        if row.bbox4 == self.image.shape[1]:
                            row.bbox4 = self.image.shape[1]-1
                            
                        self.catalogue.at[i,'bbox1'] = row.bbox1
                        self.catalogue.at[i,'bbox2'] = row.bbox2
                        self.catalogue.at[i,'bbox3'] = row.bbox3
                        self.catalogue.at[i,'bbox4'] = row.bbox4
                            
        
        print(self.catalogue)
        
        self.catalogue = self.catalogue.sort_values(by=['lifetime'],ascending=False)
        #print('after duplicate removal :',len(self.catalogue))
        
        # do enclosed_i evaluation with the bounding box to ensure we dont use the whole image.
        # plt.figure(figsize=(20,20))
        # plt.imshow(self.image,cmap='gray_r',norm=colors.LogNorm(clip=True))
        # plt.scatter(self.catalogue.y1,self.catalogue.x1,c='r',marker='x')
        # # # # plot the bounding boxes
        # for i, row in self.catalogue.iterrows():
        #       ymin = row.bbox1
        #       ymax = row.bbox3
        #       xmin = row.bbox2
        #       xmax = row.bbox4
        #       plt.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],c='r')
        # plt.savefig('test.png')
        # time.sleep(3)
        enclosed_i_list = []
        t0 = time.time()
        for i in tqdm(range(0,len(self.catalogue)),total=len(self.catalogue),desc='Calculating enclosed_i',disable=not self.output):
            row = self.catalogue.iloc[i]
            x1 = row.x1 - row.bbox1 + 1
            y1 = row.y1 - row.bbox2 + 1
            Birth = row.Birth
            Death = row.Death
            # is this a new row?
            #if row.new_row == 1:
            
            img = self.image[int(row.bbox1)-1:int(row.bbox3)+1,int(row.bbox2)-1:int(row.bbox4)+1]
            # reduce the cat to just the sources in the bounding box.
            
            cat = self.catalogue[self.catalogue['x1'] > int(row.bbox1)]
            cat = cat[cat['x1'] < int(row.bbox3)]
            cat = cat[cat['y1'] > int(row.bbox2)]
            cat = cat[cat['y1'] < int(row.bbox4)]
            cat['x1'] = cat['x1'] - int(row.bbox1) + 1
            cat['y1'] = cat['y1'] - int(row.bbox2) + 1
                
                                
            # plt.imshow(img,cmap='gray_r',norm=colors.LogNorm(clip=True,vmin=1E-13,vmax=1E-9))
            # plt.scatter(cat.y1,cat.x1,c='r',marker='x')
            # plt.savefig('test.png')
            # plt.close()       
            # # sleep
            # time.sleep(2)
            
            if self.GPU==True:
                import cupy as cp
                # this is not the best way to deal with this. We should crop the gpu version of the image.
                img_gpu = cp.asarray(img, dtype=cp.float64)
                enclosed_i = homology.make_point_enclosure_assoc_GPU(0,x1,y1,Birth,Death,cat,img_gpu)
                enclosed_i_list.append(enclosed_i)
            else:
                print(self.catalogue)
                enclosed_i = homology.make_point_enclosure_assoc_CPU(0,x1,y1,Birth,Death,cat,img)
                enclosed_i_list.append(enclosed_i)
                
        #print('enclosed_i calculated! t='+str(time.time()-t0)+' s')
        self.catalogue['enclosed_i'] = enclosed_i_list
        
        # correct for first destruction
        
        #print(len(self.catalogue))
        t0_correct_firs = time.time()
        print('BEfore',len(self.catalogue))
        self.catalogue = homology.correct_first_destruction(self.catalogue,output=not self.output)
        t1_correct_firs = time.time()
        #print('Time to correct first destruction: ',t1_correct_firs-t0_correct_firs)
        print('After',len(self.catalogue))
        # parent tag
        #print("Assigning parent tags..")
        t0_parent_tag = time.time()
        self.catalogue = homology.parent_tag_func_vectorized_new(self.catalogue)
        t1_parent_tag = time.time()
        #print('Time to assign parent tags: ',t1_parent_tag-t0_parent_tag)
        #print(self.catalogue['parent_tag'].value_counts())
        #print("Classifying sources in hirearchy..")
        t0_classify = time.time()
        self.catalogue['Class'] = self.catalogue.apply(homology.classify_single,axis=1)
        t1_classify = time.time()
        #print('Time to classify sources: ',t1_classify-t0_classify)
        # put ID at the front
       
            
            
            
        
        
    def set_background(self,detection_threshold,analysis_threshold,
                       set_bg=None,bg_map_bool=False,box_size=None,mode='mad_std'):
        
        #print('Setting background..')
        
        
        self.sigma = detection_threshold
        self.analysis_threshold = analysis_threshold
        self.bg_map = bg_map_bool
        # mode should be MAD_Std, RMS or other.
        
        if mode == 'Radio':
            # old verion was called radio.    
            mode = 'mad_std'

        # need to account dor the cutputs if not usinh bg_map.
        
        # bg_map and cutup are require only the same code.
        
        # bg_map and no cuput is the same as bg_map and cutup.
        
        # no cutup and no bg_map is just one estimation for the whole image.
        
        if self.cutup == True:
        
            # we want to do the bg_map but for the cutout dims. as the box size is in pixels.
            self.bg_map = True
            bg_map_bool = True
            # which is smaller the cutout size or the box size?
            if box_size is None:
                box_size = self.cutup_size
            else:
                if box_size > self.cutup_size:
                    box_size = self.cutup_size
                else:
                    pass
        self.box_size = box_size
        if bg_map_bool == True:
            #print('Creating a background map. Inputed Box size = ',box_size)
            # these will be returned as arrays like a map.
            std, mean_bg = background.calculate_background_map(self.image,box_size,mode=mode)
            #print('Background map created.')
            #print('Mean Background across cutouts: ', np.nanmean(std))
            #print('Median of bg distribution: ', np.nanmean(mean_bg))
            
        else:
            #print('Not creating a background map.')
            std, mean_bg = background.calculate_background(self.image,mode=mode)
            #print('Background set to: ',std)
            #print('Background mean set to: ',mean_bg)
            
        if set_bg is not None:
            # set bg should be a tuple of (std,mean_bg)
            #print('User has set the background.')
            std = set_bg[0]
            mean_bg = set_bg[1]
            #print('Background set to: ',std)
            #print('Background mean set to: ',mean_bg)
        
        
        self.local_bg = std*self.sigma
        self.analysis_threshold_val = std*self.analysis_threshold
        self.mean_bg = mean_bg
        




    def set_background_old(self,detection_threshold : float,analysis_threshold,
                       set_bg : float = None, bg_map : bool = None, 
                       box_size : int = 10, mode : str = 'Radio'):
        """Sets the background for the source finding algorithm.    

        Args:
            detection_threshold (int): _description_
            analysis_threshold (int): _description_
            set_bg (float, optional): _description_. Defaults to None.
            bg_map (bool, optional): _description_. Defaults to None.
            box_size (int, optional): _description_. Defaults to 10.
            mode (str, optional): _description_. Defaults to 'Radio'.
        """
        
        
        self.bg_map = bg_map
        self.sigma = detection_threshold
        self.analysis_threshold = analysis_threshold
        
        if mode == 'Radio':
            if self.cutup:
                
                # loop though each cutout and calculate the local background.
                
                if bg_map is not None:
                    
                    # users wants to use background map so lets make it
                    local_bg_list = []
                    analysis_threshold_list = []
                    mean_bg_list = []
                    for i, cutout in enumerate(self.cutouts):
                        local_bg_map, mean_bg = background.radio_background_map(cutout, box_size)
                        analysis_threshold_list.append(local_bg_map*self.analysis_threshold)
                        local_bg_list.append(local_bg_map*self.sigma)
                        mean_bg_list.append(mean_bg)
                else:
                    local_bg_list = []
                    analysis_threshold_list = []
                    mean_bg_list = []
                    for cutout in self.cutouts:
                        local_bg, mean_bg = background.radio_background(cutout)
                        analysis_threshold_list.append(local_bg*self.analysis_threshold)
                        local_bg_list.append(local_bg*self.sigma)
                        mean_bg_list.append(mean_bg)
                local_bg = local_bg_list    
                analysis_threshold = analysis_threshold_list
                mean_bg = mean_bg_list
                    
            else:

                # Radio background is calculated using the median absolute deviation of the total image.
                if bg_map is not None:
                    local_bg_o, mean_bg = background.radio_background_map(self.image, box_size)
                    local_bg = local_bg_o*self.sigma
                    analysis_threshold = local_bg_o*self.analysis_threshold    
                else:
                    local_bg_o, mean_bg = background.radio_background(self.image)
                    local_bg = local_bg_o*self.sigma
                    analysis_threshold = local_bg_o*self.analysis_threshold
                    
                    
        if mode == 'Optical':
            # Optical background is calculated using a random sample of pixels
            mean_bg, std_bg = background.optical_background(nsamples=1000)
            local_bg = mean_bg + self.sigma*std_bg
            analysis_threshold = mean_bg + std_bg*self.analysis_threshold

        if mode == 'other':
            # If the user has a custom background function, they can pass it in here.
            local_bg = set_bg*self.sigma
            analysis_threshold = local_bg*self.analysis_threshold
            #print('Background set to: ',local_bg)
            #print('Analysis threshold set to: ',analysis_threshold)
            
        self.analysis_threshold_val = analysis_threshold
        self.local_bg = local_bg
        self.mean_bg = mean_bg
        #print(self.mean_bg)
        
        #if bg_map:
            #print('Using bg_map for analysis.')
        #else:
            #if self.cutup:
            
             #   print('Mean Background across cutouts: ', np.nanmean(self.local_bg))
             #   print('Median of bg distribution: ', np.nanmean(self.mean_bg))
            
           # else:
              #  print('Background set to: ',self.local_bg)












    def source_characterising(self, use_gpu : bool = False):
        """Source Characterising function. This function takes the catalogue and the image and calculates the source properties.

        Args:
            use_gpu (bool, optional): Option to use the GPU True to use and False to not, 
                                      requires cupy module and a avalible GPU. Defaults to False.
        """
        
        self.catalogue, self.polygons = source.measure_source_properties(use_gpu=use_gpu,catalogue=self.catalogue,
                                                                           cutout=self.image,background_map=self.local_bg,
                                                                           output=self.output,cutupts=self.cutouts,mode=self.mode,header=self.header)
        
        if self.Xoff is not None:
            # correct for the poistion of the cutout. when using cutout from a larger image.
            self.catalogue['Xc'] = self.catalogue['Xc'] + self.Xoff
            self.catalogue['bbox1'] = self.catalogue['bbox1'] + self.Xoff 
            self.catalogue['bbox3'] = self.catalogue['bbox3'] + self.Xoff
        
        if self.Yoff is not None:
            # correct for the poistion of the cutout. when using cutout from a larger image.
            self.catalogue['Yc'] = self.catalogue['Yc'] + self.Yoff
            self.catalogue['bbox2'] = self.catalogue['bbox2'] + self.Yoff 
            self.catalogue['bbox4'] = self.catalogue['bbox4'] + self.Yoff

        if self.header is not None:
            #try:
            #print('Converting Xc and Yc to RA and DEC')
            Ra, Dec = utils.xy_to_RaDec(self.catalogue['Xc'],self.catalogue['Yc'],self.header,mode=self.mode)
            self.catalogue['RA'] = Ra
            self.catalogue['DEC'] = Dec
            self.catalogue['RA'] = self.catalogue['RA'].astype(float)
            self.catalogue['DEC'] = self.catalogue['DEC'].astype(float)
        #except:
        #        pass
        
        self._set_types_of_dataframe()
        
        if self.mode == 'optical':
                    
            def ABmag(flux):
                return -2.5*np.log10(flux)
            
            def RONoise(EFFRON,EFFGAIN,EXPTIME,Area):
                return np.sqrt(Area)*(EFFRON/EFFGAIN)*EXPTIME

            def SkyNoise(sky):
                return np.sqrt(sky)

            def SourceNoise(Flux):
                return np.sqrt(Flux)

            def Flux_err(EFFRON,EFFGAIN,EXPTIME,Area,sky,Flux):
                return np.sqrt(RONoise(EFFRON,EFFGAIN,EXPTIME,Area)**2 + SkyNoise(sky) + SourceNoise(Flux))

            def NOISE(row,local_ng):
                return np.sum(np.random.normal(row['mean_bg'],local_ng,int(row['Area'])))
            
            EFFGAIN = utils.get_EFFGAIN(self.header)
            EXPTIME = utils.get_EXPTIME(self.header)
            EFFRON = utils.get_EFFRON(self.header)
            #print('EFFGAIN: ',EFFGAIN)
            #print('EXPTIME: ',EXPTIME)
            #print('EFFRON: ',EFFRON)
            #print(self.catalogue['Noise'])
            self.catalogue['Flux_total_new'] = (self.catalogue['Flux_total']*EFFGAIN*EXPTIME - self.catalogue['Noise']*EFFGAIN*EXPTIME - RONoise(EFFRON,EFFGAIN,EXPTIME,self.catalogue['Area']))
            #print('Flux_total_new: ',self.catalogue['Flux_total_new'])
            self.catalogue['Flux_total_err'] = Flux_err(EFFRON,EFFGAIN,EXPTIME,self.catalogue['Area'],self.catalogue['Noise']*EFFGAIN*EXPTIME,self.catalogue['Flux_total_new'])
            #print('Flux_total_err: ',self.catalogue['Flux_total_err'])
            self.catalogue['SNR'] = self.catalogue['Flux_total_new']/self.catalogue['Flux_total_err']/self.catalogue['Area']
            #print('SNR: ',self.catalogue['SNR'])
            self.catalogue['MAG_err'] = 1/self.catalogue['SNR'] # use the approximate error for the magnitude.
            self.catalogue['Flux_total_new'] = self.catalogue['Flux_total_new']/(EFFGAIN*EXPTIME)
            self.catalogue['MAG_flux'] = ABmag(self.catalogue['Flux_total_new'])






    def create_polygons(self,use_gpu=False):
        '''
        Creates Polygons/contours best when you just want segmentations and not source charateristics.
        '''
        
        self.catalogue = source.create_polygons(use_gpu=use_gpu,catalogue=self.catalogue,
                                                cutout=self.image,output=self.output,cutupts=self.cutouts)








    def create_polygons_fast(self):
        
        # since we have a bounding box, we can just create a polygon in the bounding box.
        
        polygons = []
        for index, row in tqdm(self.catalogue.iterrows(),total=len(self.catalogue),desc='Creating polygons'):
            contour = utils._get_polygons_in_bbox(row.bbox2-2,row.bbox4+2,row.bbox1-2,row.bbox3+2,row.x1,row.y1,row.Birth,row.Death)
            polygons.append(contour)

        self.polygons = polygons







    def create_polygons_gpu(self):
        '''
        Recommended when using GPU acceleration. and not using the bounding box to simplify the polygon creation.
        '''
        polygons = []
        self.image_gpu = cp.asarray(self.image, dtype=cp.float64)
        
        for index, row in self.catalogue.iterrows():
            t0 = time.time()
            contour = utils._get_polygons_gpu(row.x1,row.y1,row.Birth,row.Death)
            t1 = time.time()
            #print('Time to create polygon: ',t1-t0)
            polygons.append(contour)
        self.polygons = polygons

        
        




    def _set_types_of_dataframe(self):
        """
        Sets the catalogue to the correct data types. This is important to allow for writing data. 
        Otherwise the datatypes will remain object which will try to be pickled.
        
        """
        self.catalogue['ID'] = self.catalogue['ID'].astype(int)
        self.catalogue['Birth'] = self.catalogue['Birth'].astype(float)
        self.catalogue['Death'] = self.catalogue['Death'].astype(float)
        self.catalogue['x1'] = self.catalogue['x1'].astype(float)
        self.catalogue['y1'] = self.catalogue['y1'].astype(float)
        self.catalogue['x2'] = self.catalogue['x2'].astype(float)
        self.catalogue['y2'] = self.catalogue['y2'].astype(float)
        self.catalogue['Flux_total'] = self.catalogue['Flux_total'].astype(float)
        self.catalogue['Flux_peak'] = self.catalogue['Flux_peak'].astype(float)
        self.catalogue['Area'] = self.catalogue['Area'].astype(float)
        self.catalogue['Xc'] = self.catalogue['Xc'].astype(float)
        self.catalogue['Yc'] = self.catalogue['Yc'].astype(float)
        self.catalogue['bbox1'] = self.catalogue['bbox1'].astype(float)
        self.catalogue['bbox2'] = self.catalogue['bbox2'].astype(float)
        self.catalogue['bbox3'] = self.catalogue['bbox3'].astype(float)
        self.catalogue['bbox4'] = self.catalogue['bbox4'].astype(float)
        self.catalogue['Maj'] = self.catalogue['Maj'].astype(float)
        self.catalogue['Min'] = self.catalogue['Min'].astype(float)
        self.catalogue['Pa'] = self.catalogue['Pa'].astype(float)
        self.catalogue['parent_tag'] = self.catalogue['parent_tag'].astype(float)
        self.catalogue['Class'] = self.catalogue['Class'].astype(float)
        if self.cutup:   
            self.catalogue['Y0_cutout'] = self.catalogue['Y0_cutout'].astype(float)
            self.catalogue['X0_cutout'] = self.catalogue['X0_cutout'].astype(float)
        self.catalogue['SNR'] = self.catalogue['SNR'].astype(float)
        self.catalogue['Noise'] = self.catalogue['Noise'].astype(float)







    def plot_sources(self,cmap,figsize=(10,10),norm='linear',save_path=None):
        """Plots the source polygons on the image.

        Args:
            cmap (str): matplotlib cmap to use, e.g. 'gray'. See https://matplotlib.org/stable/tutorials/colors/colormaps.html for more info.
            figsize (tuple, optional): Desired figure size. Defaults to (10,10).
            norm (str, optional): _description_. Defaults to 'linear'.
            save_path (str, optional): Save path if you desire to save the figure. Defaults to None.
            
        """
        plt.figure(figsize=figsize)
        plt.imshow(self.image,cmap=cmap,origin='lower',norm=norm)
        #plt.scatter(self.catalogue['Xc'],self.catalogue['Yc'],s=10,c='r')
        
        for i, poly in enumerate(self.polygons):
            if poly is not None:
                plt.plot(poly[:,1],poly[:,0])
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()



    def save_catalogue(self,save_path,filetype=None,overwrite=False):
        """Save Catalogue to a file.

        Args:
            save_path (str): Desired path to save the catalogue.
            filetype (str, optional): Specify the file type or include the approprate file extention. Defaults to None.
            overwrite (bool, optional): Overwrite the save file if the name is the same. Defaults to False.
        
        """
        
        # get the extension from the save_path
        #print('Saving Catalogue to file: ',save_path)
        fileextention = save_path.split('.')[-1]
        
        if filetype is None:
            filetype = fileextention
            
        if filetype == 'csv':
            self.catalogue.to_csv(save_path,index=False,overwrite=overwrite)
         #   print('Catalogue saved to: ',save_path)
        
        if filetype == 'fits':
            from astropy.table import Table
          #  print('Saving to fits with astropy')
            enclosed_i = self.catalogue['enclosed_i']
          
            for i in range(len(enclosed_i)):
                for j in range(len(enclosed_i[i])):
                    enclosed_i[i][j] = int(enclosed_i[i][j])
                if len(enclosed_i[i]) == 0:
                    enclosed_i[i] = [0]
            self.catalogue['enclosed_i'] = enclosed_i
            #print(self.catalogue)
            t = Table.from_pandas(self.catalogue)
            t.write(save_path,overwrite=overwrite)
                
        
        if filetype == 'hdf':
            self.catalogue.to_hdf(save_path,key='catalogue',mode='w')
           # print('Catalogue saved to: ',save_path)
        
        if filetype == ('txt' or 'ascii'):
            self.catalogue.to_csv(save_path,index=False,overwrite=overwrite)
            #print('Catalogue saved to: ',save_path)
        
    def open_catalogue(self,file_path,filetype=None):
        from astropy.table import Table
        
        self.catalogue = Table.read(file_path)
        
        for i in range(len(self.catalogue)):
            self.catalogue['contour'][i] = np.array(self.catalogue['contour'][i]).reshape(-1,2)
        
        self.catalogue = self.catalogue.to_pandas()    
            
            
    def save_polygons_to_ds9(self, filename):

        '''
        Saves the polygons to a ds9 region file.
        '''

        with open(filename, 'w') as f:
            f.write('# Region file format: DS9 version 4.1\n')
            f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
            for polygon in self.polygons:
                f.write('polygon(')
                for i, point in enumerate(polygon):
                    f.write('{:.2f},{:.2f}'.format(point[1], point[0])) # note this transformation as the index in some CARTA inmages start at -1.
                    if i < len(polygon) - 1:
                        f.write(',')
                f.write(')\n')
                
                
            
    def save_polygons_to_hdf5(self, filename):

        '''
        Saves the polygons to a hdf5 file.
        '''
        import h5py

        hf = h5py.File(filename, 'w')
        for i in range(len(self.catalogue)):
            key = self.catalogue['ID'][i]
            hf.create_dataset(str(key), data=self.catalogue['contour'][i])
        hf.close()
        
        
            
            
        
        