"""
File: src/homology.py
author: Rhys Shaw
date: 27-12-2023
Description: This file contains the functions that deal with calculating 
             persistence diagrams from a given image.
"""


import cripser
import numpy as np
from ..utils import utils
import time
from functools import partial
import pandas
from tqdm import tqdm
from multiprocessing import Pool, freeze_support


from collections import deque

try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as cupy_label
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False






def parent_tag_func_vectorized(df):
    """

    Vectorised implenetation of parent tag function.


    Args:

        df: pd.Dataframe - data frame for which we calculate the parent tags.


    Returns:

        df: pd.Dataframe - Pandas data frame with addition parent tag column.

    """

    enclosed_i_dict = {idx: set(row['enclosed_i']) for idx, row in df.iterrows()}

    def find_parent_tag(row):
        for idx, enclosed_i_set in enclosed_i_dict.items():
            if row.name in enclosed_i_set:
                return idx 
        return np.nan 
    
    return df.apply(find_parent_tag, axis=1)






def classify_single(row):

    """Classifiying the Rows based on orgin.

    Args:
        row: pd.series - The row that is being classified.

    Returns:
        Class: int - the Class integer that indiceates the class the row belongs too.

    """
    if row.new_row == 0:
        if len(row.enclosed_i) == 0: # no children
            if np.isnan(row.parent_tag): # no parent 
                return 0 # no children, no parent.
            else:
                return 1 # no child has parent.
        else:
            if np.isnan(row.parent_tag):
                return 2 # has children, no parent.
            else:
                return 3 # has children, has parent.
    else:
        return 4 # new row has children.
        





def make_point_enclosure_assoc(row,pd,img):
    """Returns a list of the indices of the points that are enclosed by the mask pd point.

    Args:
        row (pd.Series): _description_
        pd (pd.DataFrame): _description_
        img (np.ndarray): _description_

    Returns:
        enclosed_list (list): _description_

    """

    mask = utils.get_mask_CPU(row,img)
    
    encloses = []
    for i in range(len(pd)):
        point = pd.iloc[i]
        # we dont want to include ourselves
        if point.name == row.name:
            continue
        if mask[int(point.x1),int(point.y1)]:
            encloses.append(point.name)

    return encloses





def make_point_enclosure_assoc_GPU(Birth,Death,row,pd,img,img_gpu):
    """Returns a list of the indices of the points that are enclosed by the mask pd point.
        Uses GPU for computation.

    Args:
        Birth (float): _description_
        Death (float): _description_
        row (pd.Series): _description_
        pd (pd.DataFrame): _description_
        img (np.ndarray): _description_
        img_gpu (cp.ndarray): _description_

    Returns:
        enclosed_list (list): _description_
    """
    
    mask = utils.get_mask_GPU(Birth,Death,row,img_gpu)
    mask_coords = np.column_stack((pd['x1'], pd['y1']))
    points_inside_mask = mask[mask_coords[:, 0].astype(int), mask_coords[:, 1].astype(int)]
    encloses_vectorized = pd.iloc[points_inside_mask].index.tolist()
    # remove self from list
    encloses_vectorized.remove(row.name)
    
    return encloses_vectorized





def correct_first_destruction(pd,output):
    """
    Function for correcting for the First destruction of a parent Island.

    Args:
        pd (pd.DataFrame): Input catalogue of sources to correct.
        output (bool): True if you want interation logginf with tqdm.

    Returns:
        pd (pd.DataFrame): The new Catalogue.
   
    """

    pd['new_row'] = 0

    for i in tqdm(range(0,len(pd)),total=len(pd),desc='Correcting first destruction',disable=output):

        row = pd.iloc[i]
        
        enlosed_i = row['enclosed_i']
        
        if len(enlosed_i) >= 1: 
            new_row = row.copy()
            
            new_row['Death'] = pd.loc[enlosed_i[0]]['Death']
            
            ## this accounts for a bug were the entire series is placed in the death column.
            # not sure on the origin of this but the following corrects for it. It only occationally happends so this is not 
            # computationally expensive.
            if type(new_row['Death']) == pandas.core.series.Series:
                # get the Death value from the first item in the Series.
                new_row['Death'] = new_row['Death'].iloc[0]
            
            new_row['new_row'] = 1
            new_row.name = len(pd)+i
            pd = pandas.concat((pd,new_row.to_frame().T), ignore_index=False)
        
    return pd








def calculate_area_CPU(row, img):
    """Calculates area of source mask (for CPU)

    Args:
        row (pd.series): _description_
        img (np.ndarray): Image

    Returns:
        _type_: _description_
    """
    mask = utils.get_mask_CPU(row,img)    
    area = np.sum(mask)
    return area







def calculate_area_GPU(Birth,Death,row, img_gpu):
    """Calcualtes are of source mask (for GPU)

    Args:
        Birth (float): 
        Death (float): 
        row (pd.series): 
        img_gpu (cp.ndarray): 
    Returns:
        area (float): the calculated area of the source mask.
    """
    mask = utils.get_mask_GPU(Birth,Death,row,img_gpu)
    area = np.sum(mask)
    return area


def process_area(i,pd,img):
    # handles worker function
    return calculate_area_CPU(pd.iloc[i], img)


def process_assoc(i):
    # hanldes worker function
    return make_point_enclosure_assoc_CPU(pd.iloc[i], pd, img)









def compute_ph_components(img,local_bg,analysis_threshold_val,lifetime_limit=0,output=True,bg_map=False,area_limit=3,nproc=1,GPU=False):
    
    
    global GPU_Option 
    GPU_Option = GPU
    t0_compute_ph = time.time()
    pd = cripser.computePH(-img,maxdim=0)
    t1_compute_ph = time.time()
    print('PH computed! t='+str(t1_compute_ph-t0_compute_ph)+' s')
    pd = pandas.DataFrame(pd,columns=['dim','Birth','Death','x1','y1','z1','x2','y2','z2'],index=range(1,len(pd)+1))
    pd.drop(columns=['dim','z1','z2'],inplace=True)
    pd['lifetime'] = pd['Death'] - pd['Birth']
    pd['Birth'] = -pd['Birth'] 
    pd['Death'] = -pd['Death'] 
    
    
    if bg_map:
        
        list_of_index_to_drop = []
    
        
        for index, row in pd.iterrows():
            # check if local_bg is a map or a value
            if row['Birth'] < local_bg[int(row.x1),int(row.y1)]:
                list_of_index_to_drop.append(index)
        
        pd.drop(list_of_index_to_drop,inplace=True)
    
        
        # for each row evaluate if death is below analysis thresholdval map value at its birth point. if its below then set Death to bg map value.    
        for index, row in pd.iterrows():
            Analy_val = analysis_threshold_val[int(row.y1),int(row.x1)]
            if row['Death'] < Analy_val:
                row['Death'] = Analy_val
    
            
    else:
        
        pd = pd[pd['Birth']>local_bg] # maybe this should be at the beginning.
        pd['Death'] = np.where(pd['Death'] < analysis_threshold_val, analysis_threshold_val, pd['Death'])
       
    pd['lifetime'] = abs(pd['Death'] - pd['Birth'])
        

    print('Persis Diagram computed. Length: ',len(pd))

    if lifetime_limit > 0:
        pd = pd[pd['lifetime'] > lifetime_limit]

    
    pd.sort_values(by='Birth',ascending=False,inplace=True,ignore_index=True)
    
    if len(pd) > 0:
        
        

        area_list = []
        
        if nproc == 1:
            
            if GPU_Option == True:
                
                if GPU_AVAILABLE == True:
                    
                    # convert img to cupy array and define type so it does not have to be converted each time.
                    img_gpu = cp.asarray(img,dtype=cp.float64)
                    # Calculate area and enforce area limit Single Process.
                    print('Calculating area with GPU...')
                    t0 = time.time()
                    
                    for i in tqdm(range(0,len(pd)),total=len(pd),desc='Calculating area',disable=not output):
                        
                        row = pd.iloc[i]
                        Birth = row.Birth
                        Death = row.Death
                        area = calculate_area_GPU(Birth,Death,row,img_gpu)
                        area_list.append(area)
                        #percentage_completed = (i/len(pd))*100
                        
                        #if percentage_completed % 10 == 0:
                        #    print(percentage_completed,'%')
                       
                    print('Area calculated! t='+str(time.time()-t0)+' s')
    
                    pd['area'] = area_list
                    pd = pd[pd['area'] > area_limit]
                    
                    
                    enclosed_i_list = []
                    print('Calculating enclosed_i with GPU...')
                    t0 = time.time()
                    for i in tqdm(range(0,len(pd)),total=len(pd),desc='Calculating enclosed_i',disable=not output):
                        row = pd.iloc[i]
                        Birth = row.Birth
                        Death = row.Death
                        enclosed_i = make_point_enclosure_assoc_GPU(Birth,Death,row,pd,img,img_gpu)
                        enclosed_i_list.append(enclosed_i)
                        
                    print('enclosed_i calculated! t='+str(time.time()-t0)+' s')
            
                    pd['enclosed_i'] = enclosed_i_list
                                        
                
            elif GPU_Option == False:   
                # 1 Core no GPU.
                
                # Calculate area and enforce area limit Single Process.
    
                t0 = time.time()   
                print("No GPU")
                
                for i in tqdm(range(0,len(pd)),total=len(pd),desc='Calculating area',disable=not output):
                    area = calculate_area_CPU(pd.iloc[i],img)
                    #print(area)
                    area_list.append(area)
                    
                print('Area calculated! t='+str(time.time()-t0)+' s')
            
                pd['area'] = area_list
                pd = pd[pd['area'] > area_limit]  
                
                
                # Parent Associations Single Process
                
                enclosed_i_list = []
            
                t0 = time.time()
                for i in tqdm(range(0,len(pd)),total=len(pd),desc='Calculating enclosed_i',disable=not output):
                    row = pd.iloc[i]
                    enclosed_i = make_point_enclosure_assoc(row,pd,img)
                    enclosed_i_list.append(enclosed_i)
                
                print('enclosed_i calculated! t='+str(time.time()-t0)+' s')
                
                pd['enclosed_i'] = enclosed_i_list
                
                               
        else:
            # Multiple CPUs *** currently not working

            print('Calculating area with ',nproc,' processes')
            
            # Calculate area and enforce area limit Multi Process.
            
            t0 = time.time()
            print("Chunksize: ",len(pd)//nproc)
            index_and_args_list = [(i, pd, img) for i in range(len(pd))]

            with Pool(nproc) as p:
                area_list = list(p.starmap(process_area, range(len(pd)),chunksize=len(pd)//nproc))
            print('Area calculated! t='+str(time.time()-t0)+' s')
        
            pd['area'] = area_list
            pd = pd[pd['area'] > area_limit]     # remove 1 pixel points
                    
            print(len(pd))
            
            ## Parent Associations Multi Process.
            print('Calculating enclosed_i with ',nproc,' processes')
            t0 = time.time()
            
            print("Chunksize: ",len(pd)//nproc)
            
            with Pool(nproc) as pool:
                enclosed_i_list = list(pool.imap(process_assoc, range(len(pd)),chunksize=len(pd)//nproc))
                
            print('enclosed_i calculated! t='+str(time.time()-t0)+' s')
            
            pd['enclosed_i'] = enclosed_i_list
            

        
        pd = correct_first_destruction(pd,output=not output) 
        
        print('Calculating parent_tags... ')
        
        t0_parent_tag = time.time()
        parent_tag_list = parent_tag_func_vectorized(pd)
        pd['parent_tag'] = parent_tag_list
        t1_parent_tag = time.time()
        print('parent_tag calculated! t='+str(t1_parent_tag-t0_parent_tag)+' s')
        
        #print(pd)
        #pd['parent_tag'] = pd.apply(lambda row: parent_tag_func(row,pd), axis=1)
        
        print('Assigning Class ...')
        t0_CLass = time.time()
        pd['Class'] = pd.apply(classify_single,axis=1)
        t1_Class = time.time()
        print('Class assigned! t='+str(t1_Class-t0_CLass)+' s')
        
        return pd
    
    else:
    
        return pd