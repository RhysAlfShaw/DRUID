

import cripser
import numpy as np
from ..src import utils
from ..src import background

import time
import pandas
from tqdm import tqdm

#from multiprocessing import Pool, freeze_support

# used for debugging

import pdb

#from collections import deque

try:
    import cupy as cp
    from cupyx.scipy.ndimage import label as cupy_label
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    
# used for debugging
import matplotlib.pyplot as plt

def make_point_enclosure_assoc_GPU(id,x1,y1,Birth,Death,pd,img_gpu):
    """Returns a list of the ID of the points that are enclosed by the mask pd point.
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
    
    mask = utils.get_mask_GPU(Birth,Death,x1,y1,img_gpu).get()
    #pdb.set_trace()
    mask_coords = np.column_stack((pd['x1'], pd['y1']))
    points_inside_mask = mask[mask_coords[:, 0].astype(int), mask_coords[:, 1].astype(int)]
    encloses_vectorized = pd.iloc[points_inside_mask]['ID'].tolist()
    # remove self from list
    #print(row.ID)
    #print(encloses_vectorized)
    #encloses_vectorized.remove(row.ID)
    #print(encloses_vectorized)
    #pdb.set_trace()
    return encloses_vectorized

def make_point_enclosure_assoc_CPU(ID,x1,y1,Birth,Death,pd,img):
    """Returns a list of the ID of the points that are enclosed by the mask pd point.
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
    
    mask = utils.get_mask_CPU(x1,y1,Birth,Death,img)
    #print(mask)
    #plt.imshow(mask)
    #plt.savefig('test.png')
    #pdb.set_trace()
    mask_coords = np.column_stack((pd['x1'], pd['y1']))
    points_inside_mask = mask[mask_coords[:, 0].astype(int), mask_coords[:, 1].astype(int)]
    encloses_vectorized = pd.iloc[points_inside_mask]['ID'].tolist()
    # remove self from list
    #print(row.ID)
    #print(encloses_vectorized)
    #encloses_vectorized.remove(ID)
    #print(encloses_vectorized)
    #pdb.set_trace()
    return encloses_vectorized


def classify_single(row):

    """Classifiying the Rows based on orgin.

    Args:
        row: pd.series - The row that is being classified.

    Returns:
        Class: int - the Class integer that indiceates the class the row belongs too.

    """
    if row.new_row == 0:
        if len(row.enclosed_i) <=1: # no children
            if row.parent_tag==row.ID: # no parent 
                return 0 # no children, no parent.
            else:
                return 1 # no child has parent.
        else:
            if row.parent_tag==row.ID: # no parent
                return 2 # has children, no parent.
            else:
                return 3 # has children, has parent.
    else:
        return 4 # new row has children.
        

def parent_tag_func_vectorized(df):
    """

    Vectorised implenetation of parent tag function.


    Args:

        df: pd.Dataframe - data frame for which we calculate the parent tags.


    Returns:

        df: pd.Dataframe - Pandas data frame with addition parent tag column.

    """

    enclosed_i_dict = {row['ID']: set(row['enclosed_i']) for idx, row in df.iterrows()}
    
    def find_parent_tag(row):
        #if row.new_row == 0: # only set it if it is not a new row.
        
        for ID, enclosed_i_set in enclosed_i_dict.items():
            if row.ID in enclosed_i_set:
                return ID
            
        return np.nan 
        
        #else: # if it is a new row then we already know the parent tag.
        #    return row.parent_tags
        
    return df.apply(find_parent_tag, axis=1)

def parent_tag_func_vectorized_new(df):
    """
    Vectorised implementation of parent tag function.

    Args:
        df: pd.Dataframe - data frame for which we calculate the parent tags.

    Returns:
        df: pd.Dataframe - Pandas data frame with addition parent tag column.
    """

    # Create a dictionary that maps each enclosed ID to its parent ID
    parent_tag_dict = {enclosed_id: ID for ID, enclosed_i_set in df[['ID', 'enclosed_i']].values for enclosed_id in enclosed_i_set}

    # Define a function that finds the parent tag of a row
    def find_parent_tag(row):
        return parent_tag_dict.get(row.ID, np.nan)

    # Apply the function to each row in the DataFrame
    df['parent_tag'] = df.apply(find_parent_tag, axis=1)

    return df

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

    for i in tqdm(range(0,len(pd)),total=len(pd),desc='Correcting first destruction',disable=True):

        row = pd.iloc[i]
        #print(row)
        enlosed_i = row['enclosed_i']
        if len(enlosed_i) > 1: 
            new_row = row.copy()
            
            new_row['Death'] = pd.loc[pd['ID'] == enlosed_i[1]]['Death']
            new_row['parent_tag'] = pd.loc[pd['ID'] == enlosed_i[1]]['ID']
            ## this accounts for a bug were the entire series is placed in the death column.
            # not sure on the origin of this but the following corrects for it. It only occationally happends so this is not 
            # computationally expensive.
            if type(new_row['Death']) == pandas.core.series.Series:
                # get the Death value from the first item in the Series.
                new_row['Death'] = new_row['Death'].iloc[0] # 
          
            new_row['new_row'] = 1
            new_row['ID'] = pd['ID'].max() + 1
            new_row['enclosed_i'] = []
            
            
            pd = pd.append(new_row,ignore_index=True)
            
    return pd




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
    mask = utils.get_mask_GPU(Birth,Death,row.x1,row.y1,img_gpu)
    # get bounding box here 
    # evalute if mask is True on an edge.
    bounding_box = utils.bounding_box_gpu(mask)
    mask = mask.get()
    edge = utils.check_edge(mask)
    if edge:
        edge = 1
    
    area = np.sum(mask)
    return area, edge, bounding_box


def calculate_area_CPU(Birth,Death,row, img):
    
    mask = utils.get_mask_CPU(row.x1,row.y1,Birth,Death,img)
    # get bounding box here
    bounding_box = utils.bounding_box_cpu(mask)
    edge = utils.check_edge(mask)
    if edge:
        edge = 1
    area = np.sum(mask)
    return area, edge, bounding_box

    
def compute_ph_components(img,local_bg,analysis_threshold_val,lifetime_limit,
                          output=False,bg_map=False,area_limit=3,GPU=False,lifetime_limit_fraction=2,
                          mean_bg=None, IDoffset=0, box_size=None,detection_threshold=None):
    
    global GPU_Option
    GPU_Option = GPU
    print('Computing PH components...ls ')
    t0_compute_ph = time.time()
    pd = cripser.computePH(-img,maxdim=0)
    t1_compute_ph = time.time()
    print('Time to compute PH: {}'.format(t1_compute_ph-t0_compute_ph))
    pd = pandas.DataFrame(pd,columns=['dim','Birth','Death','x1','y1','z1','x2','y2','z2'],index=range(1,len(pd)+1))
    pd.drop(columns=['dim','z1','z2'],inplace=True)
    pd['lifetime'] = pd['Death'] - pd['Birth']
    pd['Birth'] = -pd['Birth'] 
    pd['Death'] = -pd['Death']
     
    #print("mean_bg: ",mean_bg)
    # get rid of birth less than 0, helps speed up the code alittle.
    mean_bg_temp = np.nanmean(local_bg)/detection_threshold
    #print('mean_bg: ',mean_bg_temp)
    # get rid of alot of defintly not sources.
    pd = pd[pd['Birth']>mean_bg_temp]

    pd['bg'] = 0
    pd['edge_flag'] = 0
    pd['mean_bg'] = 0   # this is the mean of the background
    # assign each row the local bg valuw from the map.
    if bg_map:
        
        # for each row we need to assign the local bg value.
        
        for index, row in pd.iterrows():
            
            pd.loc[index,'bg'] = background.get_bg_value_from_result_image((int(row.x1),int(row.y1)), box_size, local_bg)
            #print('Detection_Thresh: ',row['bg'])
            #print('Birth :',row['Birth'])
            pd.loc[index,'mean_bg'] = background.get_bg_value_from_result_image((int(row.x1),int(row.y1)), box_size, mean_bg)
            #print(row['mean_bg'])
            # evaluate if the death value is below the analysis threshold value at the birth point.
            # if it is then set the death value to the analysis threshold value.
            
            Anal_val = background.get_bg_value_from_result_image((int(row.x1),int(row.y1)), box_size, analysis_threshold_val)
            
            if row['Death'] < Anal_val:
                pd.loc[index,'Death'] = Anal_val
            
                
        
    else:
        # no bg map so just assign the local bg value. asn this should be single value.
        pd['bg'] = local_bg
        pd['mean_bg'] = mean_bg
        
        for index, row in pd.iterrows():
            if row['Death'] < analysis_threshold_val:
                pd.loc[index,'Death'] = analysis_threshold_val
    
    #print('Before Cull',len(pd))
    #print('mean_bg: ',np.mean(pd['bg']))
    pd = pd[pd['Birth'] > pd['bg']] # maybe this should be at the beginning.
    # also make sure Death is less than Birth
    #pd = pd[pd['Death'] < pd['Birth']]
    #print('After Cull',len(pd))    
    # if bg_map:
        
    #     list_of_index_to_drop = []
    
        
    #     for index, row in pd.iterrows():
    #         # check if local_bg is a map or a value
    #         if row['Birth'] < local_bg[int(row.x1),int(row.y1)]:
    #             list_of_index_to_drop.append(index)
    
    #     pd.drop(list_of_index_to_drop,inplace=True)
       
    
    #     # for each row evaluate if death is below analysis thresholdval map value at its birth point. if its below then set Death to bg map value.    
    #     for index, row in pd.iterrows():
    #         Analy_val = analysis_threshold_val[int(row.y1),int(row.x1)]
    #         if row['Death'] < Analy_val:
    #             row['Death'] = Analy_val
    #         # assign each row the local bg value
    #         row['bg'] = local_bg[int(row.y1),int(row.x1)]
    #         row['mean_bg'] = mean_bg[int(row.y1),int(row.x1)]
            
    # else:
    #     print(local_bg)
    #     pd = pd[pd['Birth']>local_bg] # maybe this should be at the beginning.
    #     pd['Death'] = np.where(pd['Death'] < analysis_threshold_val, analysis_threshold_val, pd['Death'])
    #     pd['bg'] = local_bg
    #     pd['mean_bg'] = mean_bg
        
    pd['lifetime'] = abs(pd['Death'] - pd['Birth'])
    
    pd['lifetimeFrac'] = pd['Birth']/pd['Death']
    pd = pd[pd['lifetimeFrac']>lifetime_limit_fraction]
    pd = pd[pd['lifetime'] > lifetime_limit]
    pd.sort_values(by='lifetime',ascending=False,inplace=True,ignore_index=True)
    
    pd['ID'] = pd.index + IDoffset
    
    
    # begins here
    
    if len(pd) > 0:
        
        area_list=[]
        edge_list=[]
        bbox1 = []
        bbox2 = []
        bbox3 = []
        bbox4 = []

        if GPU_Option ==True:
            if GPU_AVAILABLE == True:
                img_gpu = cp.asarray(img,dtype=cp.float64)
                # Calculate area and enforce area limit Single Process.
                #print('Calculating area with GPU...')
                t0 = time.time()
                
                for i in tqdm(range(0,len(pd)),total=len(pd),desc='Calculating area',disable=not output):
                    
                    row = pd.iloc[i]
                    Birth = row.Birth
                    Death = row.Death
                    area, edge, bbox = calculate_area_GPU(Birth,Death,row,img_gpu)
                    area_list.append(area)
                    edge_list.append(edge)
                    bbox1.append(bbox[0].get())
                    bbox2.append(bbox[1].get())
                    bbox3.append(bbox[2].get())
                    bbox4.append(bbox[3].get())
                    
                t1 = time.time()
                #print('Time to calculate area and inital bbox: {}'.format(t1-t0))
                pd['area'] = area_list
                pd['edge_flag'] = edge_list
                pd['bbox1'] = bbox1
                pd['bbox2'] = bbox2
                pd['bbox3'] = bbox3
                pd['bbox4'] = bbox4
                pd = pd[pd['area']>area_limit]
                
        else:
            
            #print('Calculating area with CPU...')
            t0 = time.time()
            for i in tqdm(range(0,len(pd)),total=len(pd),desc='Calculating area',disable=not output):
                
                row = pd.iloc[i]
                Birth = row.Birth
                Death = row.Death
                area, edge, bbox = calculate_area_CPU(Birth,Death,row,img)
                area_list.append(area)
                edge_list.append(edge)
                bbox1.append(bbox[0])
                bbox2.append(bbox[1])
                bbox3.append(bbox[2])
                bbox4.append(bbox[3])

            t1 = time.time()
            #print('Time to calculate area and inital bbox: {}'.format(t1-t0))
            pd['area'] = area_list
            pd['edge_flag'] = edge_list
            pd['bbox1'] = bbox1
            pd['bbox2'] = bbox2
            pd['bbox3'] = bbox3
            pd['bbox4'] = bbox4
            pd = pd[pd['area']>area_limit]
        
    return pd
        