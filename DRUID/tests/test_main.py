from DRUID.main import sf
import pytest
import numpy as np

def test_sf():
    '''
    Test initalisation of the main Class
    '''
    arr2d = np.random.rand(100,100)
    assert sf(image=arr2d, image_path=None,
              mode="Radio",pb_path=None,cutup=False,
              cutup_size=None,cutup_buff=None,output=False,
              area_limit=5,smooth_sigma=1,nproc=1,GPU=True,
              header=None,Xoff=None,Yoff=None) != None
    
    