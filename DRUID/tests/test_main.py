from DRUID.main import sf
import pytest

def test_sf():
    '''
    Test initalisation of the main Class
    '''
    assert sf(image=None, image_path=None,mode="Radio",pb_path=None)
    
    