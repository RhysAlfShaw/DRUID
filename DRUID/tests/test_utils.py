import pytest 
import numpy as np
from DRUID.src.utils.utils import smoothing, get_region_props

test_image = np.array([[1,2,3,4,5],
                        [6,7,8,9,10],
                        [11,12,13,14,15],
                        [16,17,18,19,20],
                        [21,22,23,24,25]])

def test_smoothing_works():
    # test that the smoothing function returns the correct shape image that went in.
    
    assert smoothing(test_image, 3).shape == (5,5)
  

mask = np.array([[0,0,0,0,0],
                [0,1,1,1,0],
                [0,1,1,1,0],
                [0,1,1,1,0],
                [0,0,0,0,0]])
def test_regionprops_function_params():
    # test that the regionprops function returns the correct shape image that went in.
    
    region = get_region_props(mask, test_image)[0]
     
    assert region.area == 9
    assert region.centroid == (2.0,2.0)
    assert region.max_intensity == 19
    assert region.major_axis_length == pytest.approx(3.2659, 0.0001)
    assert region.minor_axis_length == pytest.approx(3.2659, 0.0001)
    assert region.orientation == pytest.approx(0.7853, 0.001)
    
    