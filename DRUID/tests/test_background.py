from DRUID.src.background.background import calculate_background
import pytest
import numpy as np
# test the use of the calculate_background function

# prehaps use a sample more realistic array with cahracteristics of Radio and Optical data.
test_array = np.array([[1,2,3,4,5],
                        [6,7,8,9,10],
                        [11,12,13,14,15],
                        [16,17,18,19,20],
                        [21,22,23,24,25]])
def test_calculate_background_not_valid():
    # test that a parameter not valid will fail
    with pytest.raises(ValueError):
        calculate_background(np.ones((10,10)), mode='not_valid')

def test_mad_std_value():
    # test that the mad_std value is correct
    # assert has to be approximately equal to the value because of float
    
    assert calculate_background(test_array, mode='mad_std')[0]== pytest.approx(8.8956, 0.001) 
    
def test_rms_std_value():
    # test that the rms_std value is correct
    # assert has to be approximately equal to the value because of float
    assert calculate_background(test_array, mode='rms')[0]== pytest.approx(14.8660, 0.001)
    
def test_mean_value():
    # test that the mean value is correct
    # assert has to be approximately equal to the value because of float
    assert calculate_background(test_array, mode='mad_std')[1]== pytest.approx(13.0, 0.001)