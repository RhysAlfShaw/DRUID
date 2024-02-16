import pytest 
import numpy as np
from astropy.io import fits
from DRUID.src.utils import smoothing, get_region_props, model_beam_func, flux_correction_factor, bounding_box_cpu, open_image, calculate_beam, xy_to_RaDec, generate_2d_gaussian
import pandas 

PATH_test_image_file = 'https://drive.google.com/uc?id=10a6goXcr6wEHX9U5LQ07cCEo2nGQ9QK5'



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
    assert abs(region.orientation) == pytest.approx(0.7853, 0.001)

def test_model_beam_function():
    test_beam =  np.array([[0.00193045, 0.01426423, 0.03877421, 0.03877421, 0.01426423],
                    [0.01426423, 0.10539922, 0.2865048 , 0.2865048 , 0.10539922],
                 [0.03877421, 0.2865048 , 0.77880078, 0.77880078, 0.2865048 ],
                 [0.03877421, 0.2865048 , 0.77880078, 0.77880078, 0.2865048 ],
                 [0.01426423, 0.10539922, 0.2865048 , 0.2865048 , 0.10539922]])
    model_beam = model_beam_func(1,(5,5),2.5,2.5,1,1,0)
    # make sure the model beam and the test beam are the same.
    assert np.allclose(model_beam, test_beam, rtol=1e-05, atol=1e-08)    

def test_flux_correction_factor():
    
    # test if the correction value returned is correct.
    mask = np.array([[0,0,0,0,0],
                    [0,1,1,1,0],
                    [0,1,1,1,0],
                    [0,1,1,1,0],
                    [0,0,0,0,0]])
    model_beam = model_beam_func(1,(5,5),2.5,2.5,1,1,0)# gaussian beam model of sixe 5x5.
    assert flux_correction_factor(mask, model_beam) == pytest.approx(1.38389, 0.0001)
    
def test_bouding_box_cpu():
    # test if the bounding box function returns the correct shape image that went in.
    mask = np.array([[0,0,0,0,0],
                    [0,1,1,1,0],
                    [0,1,1,1,0],
                    [0,1,1,1,0],
                    [0,0,0,0,0]])
    bounding_box = bounding_box_cpu(mask)
    assert bounding_box == (1,1,3,3)
    
def test_open_image():
    image, header = open_image(PATH_test_image_file)
    assert image.shape == (256,256)
    assert type(header) == fits.header.Header
    
def test_calculate_beam():
    image, header = open_image(PATH_test_image_file)
    header['BMAJ'] = 0.0001388888888888889
    header['BMIN'] = 0.0001388888888888889
    header['BPA'] = 0.0
    beam,bmaj,bmin,bpa = calculate_beam(header)
    assert beam == pytest.approx(0.28327, 0.0001)
    assert bmaj == pytest.approx(0.5000, 0.0001)
    assert bmin == pytest.approx(0.5000, 0.0001)
    assert bpa == pytest.approx(0.0, 0.0001)
    
def test_xy_to_RaDec():
    x,y = 128,128
    image, header = open_image(PATH_test_image_file)
    ra, dec = xy_to_RaDec(128,128,header,mode='Radio')
    assert ra == pytest.approx(230.68, 0.0001)
    assert dec == pytest.approx(54.64416, 0.0001)

