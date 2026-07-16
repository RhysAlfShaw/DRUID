"""
Unit tests for source property calculations (flux, SNR, geometry).
"""
import pytest
import numpy as np
import polars as pl
from DRUID.src.properties import calculate_properties, calculate_radio_flux_error

def test_calculate_radio_flux_error():
    """Test flux error analytic derivation."""
    bg_rms = np.array([1.0, 1.0, 1.0])
    area = 100
    bmaj, bmin = 2.0, 2.0
    
    err = calculate_radio_flux_error(bg_rms, area, bmaj, bmin)
    assert not np.isnan(err)
    assert err > 0

def test_calculate_properties():
    """
    Test properties calculated on raw image while mask bounds dictate via smoothed image.
    """
    raw_image = np.zeros((30, 30))
    smoothed_image = np.zeros((30, 30))
    bg = np.zeros((30, 30))
    rms = np.ones((30, 30))
    
    # Simulate a bright sharp core in raw image
    raw_image[10:20, 10:20] = 5.0 
    
    # Simulate a wider, dimmer dispersion in smoothed image
    smoothed_image[8:22, 8:22] = 2.0 
    
    cat = pl.DataFrame({
        "birth": [2.1], # Mask encompasses smoothed_image's 2.0 block
        "death": [0.0],
        "x1": [15],
        "y1": [15],
        "area": [196] # (14 x 14 block area)
    })
    
    result = calculate_properties(
        cat, raw_image=raw_image, smoothed_image=smoothed_image, 
        background=bg, background_rms=rms, position=(0,0), 
        analysis_threshold=1.0, mode="radio", BMAJ=2.0, BMIN=2.0
    )
    
    # Peak flux should pull from the 5.0 raw array, NOT the 2.0 smoothed one
    assert result["flux_peak"][0] == 5.0
    
    # Total flux is the 10x10 core inside the 14x14 bounds 
    # (100 pixels * 5.0) + (96 pixels * 0.0) = 500
    np.testing.assert_allclose(result["flux"][0], 500.0, atol=1e-3)