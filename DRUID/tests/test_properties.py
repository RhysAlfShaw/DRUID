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
    Test integration of skimage.measure with Polars catalogs.
    """
    image = np.zeros((30, 30))
    bg = np.zeros((30, 30))
    rms = np.ones((30, 30))
    
    image[10:20, 10:20] = 5.0
    
    # Mock homology dataframe row
    cat = pl.DataFrame({
        "birth": [5.1],
        "death": [0.0],
        "x1": [15],
        "y1": [15],
        "area": [100]
    })
    
    result = calculate_properties(
        cat, image, bg, rms, position=(0,0), 
        analysis_threshold=1.0, mode="radio", BMAJ=2.0, BMIN=2.0
    )
    
    assert "flux" in result.columns
    assert "maj" in result.columns
    assert "snr" in result.columns
    np.testing.assert_allclose(result["flux"][0], 500.0, atol=1e-3)