"""
Unit tests for background and RMS map estimation.
"""
import pytest
import numpy as np
from astropy.io import fits
from photutils.background import MedianBackground
from DRUID.src.background import make_source_mask, calculate_background_maps

@pytest.fixture
def dummy_fits_file(tmp_path) -> str:
    """Fixture to generate a dummy FITS file with a noise floor."""
    rng = np.random.default_rng(42) # Reproducible seed
    data = rng.uniform(5, 15, size=(100, 100))
    file_path = tmp_path / "dummy_image.fits"
    fits.PrimaryHDU(data).writeto(file_path)
    return str(file_path)

def test_calculate_background_maps_defaults(dummy_fits_file):
    """Test background map generation with default estimators."""
    bg_map, rms_map = calculate_background_maps(dummy_fits_file)
    
    assert bg_map.shape == (100, 100)
    assert rms_map.shape == (100, 100)
    assert bg_map.dtype == np.float64
    assert np.all(bg_map > 0)
    assert np.all(rms_map > 0)

def test_calculate_background_maps_array_input():
    """Test background map generation passing a NumPy array directly."""
    rng = np.random.default_rng(42)
    data = rng.normal(10, 1, size=(50, 50))
    bg_map, rms_map = calculate_background_maps(data, bg_estimator=MedianBackground())
    
    assert bg_map.shape == (50, 50)
    np.testing.assert_allclose(np.median(bg_map), 10.0, rtol=0.1)