"""
Integration tests for the DRUID main pipeline.
"""
import pytest
import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter
from DRUID.main import sf, _worker
import DRUID.main as main_module 

@pytest.fixture
def pipeline_image():
    """Provides a baseline image with a single synthetic source."""
    img = np.random.normal(5, 0.5, size=(100, 100))
    img[45:55, 45:55] += 15.0 # Bright source
    return img

def test_pipeline_sequential_with_smoothing(pipeline_image):
    """Test end-to-end pipeline with smooth_sigma > 0."""
    finder = sf(image=pipeline_image, verbose=False, num_threads=1, smooth_sigma=1.0)
    finder.set_background(analysis_threshold=3)
    finder.phsf()
    
    assert hasattr(finder, "catalog")
    assert isinstance(finder.catalog, pl.DataFrame)
    assert not finder.catalog.is_empty()
    assert "ID" in finder.catalog.columns
    # Ensure smoothed_image was allocated
    assert getattr(finder, "smoothed_image", None) is not None

def test_worker_function(pipeline_image):
    """
    Test the inner multiprocessing worker with raw and smoothed global arrays.
    """
    bg = np.ones((100, 100)) * 5.0
    rms = np.ones((100, 100)) * 0.5
    smoothed_image = gaussian_filter(pipeline_image, sigma=1.0)
    
    # Inject into the main module namespace 
    main_module.global_image = pipeline_image
    main_module.global_smoothed_image = smoothed_image
    main_module.global_background_map = bg
    main_module.global_background_rms_map = rms
    
    bbox = (40, 40, 60, 60)
    position = (40, 40)
    island_info = (bbox, position)
    
    result_cat = _worker(
        island_info,
        analysis_threshold=3.0,
        lifetime_limit=0.0,
        lifetime_limit_fraction=1.0,
        mode="radio", BMAJ=2.0, BMIN=2.0
    )
    
    assert isinstance(result_cat, pl.DataFrame)
    assert not result_cat.is_empty()
    
    # Cleanup namespace
    main_module.global_image = None
    main_module.global_smoothed_image = None
    main_module.global_background_map = None
    main_module.global_background_rms_map = None