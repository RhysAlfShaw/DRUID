"""
Integration tests for the DRUID main pipeline.
"""
import pytest
import numpy as np
import polars as pl
from DRUID.main import sf, _worker
import DRUID.main as main_module # Imported to mock globals

@pytest.fixture
def pipeline_image():
    """Provides a baseline image with a single synthetic source."""
    img = np.random.normal(5, 0.5, size=(100, 100))
    img[45:55, 45:55] += 15.0 # Bright source
    return img

def test_pipeline_sequential(pipeline_image):
    """Test end-to-end pipeline executing sequentially."""
    finder = sf(image=pipeline_image, verbose=False, num_threads=1, cashe=False)
    finder.set_background(analysis_threshold=3)
    finder.phsf()
    
    assert hasattr(finder, "catalog")
    assert isinstance(finder.catalog, pl.DataFrame)
    assert not finder.catalog.is_empty()
    assert "ID" in finder.catalog.columns
    assert finder.catalog["flux_peak"].max() > 10.0

def test_worker_function(pipeline_image):
    """
    Test the inner multiprocessing worker. 
    Requires binding module-level globals to simulate shared memory attachment.
    """
    # 1. Setup mock data
    bg = np.ones((100, 100)) * 5.0
    rms = np.ones((100, 100)) * 0.5
    
    # 2. Inject into the main module namespace (simulating _worker_init)
    main_module.global_image = pipeline_image
    main_module.global_background_map = bg
    main_module.global_background_rms_map = rms
    
    bbox = (40, 40, 60, 60) # min_row, min_col, max_row, max_col
    position = (40, 40)
    island_info = (bbox, position)
    
    # 3. Execute worker
    result_cat = _worker(
        island_info,
        analysis_threshold=3.0,
        lifetime_limit=0.0,
        lifetime_limit_fraction=1.0,
        mode="radio", BMAJ=2.0, BMIN=2.0
    )
    
    assert isinstance(result_cat, pl.DataFrame)
    assert not result_cat.is_empty()
    assert "Island_X" in result_cat.columns
    assert "Island_Y" in result_cat.columns
    
    # 4. Cleanup namespace
    main_module.global_image = None
    main_module.global_background_map = None
    main_module.global_background_rms_map = None