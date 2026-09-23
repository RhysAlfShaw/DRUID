"""
Unit tests for initial source island thresholding and bounding box generation.
"""
import pytest
import numpy as np
from DRUID.src.source import create_source_islands

def test_create_source_islands():
    """
    Test extraction of standard and massive source islands based on thresholds.
    """
    image = np.zeros((100, 100))
    bg = np.ones((100, 100))
    rms = np.ones((100, 100)) * 0.5
    
    # Create one standard source (area ~ 25)
    image[20:25, 20:25] = 10 
    # Create one massive source (area = 400)
    image[50:70, 50:70] = 10 
    
    # threshold = 1 + (3 * 0.5) = 2.5
    islands = create_source_islands(
        image, bg, rms, 
        analysis_threshold=3, area_limit=10, max_area_limit=200, verbose=False
    )
    
    # Check standard queue
    assert len(islands["bboxes"]) == 1
    assert len(islands["positions"]) == 1
    
    # Check massive queue
    assert len(islands["massive_bboxes"]) == 1
    assert len(islands["massive_positions"]) == 1