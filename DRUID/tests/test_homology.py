"""
Unit tests for Persistent Homology computation (Cripser & Polars).
"""
import pytest
import numpy as np
import polars as pl
from DRUID.src.homology import compute_homology

def test_compute_homology_schema():
    """
    Test that compute_homology returns the mathematically expected schema.
    """
    image = np.zeros((50, 50))
    # Add a topological maximum (birth)
    image[20:30, 20:30] = 10.0
    image[25, 25] = 20.0 
    
    df = compute_homology(
        image, 
        analysis_threshold=1.0, 
        lifetime_limit=0.1,
        area_size_threshold=2
    )
    
    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()
    
    expected_cols = {
        "birth", "death", "x1", "y1", "lifetime", "lifetimeFrac",
        "area", "bbox_min_y", "ID", "encloses", "parent_tag", "contour"
    }
    assert expected_cols.issubset(set(df.columns))
    
    # Check that birth is strictly greater than death
    assert df.filter(pl.col("birth") <= pl.col("death")).is_empty()