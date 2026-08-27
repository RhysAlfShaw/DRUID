"""
Unit tests for DRUID utilities.
"""
import pytest
import numpy as np
import polars as pl
from DRUID.src.utils import combine_polars_catalogs, generate_2d_gaussian

def test_combine_polars_catalogs():
    """
    Test the concatenation and dense ranking of catalog IDs.
    """
    df1 = pl.DataFrame({"id": [1, 2], "flux": [10.5, 20.1]})
    df2 = pl.DataFrame({"id": [1, 2], "flux": [30.4, 40.2]})
    
    combined = combine_polars_catalogs([df1, df2])
    
    assert combined.height == 4
    # Ensure dense ranking re-indexes the IDs sequentially
    assert combined["id"].to_list() == [1, 2, 3, 4]

def test_combine_polars_catalogs_empty():
    """Ensure a ValueError is raised for empty catalog lists."""
    with pytest.raises(ValueError, match="No catalogs provided"):
        combine_polars_catalogs([])

def test_generate_2d_gaussian():
    """
    Test the 2D Gaussian generation for mathematical correctness.
    """
    shape = (50, 50)
    center = (25, 25)
    sigma = 5
    
    gauss = generate_2d_gaussian(
        A=1.0, shape=shape, center=center, 
        sigma_x=sigma, sigma_y=sigma, norm=False
    )
    
    assert gauss.shape == shape
    np.testing.assert_allclose(gauss[25, 25], 1.0, atol=1e-5)
    # Check symmetric decay
    np.testing.assert_allclose(gauss[20, 25], gauss[30, 25], atol=1e-5)