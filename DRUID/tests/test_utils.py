import pytest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from astropy.io import fits

from DRUID.src.utils import get_image_from_path, combine_polars_catalogs


@pytest.fixture
def create_fits_file(tmp_path):
    """A fixture to create a FITS file with given data."""

    def _create_fits(data, filename="test.fits"):
        file_path = tmp_path / filename
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(file_path, overwrite=True)
        return str(file_path)

    return _create_fits


def test_get_image_from_path_2d(create_fits_file):
    """Test loading a standard 2D FITS image."""
    image_data = np.arange(100, dtype=np.float32).reshape(10, 10)
    fits_path = create_fits_file(image_data)

    loaded_image = get_image_from_path(fits_path)

    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.shape == (10, 10)
    assert np.array_equal(loaded_image, image_data)


def test_get_image_from_path_3d_squeezes(create_fits_file):
    """Test that a 3D FITS image is correctly squeezed to 2D."""
    image_data = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
    fits_path = create_fits_file(image_data)

    loaded_image = get_image_from_path(fits_path)

    assert loaded_image.shape == (10, 10)
    assert np.array_equal(loaded_image, image_data.squeeze())


def test_get_image_from_path_4d_squeezes(create_fits_file):
    """Test that a 4D FITS image is correctly squeezed to 2D."""
    image_data = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    fits_path = create_fits_file(image_data)

    loaded_image = get_image_from_path(fits_path)

    assert loaded_image.shape == (10, 10)
    assert np.array_equal(loaded_image, image_data.squeeze())


def test_get_image_from_path_file_not_found():
    """Test that an error is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        get_image_from_path("non_existent_file.fits")


def test_combine_polars_catalogs_basic():
    """Test combining a list of simple Polars DataFrames."""
    cat1 = pl.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    cat2 = pl.DataFrame({"A": [3, 4], "B": ["z", "w"]})
    catalogs = [cat1, cat2]

    combined = combine_polars_catalogs(catalogs)

    expected = pl.DataFrame({"A": [1, 2, 3, 4], "B": ["x", "y", "z", "w"]})

    assert_frame_equal(combined, expected)
    assert combined.shape == (4, 2)


def test_combine_polars_catalogs_with_uppercase_id():
    """Test that a column named 'ID' (uppercase) is not re-indexed."""
    cat1 = pl.DataFrame({"ID": [0, 1], "data": [10, 20]})
    cat2 = pl.DataFrame({"ID": [0, 1], "data": [30, 40]})
    catalogs = [cat1, cat2]

    combined = combine_polars_catalogs(catalogs)

    # The 'ID' column should remain as is, with duplicates
    expected = pl.DataFrame({"ID": [0, 1, 0, 1], "data": [10, 20, 30, 40]})
    assert_frame_equal(combined, expected)


def test_combine_polars_catalogs_empty_list():
    """Test that combining an empty list of catalogs raises a ValueError."""
    with pytest.raises(ValueError, match="No catalogs provided to combine."):
        combine_polars_catalogs([])
