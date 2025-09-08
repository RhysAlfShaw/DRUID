import pytest
import numpy as np
from astropy.io import fits
from photutils.background import MedianBackground, StdBackgroundRMS
from DRUID.src.background import (
    make_source_mask,
    calculate_background_maps,
    make_gaussian_sources_image,
)
import os


@pytest.fixture
def dummy_fits_file(tmp_path):
    """Creates a dummy FITS file with a simple image."""
    data = np.random.rand(100, 100) * 10 + 5  # Random data with some offset
    hdu = fits.PrimaryHDU(data)
    file_path = tmp_path / "dummy_image.fits"
    hdu.writeto(file_path)
    return file_path


@pytest.fixture
def dummy_fits_file_with_source(tmp_path):
    """Creates a dummy FITS file with a simple image and a source."""
    image_size = (100, 100)
    sources = [
        {
            "amplitude": 100,
            "x_mean": 50,
            "y_mean": 50,
            "x_stddev": 5,
            "y_stddev": 5,
            "theta": 0,
        }
    ]
    data = make_gaussian_sources_image(image_size, sources)
    data += np.random.normal(0, 1, size=image_size)  # Add noise
    hdu = fits.PrimaryHDU(data)
    file_path = tmp_path / "dummy_image_with_source.fits"
    hdu.writeto(file_path)
    return file_path


def test_make_source_mask_with_sources(dummy_fits_file_with_source):
    """Test make_source_mask with an image containing a known source."""
    with fits.open(dummy_fits_file_with_source) as hdul:
        data = hdul[0].data
    mask = make_source_mask(data, nsigma=3.0, kernel_size=3)
    assert mask.shape == data.shape
    assert np.any(mask)  # Expect some sources to be masked


def test_calculate_background_maps_defaults(dummy_fits_file_with_source):
    """Test calculate_background_maps with default parameters."""

    background_map, background_rms_map = calculate_background_maps(
        str(dummy_fits_file_with_source)
    )

    with fits.open(dummy_fits_file_with_source) as hdul:
        data_shape = hdul[0].data.shape

    assert background_map.shape == data_shape
    assert background_rms_map.shape == data_shape
    assert isinstance(background_map, np.ndarray)
    assert isinstance(background_rms_map, np.ndarray)


def test_calculate_background_maps_custom_estimator_str(dummy_fits_file_with_source):
    """Test calculate_background_maps with a string-specified background estimator."""
    background_map, background_rms_map = calculate_background_maps(
        str(dummy_fits_file_with_source), bg_estimator="mean"
    )
    with fits.open(dummy_fits_file_with_source) as hdul:
        data_shape = hdul[0].data.shape
    assert background_map.shape == data_shape
    assert background_rms_map.shape == data_shape


def test_calculate_background_maps_custom_estimator_obj(dummy_fits_file_with_source):
    """Test calculate_background_maps with a BackgroundBase object estimator."""
    custom_estimator = MedianBackground()
    background_map, background_rms_map = calculate_background_maps(
        str(dummy_fits_file_with_source), bg_estimator=custom_estimator
    )
    with fits.open(dummy_fits_file_with_source) as hdul:
        data_shape = hdul[0].data.shape
    assert background_map.shape == data_shape
    assert background_rms_map.shape == data_shape


def test_calculate_background_maps_invalid_estimator_str(dummy_fits_file_with_source):
    """Test calculate_background_maps with an invalid string-specified background estimator,
    expecting it to default to MedianBackground."""
    background_map, background_rms_map = calculate_background_maps(
        str(dummy_fits_file_with_source), bg_estimator="not_an_estimator"
    )
    with fits.open(dummy_fits_file_with_source) as hdul:
        data_shape = hdul[0].data.shape
    assert background_map.shape == data_shape
    assert background_rms_map.shape == data_shape
    # Further checks could involve inspecting the bkg_estimator used if it were returned or logged


def test_calculate_background_maps_file_not_found(tmp_path):
    """Test calculate_background_maps with a non-existent FITS file."""
    non_existent_file = tmp_path / "non_existent.fits"
    with pytest.raises(FileNotFoundError):
        calculate_background_maps(str(non_existent_file))


def test_make_gaussian_sources_image_no_sources():
    """Test make_gaussian_sources_image with an empty list of sources."""
    image_size = (50, 50)
    sources = []
    image = make_gaussian_sources_image(image_size, sources)
    assert image.shape == image_size
    assert np.all(image == 0)


def test_make_gaussian_sources_image_single_source():
    """Test make_gaussian_sources_image with a single source."""
    image_size = (100, 100)
    sources = [
        {
            "amplitude": 50,
            "x_mean": 25,
            "y_mean": 25,
            "x_stddev": 3,
            "y_stddev": 3,
            "theta": 0,
        }
    ]
    image = make_gaussian_sources_image(image_size, sources)
    assert image.shape == image_size
    assert np.sum(image) > 0  # Check that the source contributes to the image
    # Check peak value is close to amplitude (could be affected by pixel grid)
    assert np.isclose(np.max(image), sources[0]["amplitude"], atol=1)
