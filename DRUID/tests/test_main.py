import pytest
import numpy as np
import polars as pl
from astropy.io import fits
import os
import shutil

from DRUID.main import sf, _worker
from DRUID.src.background import make_gaussian_sources_image


@pytest.fixture
def simple_image_data():
    """Creates a simple 100x100 image with one Gaussian source and noise."""
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
    image = make_gaussian_sources_image(image_size, sources)
    image += np.random.normal(5, 1, size=image_size)  # Add background and noise
    return image


@pytest.fixture
def empty_image_data():
    """Creates an empty 100x100 image with just noise."""
    return np.random.normal(5, 1, size=(100, 100))


@pytest.fixture
def simple_fits_file(tmp_path, simple_image_data):
    """Creates a dummy FITS file with a simple image."""
    file_path = tmp_path / "simple_image.fits"
    hdu = fits.PrimaryHDU(simple_image_data)
    hdu.writeto(file_path)
    return str(file_path)


def test_sf_init_with_numpy_array(simple_image_data):
    """Test sf initialization with a NumPy array."""
    finder = sf(image=simple_image_data, verbose=False)
    assert isinstance(finder.image, np.ndarray)
    assert np.array_equal(finder.image, simple_image_data)


def test_sf_init_with_fits_path(simple_fits_file, simple_image_data):
    """Test sf initialization with a FITS file path."""
    finder = sf(image=simple_fits_file, verbose=False)
    assert isinstance(finder.image, np.ndarray)
    assert np.array_equal(finder.image, simple_image_data)


def test_sf_init_no_image_raises_error():
    """Test that sf initialization raises ValueError if no image is provided."""
    with pytest.raises(ValueError, match="No image provided"):
        sf(verbose=False)


def test_sf_init_invalid_path_raises_error():
    """Test that sf initialization raises ValueError for an invalid file path."""
    with pytest.raises(ValueError, match="Could not load image from path"):
        sf(image="non_existent_file.fits", verbose=False)


def test_sf_init_invalid_type_raises_error():
    """Test that sf initialization raises TypeError for an invalid image type."""
    with pytest.raises(TypeError, match="Image must be a file path"):
        sf(image=12345, verbose=False)


def test_set_background(simple_image_data):
    """Test the set_background method."""
    finder = sf(image=simple_image_data, verbose=False)
    finder.set_background(detection_threshold=5, analysis_threshold=3)
    assert hasattr(finder, "background_map")
    assert hasattr(finder, "background_rms_map")
    assert finder.background_map.shape == simple_image_data.shape
    assert finder.background_rms_map.shape == simple_image_data.shape
    assert finder.detection_threshold == 5
    assert finder.analysis_threshold == 3


def test_set_background_caching(simple_image_data, tmp_path):
    """Test the caching mechanism of the set_background method."""
    cache_dir = tmp_path / "druid_cache"
    os.makedirs(cache_dir)

    # First run, should calculate and save
    finder1 = sf(
        image=simple_image_data,
        verbose=False,
        cashe=True,
        working_directory=str(cache_dir),
    )
    finder1.set_background()

    bg_map_path = cache_dir / "background_map.npy"
    bg_rms_map_path = cache_dir / "background_rms_map.npy"

    assert bg_map_path.exists()
    assert bg_rms_map_path.exists()

    # Second run, should load from cache
    finder2 = sf(
        image=simple_image_data,
        verbose=False,
        cashe=True,
        working_directory=str(cache_dir),
    )
    finder2.set_background()

    assert np.array_equal(finder1.background_map, finder2.background_map)
    assert np.array_equal(finder1.background_rms_map, finder2.background_rms_map)

    shutil.rmtree(cache_dir)


# def test_phsf_raises_error_if_no_background(simple_image_data):
#     """Test that phsf raises ValueError if background is not set."""
#     finder = sf(image=simple_image_data, verbose=False)
#     with pytest.raises(ValueError, match="Background map and RMS map must be set"):
#         finder.phsf()


def test_phsf_sequential(simple_image_data):
    """Test phsf with sequential processing (num_threads=1)."""
    finder = sf(image=simple_image_data, verbose=False, num_threads=1)
    finder.set_background(analysis_threshold=3)
    finder.phsf()

    assert hasattr(finder, "catalog")
    assert isinstance(finder.catalog, pl.DataFrame)
    assert not finder.catalog.is_empty()
    assert "ID" in finder.catalog.columns
    assert "birth" in finder.catalog.columns


@pytest.mark.skipif(os.cpu_count() < 2, reason="Test requires at least 2 CPU cores")
def test_phsf_parallel(simple_image_data):
    """Test phsf with parallel processing (num_threads > 1)."""
    finder = sf(image=simple_image_data, verbose=False, num_threads=2)
    finder.set_background(analysis_threshold=3)
    finder.phsf()

    assert hasattr(finder, "catalog")
    assert isinstance(finder.catalog, pl.DataFrame)
    assert not finder.catalog.is_empty()
    assert "ID" in finder.catalog.columns


def test_phsf_no_sources_found(empty_image_data):
    """Test phsf on an image with no sources, expecting an empty catalog."""
    finder = sf(image=empty_image_data, verbose=False)
    # Set a high threshold to ensure no sources are found
    finder.set_background(analysis_threshold=100)
    finder.phsf()

    assert hasattr(finder, "catalog")
    assert isinstance(finder.catalog, pl.DataFrame)
    assert finder.catalog.is_empty()


def test_worker_function(simple_image_data):
    """Test the internal _worker function directly."""
    # Simulate a source island cutout
    island_image = simple_image_data[30:70, 30:70]
    position = (30, 30)
    background_rms = 1.0  # For simplicity
    background = 5.0

    iterable = (island_image, position, background, background_rms)

    result_cat = _worker(
        iterable,
        analysis_threshold=3.0,
        lifetime_limit=0.1,
        lifetime_limit_fraction=1.0,
    )

    assert isinstance(result_cat, pl.DataFrame)
    assert not result_cat.is_empty()
    assert "Island_X" in result_cat.columns
    assert "Island_Y" in result_cat.columns
    assert result_cat["Island_X"][0] == position[0]
    assert result_cat["Island_Y"][0] == position[1]
