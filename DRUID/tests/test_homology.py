import pytest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from DRUID.src.homology import (
    compute_homology,
    get_mask_CPU,
    get_enclosing_mask_CPU,
    bounding_box_cpu,
    parent_tag_func_pl,
    correct_first_destruction_pl,
)
from DRUID.src.background import make_gaussian_sources_image


@pytest.fixture
def simple_image():
    """Creates a simple 100x100 image with one Gaussian source."""
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
    return make_gaussian_sources_image(image_size, sources) + 0.1


@pytest.fixture
def nested_source_image():
    """Creates an image with two nested Gaussian sources."""
    image_size = (100, 100)
    sources = [
        {
            "amplitude": 100,
            "x_mean": 50,
            "y_mean": 50,
            "x_stddev": 10,
            "y_stddev": 10,
            "theta": 0,
        },
        {
            "amplitude": 50,
            "x_mean": 50,
            "y_mean": 50,
            "x_stddev": 3,
            "y_stddev": 3,
            "theta": 0,
        },
    ]
    return make_gaussian_sources_image(image_size, sources) + 0.1


def test_compute_homology_simple_source(simple_image):
    """Test compute_homology on an image with a single, simple source."""
    result_df = compute_homology(
        simple_image, analysis_threshold=1.0, lifetime_limit=0.1
    )

    assert isinstance(result_df, pl.DataFrame)
    assert not result_df.is_empty()
    assert result_df["birth"].max() == pytest.approx(100, abs=1)
    expected_cols = {
        "birth",
        "death",
        "x1",
        "y1",
        "lifetime",
        "lifetimeFrac",
        "area",
        "bbox_min_y",
        "ID",
        "encloses",
        "parent_tag",
        "contour",
    }
    assert expected_cols.issubset(result_df.columns)


def test_get_mask_cpu():
    """Test the get_mask_CPU function."""
    img = np.array([[0, 0, 0, 0], [0, 5, 5, 0], [0, 5, 5, 0], [0, 0, 0, 0]])
    mask = get_mask_CPU(x1=1, y1=1, Birth=6, Death=4, img=img)
    expected_mask = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )
    assert np.array_equal(mask, expected_mask)


def test_get_enclosing_mask_cpu():
    """Test the get_enclosing_mask_CPU function."""
    mask = np.array(
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=bool
    )
    component_mask = get_enclosing_mask_CPU(x=1, y=1, mask=mask)
    expected = np.array(
        [
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    assert np.array_equal(component_mask, expected)

    component_mask_none = get_enclosing_mask_CPU(x=0, y=0, mask=mask)
    assert component_mask_none is None


def test_bounding_box_cpu():
    """Test the bounding_box_cpu function."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    bbox = bounding_box_cpu(mask)
    assert bbox == (2, 3, 4, 6)


def test_parent_tag_func_pl():
    """Test the parent_tag_func_pl function."""
    df = pl.DataFrame(
        {
            "ID": [0, 1, 2, 3],
            "encloses": [[1, 2], [], [], [0]],
        }
    )
    result = parent_tag_func_pl(df)
    expected = pl.DataFrame(
        {
            "ID": [0, 1, 2, 3],
            "encloses": [[1, 2], [], [], [0]],
            "parent_tag": [0, 0, 0, 3],
        }
    )
    assert_frame_equal(result, expected)


def test_correct_first_destruction_pl():
    """Test the correct_first_destruction_pl function."""
    df = pl.DataFrame(
        {
            "ID": [0, 1, 2],
            "death": [10.0, 5.0, 8.0],
            "encloses": [[1, 2], [], []],
            "parent_tag": [0, 0, 0],
        }
    )
    result = correct_first_destruction_pl(df)
    assert len(result) == 4
    new_row = result.filter(pl.col("ID") == 3)
    assert not new_row.is_empty()
    assert new_row["death"][0] == 5.0
    assert new_row["parent_tag"][0] == 1
    assert new_row["new_row"][0] == 1
