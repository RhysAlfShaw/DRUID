"""
Author: Rhys Shaw
Date: 01-07-2025
"""

import numpy as np
from skimage.measure import regionprops
from skimage.measure import label
from tqdm import tqdm

import numpy as np
from skimage.measure import regionprops, label, regionprops_table
from tqdm import tqdm
import pandas as pd  #
import polars as pl


def create_source_islands(
    image,
    background_map,
    background_rms_map,
    detection_threshold=5,
    analysis_threshold=3,
    area_limit=2,
    verbose=True,
):
    """
    Create source islands from the background and RMS maps. And create cutouts of them.

    Parameters
    ----------
    background_map : numpy.ndarray
        The background map of the image.
    background_rms_map : numpy.ndarray
        The RMS map of the image.
    detection_threshold : float, optional
        Threshold for detecting sources, by default 5.
    analysis_threshold : float, optional
        Threshold for analyzing sources, by default 3.

    Returns
    -------
        a dictionary of source islands, with keys, array (the cropped image),
        poistion (the position of the source in the original image),
    """

    thresholded_image = np.where(
        image > background_map + analysis_threshold * background_rms_map, image, 0
    )
    import time

    t0 = time.time()
    labeled_image = label(thresholded_image > 0, connectivity=2)
    t1 = time.time()
    if verbose:
        print(
            f"Labeling connected components took {t1 - t0:.2f} seconds. Found {np.unique(labeled_image).size - 1} components."
        )
    t0 = time.time()
    properties = regionprops(labeled_image, intensity_image=thresholded_image)
    t1 = time.time()
    if verbose:
        print(
            f"Calculating region properties took {t1 - t0:.2f} seconds. Found {len(properties)} properties."
        )
    components = []
    source_islands_positions = []
    source_island_bg_rms = []
    source_island_bg = []
    min_area = area_limit
    area = []
    for prop in properties:
        if prop.area < min_area:
            continue
        area.append(prop.area)
        # prop.intensity_image is the cropped and masked component
        components.append(np.array(prop.intensity_image))

        # prop.bbox returns (min_row, min_col, max_row, max_col)
        y_min, x_min, _, _ = prop.bbox
        source_islands_positions.append((y_min, x_min))
        # Get the background and RMS values for the component
        source_island_bg.append(
            background_map[y_min : prop.bbox[2], x_min : prop.bbox[3]].mean()
        )
        source_island_bg_rms.append(
            background_rms_map[y_min : prop.bbox[2], x_min : prop.bbox[3]].mean()
        )
    import matplotlib.pyplot as plt

    plt.hist(area)
    plt.yscale("log")
    plt.savefig("area_distribution.png")

    t1 = time.time()
    if verbose:
        print(
            f"Cropping components took {t1 - t0:.2f} seconds. Found {len(components)} source islands."
        )

    source_islands = {
        "island_image": components,
        "positions": source_islands_positions,
        "background": source_island_bg,
        "background_rms": source_island_bg_rms,
    }

    shuffled_islands = shuffle_in_unison(
        [
            source_islands["island_image"],
            source_islands["positions"],
            source_islands["background"],
            source_islands["background_rms"],
        ]
    )
    return shuffled_islands


def shuffle_in_unison(arrays):
    """Shuffle multiple arrays in unison, preserving the correspondence between them."""
    assert all(
        len(arr) == len(arrays[0]) for arr in arrays
    ), "All arrays must have the same length."
    p = np.random.permutation(len(arrays[0]))
    return {
        key: [array[i] for i in p]
        for key, array in zip(
            ["island_image", "positions", "background", "background_rms"], arrays
        )
    }


def create_source_islands_optimized(
    image,
    background_map,
    background_rms_map,
    detection_threshold=5,  # Not used in current logic, but kept for signature
    analysis_threshold=3,
    area_limit=2,
    verbose=True,
):
    """
    Create source islands from the background and RMS maps. And create cutouts of them.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    background_map : numpy.ndarray
        The background map of the image.
    background_rms_map : numpy.ndarray
        The RMS map of the image.
    detection_threshold : float, optional
        Threshold for detecting sources (currently not used for analysis logic), by default 5.
    analysis_threshold : float, optional
        Threshold for analyzing sources, by default 3.
    area_limit : int, optional
        Minimum area (in pixels) for a detected region to be considered a source island, by default 2.
    verbose : bool, optional
        If True, display progress bars, by default True.

    Returns
    -------
        a dictionary of source islands, with keys, array (the cropped image),
        poistion (the position of the source in the original image),
    """

    if verbose:
        print(
            "Step 1: Applying analysis threshold and labeling connected components..."
        )

    # Create a boolean mask directly. This avoids creating a full-size float array of zeros.
    analysis_mask = image > (background_map + analysis_threshold * background_rms_map)

    # Label connected components on the boolean mask
    # connectivity=2 is 8-connectivity for 2D images
    labeled_image = label(analysis_mask, connectivity=2)

    # Use regionprops_table for efficiency, requesting only necessary properties
    # 'bbox' for cropping, 'label' for filtering, 'area' for filtering
    # 'image' would give the cropped binary mask, 'intensity_image' would give cropped intensities.
    # We will slice the original image/thresholded data later for actual intensities.
    if verbose:
        print("Step 2: Measuring region properties...")

    # We only need 'bbox' and 'area' for filtering and cropping
    # If you need other properties for analysis later, add them here.
    properties_table = regionprops_table(
        labeled_image,
        intensity_image=image,  # Pass the original image for intensity measurements
        properties=("label", "bbox", "area"),
    )

    # Convert to DataFrame for easier filtering
    props_df = pd.DataFrame(properties_table)

    if verbose:
        print(f"Initial regions found: {len(props_df)}")
        print(f"Step 3: Filtering regions by area (>{area_limit} pixels)...")

    # Filter out components smaller than area_limit pixels
    # Filtering on the DataFrame is much faster than iterating a list of RegionProperties objects.
    filtered_props_df = props_df[props_df["area"] >= area_limit]

    if verbose:
        print(f"Regions after area filtering: {len(filtered_props_df)}")
        print("Step 4: Extracting source island cutouts...")

    components = []
    source_islands_positions = []

    # Iterate through the filtered DataFrame rows
    # Using itertuples() is generally faster than iterrows() for DataFrames
    for row in tqdm(
        filtered_props_df.itertuples(),
        total=len(filtered_props_df),
        disable=not verbose,
    ):
        min_row, min_col, max_row, max_col = row.bbox

        # Slice the *original* image directly to get the intensities within the bounding box
        # This is more efficient than recreating a masked array for each component.
        # Ensure max_row and max_col are exclusive in python slicing, so bbox_coords[2] and bbox_coords[3] work directly
        component_image_cutout = image[
            min_row:max_row, min_col:max_col
        ].copy()  # .copy() to ensure it's a separate array

        # To get the thresholded values only within the cutout (if needed):
        # component_thresholded_cutout = thresholded_image[min_row:max_row, min_col:max_col]
        # Or even better, apply the threshold condition directly to the cutout:
        component_analysis_cutout = component_image_cutout * (
            component_image_cutout
            > (
                background_map[min_row:max_row, min_col:max_col]
                + analysis_threshold
                * background_rms_map[min_row:max_row, min_col:max_col]
            )
        )

        position = (min_row, min_col)
        source_islands_positions.append(position)
        components.append(component_analysis_cutout)  # Store the thresholded cutout

    source_islands = {
        "island_image": components,
        "positions": source_islands_positions,
    }

    if verbose:
        print("Source island creation complete.")

    return source_islands
