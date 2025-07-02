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

    labeled_image = label(thresholded_image > 0, connectivity=2)
    properties = regionprops(labeled_image, intensity_image=thresholded_image)

    # filter out components smaller than 5 pixels
    min_area = area_limit
    filtered_labels = [prop.label for prop in properties if prop.area >= min_area]

    filtered_labeled_image = np.zeros_like(labeled_image)
    for label_value in filtered_labels:
        filtered_labeled_image[labeled_image == label_value] = label_value

    labeled_image = filtered_labeled_image

    # for each label crop around it.
    unique_labels = np.unique(labeled_image)
    components = []
    source_islands_positions = []
    for label_value in tqdm(unique_labels):
        if label_value == 0:
            continue  # Skip the background label
        component_mask = labeled_image == label_value
        component = np.where(component_mask, thresholded_image, 0)
        # crop around the component
        y_indices, x_indices = np.where(component_mask)

        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        component = component[y_min : y_max + 1, x_min : x_max + 1]
        position = (y_min, x_min)
        source_islands_positions.append(position)
        components.append(component)

    source_islands = {
        "island_image": components,
        "positions": source_islands_positions,
    }

    return source_islands


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
        # Bounding box is (min_row, min_col, max_row, max_col)
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
