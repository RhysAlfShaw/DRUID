"""
Author: Rhys Shaw
Date: 01-07-2025
"""

import numpy as np
from skimage.measure import regionprops
from skimage.measure import label
from tqdm import tqdm


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
