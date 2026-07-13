"""
Author: Rhys Shaw
Date: 01-07-2025
"""

import numpy as np
from skimage.measure import regionprops_table, label
from tqdm import tqdm
import pandas as pd


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
    Create source islands using optimized vectorization.
    Returns bounding boxes instead of full arrays to save IPC overhead.
    """
    if verbose:
        print(
            "Step 1: Applying analysis threshold and labeling connected components..."
        )

    # Vectorized boolean mask creation
    analysis_mask = image > (background_map + analysis_threshold * background_rms_map)
    labeled_image = label(analysis_mask, connectivity=2)

    if verbose:
        print("Step 2: Measuring region properties...")

    # Use regionprops_table for C-level fast property extraction
    properties_table = regionprops_table(
        labeled_image,
        properties=("label", "bbox", "area"),
    )

    props_df = pd.DataFrame(properties_table)

    if verbose:
        print(f"Initial regions found: {len(props_df)}")
        print(f"Step 3: Filtering regions by area (>={area_limit} pixels)...")

    # Fast pandas filtering
    filtered_props_df = props_df[props_df["area"] >= area_limit]

    if verbose:
        print(f"Regions after area filtering: {len(filtered_props_df)}")

    # Extract metadata arrays
    # bbox columns from regionprops_table are bbox-0, bbox-1, bbox-2, bbox-3
    bboxes = list(
        zip(
            filtered_props_df["bbox-0"],
            filtered_props_df["bbox-1"],
            filtered_props_df["bbox-2"],
            filtered_props_df["bbox-3"],
        )
    )

    positions = list(zip(filtered_props_df["bbox-0"], filtered_props_df["bbox-1"]))

    source_islands = {
        "bboxes": bboxes,
        "positions": positions,
    }

    if verbose:
        print("Source island creation complete.")

    return source_islands
