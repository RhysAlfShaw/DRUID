import numpy as np
from skimage.measure import regionprops_table, label
import polars as pl

def create_source_islands(
    image,
    background_map,
    background_rms_map,
    detection_threshold=5,
    analysis_threshold=3,
    area_limit=2,
    max_area_limit=10000,
    verbose=True,
):
    """
    Create source islands using optimized vectorization.
    Returns bounding boxes instead of full arrays to save IPC overhead.
    Separates massive islands for cataloging without computing homology.
    """
    if verbose:
        print("Step 1: Applying analysis threshold and labeling connected components...")

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

    # Initialize Polars DataFrame directly from the dictionary of arrays
    props_df = pl.DataFrame(properties_table)

    if verbose:
        print(f"Initial regions found: {props_df.height}")
        print(f"Step 3: Filtering regions by area ({area_limit} <= area <= {max_area_limit} pixels)...")

    # ---> FAST POLARS FILTERING <---
    # Standard processing queue
    filtered_props_df = props_df.filter(
        (pl.col("area") >= area_limit) & 
        (pl.col("area") <= max_area_limit)
    )
    
    # Flagged massive islands
    massive_props_df = props_df.filter(pl.col("area") > max_area_limit)

    if verbose:
        if massive_props_df.height > 0:
            print(f"Flagged {massive_props_df.height} massive region(s) to retain for the final catalog.")
        print(f"Regions queued for homology processing: {filtered_props_df.height}")

    # Extract standard metadata (for the multiprocessing pool)
    bboxes = list(
        zip(
            filtered_props_df["bbox-0"].to_numpy(),
            filtered_props_df["bbox-1"].to_numpy(),
            filtered_props_df["bbox-2"].to_numpy(),
            filtered_props_df["bbox-3"].to_numpy(),
        )
    )

    positions = list(
        zip(
            filtered_props_df["bbox-0"].to_numpy(),
            filtered_props_df["bbox-1"].to_numpy(),
        )
    )

    # Extract massive metadata (to bypass pool but append to catalog)
    massive_bboxes = list(
        zip(
            massive_props_df["bbox-0"].to_numpy(),
            massive_props_df["bbox-1"].to_numpy(),
            massive_props_df["bbox-2"].to_numpy(),
            massive_props_df["bbox-3"].to_numpy(),
        )
    )

    massive_positions = list(
        zip(
            massive_props_df["bbox-0"].to_numpy(),
            massive_props_df["bbox-1"].to_numpy(),
        )
    )

    source_islands = {
        "bboxes": bboxes,
        "positions": positions,
        "massive_bboxes": massive_bboxes,       # <-- New: Saved massive bounding boxes
        "massive_positions": massive_positions, # <-- New: Saved massive coordinates
    }

    if verbose:
        print("Source island creation complete.")

    return source_islands