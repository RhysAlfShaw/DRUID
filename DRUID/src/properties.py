"""
Author: Rhys Shaw
Date: 08-09-2025
"""

from skimage import measure
import numpy as np
from skimage.draw import polygon
import polars as pl


def get_region_properties(mask, image):
    labeled_mask = measure.label(mask)

    properties = measure.regionprops(labeled_mask, intensity_image=image)

    return properties


def calculate_properties(
    cat, image, background, background_rms, position, analysis_threshold
):

    # create a mask of source based on birth and death
    mask = np.zeros_like(image, dtype=bool)
    mask = np.logical_or(mask, image > (analysis_threshold * background_rms))

    # bbox_data in cat to reduce to min size.?
    # Calculate using

    source_properties = get_region_properties(mask, image)

    # cat_with_properties = cat.with_columns(
    #     [
    #         pl.Series("area", [prop.area for prop in source_properties]),
    #         pl.Series(
    #             "centroid_row",
    #             [prop.centroid[0] + position[0] for prop in source_properties],
    #         ),
    #         pl.Series(
    #             "centroid_col",
    #             [prop.centroid[1] + position[1] for prop in source_properties],
    #         ),
    #     ]
    # )
    return cat


if __name__ == "__main__":

    # open dummy image parquet
    dummy_image = np.load("DRUID/temp/image_3C401.npy")
    dummy_background = np.load("DRUID/temp/background_3C401.npy")
    dummy_background_rms = np.load("DRUID/temp/background_rms_3C401.npy")

    import source
    import homology

    source_islands = source.create_source_islands(
        dummy_image, dummy_background, dummy_background_rms, 5, 3, 15, False
    )

    images_to_process = source_islands["island_image"]
    print(len(images_to_process))
    iterable_images = zip(
        images_to_process,
        source_islands["positions"],
        source_islands["background"],
        source_islands["background_rms"],
    )
    # i = 0
    for img, pos, back, back_rms in iterable_images:
        cat = homology.compute_homology(
            img,
            analysis_threshold=3 * back_rms,
            lifetime_limit=0.0,
            lifetime_limit_fraction=1.4,
        )
        cat_with_props = calculate_properties(
            cat,
            img,
            back,
            back_rms,
            pos,
            analysis_threshold=3,
        )
        print(cat_with_props)
        # i += 1
