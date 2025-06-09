import cripser
import numpy as np
import polars as pl
from scipy.ndimage import label as scipy_label
from tqdm import tqdm


def get_enclosing_mask_CPU(x, y, mask):
    """
    Returns the connected components inside the mask starting from the point (x, y).
    """
    from skimage.measure import label

    labeled_mask, num_features = scipy_label(mask)

    # Check if the specified pixel is within the mask
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        label_at_pixel = labeled_mask[y, x]

        if label_at_pixel != 0:
            # Extract the connected component containing the specified pixel
            component_mask = labeled_mask == label_at_pixel
            return component_mask
        else:
            # The specified pixel is not part of any connected component
            return None
    else:
        # The specified pixel is outside the mask
        return None


def bounding_box_cpu(mask):
    # Get the indices of elements that are True
    rows, cols = np.where(mask)
    # Get the minimum and maximum x and y coordinates
    min_y, max_y = np.min(rows), np.max(rows)
    min_x, max_x = np.min(cols), np.max(cols)
    # Return the bounding box as a tuple of tuples
    return min_y, min_x, max_y, max_x


def get_mask_CPU(x1, y1, Birth, Death, img):
    """Get mask for a single row uses the CPU

    Args:
        row (pd.Series): row we want to get the mask for.
        img (np.ndarray): image that the source is in.

    Returns:
        mask_enclosed(np.ndarray): Array of the mask

    """

    mask = np.zeros(img.shape)
    mask = np.logical_or(mask, np.logical_and(img <= Birth, img > Death))
    mask_enclosed = get_enclosing_mask_CPU(int(y1), int(x1), mask)

    return mask_enclosed


def compute_ph_components(
    threholded_image,
    local_bg,
    analysis_threshold_val,
):
    print("Computing persistent homology components...")


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from astropy.io import fits

    print("DRUID - Homology.py test script")
    # get example tresholded image.
    dummy_data_path = "DRUID/temp/dummy_image.fits"

    background_map_path = "DRUID/temp/background_map.fits"
    background_rms_map_path = "DRUID/temp/background_rms_map.fits"

    # load the images
    dummy_data = fits.open(dummy_data_path)[0].data
    background_map = fits.open(background_map_path)[0].data
    background_rms_map = fits.open(background_rms_map_path)[0].data

    # set anything in the mask to 0
    thresholded_image = np.where(
        dummy_data > background_map + 10 * background_rms_map, dummy_data, 0
    )

    # sort by area and remove components smaller than 5 pixels

    from skimage.measure import regionprops
    from skimage.measure import label

    labeled_image = label(thresholded_image > 0, connectivity=2)
    properties = regionprops(labeled_image, intensity_image=thresholded_image)

    # filter out components smaller than 5 pixels
    min_area = 5
    filtered_labels = [prop.label for prop in properties if prop.area >= min_area]

    # create a new labeled image with only the filtered labels
    filtered_labeled_image = np.zeros_like(labeled_image)
    for label_value in filtered_labels:
        filtered_labeled_image[labeled_image == label_value] = label_value

    # use the filtered labeled image for further processing
    labeled_image = filtered_labeled_image

    # for each label crop around it.
    unique_labels = np.unique(labeled_image)
    components = []
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
        components.append(component)

    print(f"Found {len(components)} components.")
    # plt.figure(figsize=(8, 6))
    # plt.imshow(
    #     components[0], origin="lower", cmap="nipy_spectral", interpolation="nearest"
    # )
    # plt.title("Labeled Image")
    # plt.colorbar()
    # plt.show()
    import time

    img = components[1]
    t0_compute_ph = time.time()
    pd = cripser.computePH(-img, maxdim=0)
    t1_compute_ph = time.time()
    print(
        f"Computed persistent homology in {t1_compute_ph - t0_compute_ph:.2f} seconds."
    )
    # create polar dataframe to handle the data

    columns = ["dim", "birth", "death", "x1", "y1", "z1", "x2", "y2", "z2"]
    polar_df = pl.DataFrame(pd, schema=columns)
    # drop cols dim, z1, z2
    polar_df = polar_df.drop(["dim", "z1", "z2"])
    # create ne column lifetime death - birth
    polar_df = polar_df.with_columns(
        (polar_df["death"] - polar_df["birth"]).alias("lifetime")
    )
    # make column birth and death - birth and death.
    polar_df = polar_df.with_columns(
        [(-polar_df["birth"]).alias("birth"), (-polar_df["death"]).alias("death")]
    )

    # lifetime_threshold. this is setby the user.

    polar_df = polar_df.with_columns(
        (polar_df["birth"] - polar_df["death"]).alias("lifetimeFrac")
    )
    print(len(polar_df))
    liftetime_limit_fraction = 3.0  # set the lifetime limit fraction
    # filter out components with lifetime less than 3
    polar_df = polar_df.filter(polar_df["lifetime"] > liftetime_limit_fraction)
    print(
        f"Filtered polar dataframe to {len(polar_df)} components with lifetime > {liftetime_limit_fraction}."
    )
    print(polar_df)

    # filter by pixel size. get bounding box of the component, and contour?.

    # for each of the components compute the area left between the birth and death.

    # compute the area of the component
    # for each of the components compute the area left between the birth and death.

    # compute the area of the component
    areas = []
    bbox_min_y_list = []
    bbox_min_x_list = []
    bbox_max_y_list = []
    bbox_max_x_list = []

    # Assuming 'components[0]' is the correct component for all rows in polar_df
    # If each row corresponds to a different component, you'll need to adjust this.
    # For now, let's stick to the logic in your snippet.
    component_img = components[0]

    for row_tuple in polar_df.iter_rows(
        named=True
    ):  # named=True gives you a dictionary per row
        # get mask of the component using birth and death values.
        birth = row_tuple["birth"]
        death = row_tuple["death"]
        x1 = row_tuple["x1"]
        y1 = row_tuple["y1"]

        mask = get_mask_CPU(
            x1,  # Note: Your get_mask_CPU expects x1, y1, Birth, Death, img
            y1,
            birth,
            death,
            component_img,  # Use the pre-selected component
        )

        if mask is not None:
            bounding_box = bounding_box_cpu(mask)
            area = np.sum(mask)

            areas.append(area)
            bbox_min_y_list.append(bounding_box[0])
            bbox_min_x_list.append(bounding_box[1])
            bbox_max_y_list.append(bounding_box[2])
            bbox_max_x_list.append(bounding_box[3])
        else:
            # Handle cases where mask is None (e.g., point outside, no component)
            # Append NaN or a placeholder, or filter these rows out later
            areas.append(np.nan)
            bbox_min_y_list.append(np.nan)
            bbox_min_x_list.append(np.nan)
            bbox_max_y_list.append(np.nan)
            bbox_max_x_list.append(np.nan)

    # Add the new columns to the DataFrame
    polar_df = polar_df.with_columns(
        [
            pl.Series("area", areas),
            pl.Series("bbox_min_y", bbox_min_y_list),
            pl.Series("bbox_min_x", bbox_min_x_list),
            pl.Series("bbox_max_y", bbox_max_y_list),
            pl.Series("bbox_max_x", bbox_max_x_list),
        ]
    )

    # remove those with area < 5 pixels
    polar_df = polar_df.filter(polar_df["area"] > 5)

    # plot the components with bounding boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(
        component_img, origin="lower", cmap="nipy_spectral", interpolation="nearest"
    )
    for row_tuple in polar_df.iter_rows(named=True):

        bbox_min_y = row_tuple["bbox_min_y"]
        bbox_min_x = row_tuple["bbox_min_x"]
        bbox_max_y = row_tuple["bbox_max_y"]
        bbox_max_x = row_tuple["bbox_max_x"]

        if not np.isnan(bbox_min_y) and not np.isnan(bbox_min_x):
            # Draw the bounding box
            plt.gca().add_patch(
                plt.Rectangle(
                    (bbox_min_x, bbox_min_y),
                    bbox_max_x - bbox_min_x,
                    bbox_max_y - bbox_min_y,
                    edgecolor="blue",
                    facecolor="none",
                    linewidth=2,
                )
            )

    plt.title("Component with Bounding Boxes")
    plt.colorbar()
    plt.show()
