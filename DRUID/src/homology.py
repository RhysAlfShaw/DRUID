import cripser
import numpy as np
import polars as pl
from scipy.ndimage import label as scipy_label
from tqdm import tqdm


def classify_single(row):
    """Classifiying the Rows based on orgin.

    Args:
        row: pd.series - The row that is being classified.

    Returns:
        Class: int - the Class integer that indiceates the class the row belongs too.

    """
    print(row)
    if row["new_row"] == 0:
        if len(row["encloses"]) == 0:  # no children
            if np.isnan(row["parent_tag"]):  # no parent
                return 0  # no children, no parent.
            else:
                return 1  # no child has parent.
        else:
            if np.isnan(row["parent_tag"]):
                return 2  # has children, no parent.
            else:
                return 3  # has children, has parent.
    else:
        return 4  # new row has children.


def correct_first_destruction_pl(df: pl.DataFrame) -> pl.DataFrame:
    """
    Function for correcting for the First destruction of a parent Island, adapted for Polars.

    This function identifies rows with "enclosed" islands, creates a new row for each,
    and inherits properties from the first enclosed island.

    Args:
        df (pl.DataFrame): Input catalogue of sources to correct.

    Returns:
        pl.DataFrame: The new catalogue with added rows.
    """
    # Ensure the 'new_row' column exists, initializing to 0
    if "new_row" not in df.columns:
        df = df.with_columns(pl.lit(0, dtype=pl.Int8).alias("new_row"))

    # 1. Filter the DataFrame to find all rows that have enclosed islands.
    islands_to_split = df.filter(pl.col("encloses").list.len() > 0)

    # If no such rows exist, return the original DataFrame.
    if islands_to_split.is_empty():
        return df

    # 2. Perform a self-join to fetch the 'Death' attribute from the parent island.
    # The parent is identified by the first ID in the 'enclosed_i' list.
    new_rows_base = islands_to_split.join(
        # Select only the necessary columns from the right side of the join
        df.select(["ID", "death"]),
        # Join condition: first element of 'enclosed_i' matches 'ID'
        left_on=pl.col("encloses").list.get(0),
        right_on="ID",
        how="inner",
        # Suffix prevents column name collisions ('Death' becomes 'Death_parent')
        suffix="_parent",
    )

    # If the join results in an empty DataFrame, return the original.
    if new_rows_base.is_empty():
        return df

    # 3. Generate a range of new, unique IDs for the rows to be added.
    # This correctly assigns a different ID to each new row.
    max_id = df["ID"].max()
    num_new_rows = len(new_rows_base)
    id_dtype = df.schema["ID"]  # Match the original ID data type
    new_ids = pl.int_range(
        start=max_id + 1,
        end=max_id + num_new_rows + 1,
        dtype=id_dtype,
        eager=True,  # Generate the series of new IDs immediately
    )

    # 4. Construct the new rows with updated and new values.
    new_rows = (
        new_rows_base.with_columns(
            # Overwrite the original ID with the new unique ID
            ID=new_ids,
            # Update 'Death' with the value from the joined parent
            Death=pl.col("death_parent"),
            # Set 'parent_tag' to the ID of the parent island
            parent_tag=pl.col("ID_parent"),
            # Mark this as a newly generated row
            new_row=pl.lit(1, dtype=pl.Int8),
            # Set 'enclosed_i' to an empty list
            enclosed_i=pl.lit(None, dtype=df.schema["encloses"]),
        )
        # Remove temporary columns created by the join
        .drop(["ID_parent", "death_parent"])
        # Ensure the column order matches the original DataFrame
        .select(df.columns)
    )

    # 5. Concatenate the original DataFrame with the newly created rows.
    return pl.concat([df, new_rows], how="vertical")


def parent_tag_func_pl(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sets the 'parent_tag' for each row based on 'enclosed_i' lists.

    This function identifies parent-child relationships where a parent's
    'enclosed_i' list contains child IDs. It then creates a 'parent_tag'
    column where each child's tag is set to its parent's ID. If an ID is
    not a child, its 'parent_tag' is set to its own ID.

    Args:
        df: The input Polars DataFrame. Must contain 'ID' and 'enclosed_i'
            (list of IDs) columns.

    Returns:
        The DataFrame with an added 'parent_tag' column.
    """
    # 1. Filter to get only the rows that are parents (i.e., they enclose other islands).
    # We also select only the necessary columns for creating the mapping.
    parents = df.filter(pl.col("encloses").list.len() > 1).select(
        pl.col("ID").alias("parent_id"), pl.col("encloses")
    )

    # 2. Create the parent-child mapping.
    # We "explode" the 'enclosed_i' list so that each child ID gets its own row
    # next to its parent's ID. This is the Polars way to create a lookup table.
    mapping = (
        parents.explode("encloses")
        .rename({"encloses": "child_id"})
        .filter(pl.col("child_id") != pl.col("parent_id"))  # Exclude self-references
    )

    # 3. Join the original DataFrame with the mapping.
    # This will add a 'parent_id' column to our DataFrame, but it will only
    # have values for rows that are children. Other rows will have null.
    df_with_parent_info = df.join(
        mapping, left_on="ID", right_on="child_id", how="left"
    )

    # 4. Create the final 'parent_tag' column.
    # We use coalesce() to fill in the missing values. It takes the first
    # non-null value it finds. So, if 'parent_id' exists, we use it;
    # otherwise, we fall back to the row's own 'ID'.
    df_final = df_with_parent_info.with_columns(
        parent_tag=pl.when(pl.col("parent_id").is_not_null())
        .then(pl.col("parent_id"))
        .otherwise(pl.col("ID"))
    ).drop(
        "parent_id"
    )  # Clean up the temporary column
    return df_final


def make_point_enclosure_assoc_CPU(x1, y1, Birth, Death, pd, img):
    """Returns a list of the ID of the points that are enclosed by the mask pd point.
        Uses GPU for computation.

    Args:
        Birth (float): _description_
        Death (float): _description_
        row (pd.Series): _description_
        pd (pl.DataFrame): polars DataFrame with point data.
        img (np.ndarray): _description_
        img_gpu (cp.ndarray): _description_

    Returns:
        enclosed_list (list): _description_
    """
    mask = get_mask_CPU(x1, y1, Birth, Death, img)
    mask_coords = np.column_stack((pd["x1"], pd["y1"]))

    points_inside_mask = mask[
        mask_coords[:, 0].astype(int), mask_coords[:, 1].astype(int)
    ]
    encloses_vectorized = pd.filter(points_inside_mask)["ID"].to_list()
    return encloses_vectorized


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

    import time

    img = components[2]
    t0_compute_ph = time.time()
    pd = cripser.computePH(-img, maxdim=0)
    t1_compute_ph = time.time()

    print(
        f"Computed persistent homology in {t1_compute_ph - t0_compute_ph:.2f} seconds."
    )

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
    liftetime_limit_fraction = 1.0  # set the lifetime limit fraction
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
    component_img = components[2]

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
            areas.append(0)
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

    plt.figure(figsize=(10, 10))
    plt.imshow(component_img, cmap="gray", origin="lower")
    plt.title("Component Image with Bounding Boxes")

    # remove those with area < 5 pixels
    polar_df = polar_df.filter(polar_df["area"] > 2)

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

    plt.colorbar()
    plt.show()
    # assign an ID to each point in the polar_df
    polar_df = polar_df.with_columns(pl.Series("ID", range(len(polar_df))))

    polar_df = polar_df.with_columns(
        pl.Series(
            "encloses",
            [
                make_point_enclosure_assoc_CPU(
                    row["x1"],
                    row["y1"],
                    row["birth"],
                    row["death"],
                    polar_df,
                    component_img,
                )
                for row in polar_df.iter_rows(named=True)
            ],
        )
    )

    print("Enclosure associations computed.")
    print(polar_df)
    print(len(polar_df))
    # correct first destruction
    polar_df = correct_first_destruction_pl(polar_df)
    print("First destruction corrected.")
    print(polar_df)
    print(len(polar_df))

    # assign parent tags
    polar_df = parent_tag_func_pl(polar_df)
    print("Parent tags assigned.")
    print(polar_df)
    print(len(polar_df))

    # calculate contours

    # # Classify the components by iterating over each row and applying the classify_single function
    # polar_df = polar_df.with_columns(
    #     pl.col(
    #         "Class", [classify_single(row) for row in polar_df.iter_rows(named=True)]
    #     )  # Apply classification function
    # )
    # print("Components classified.")
    # print(polar_df)
    # print(len(polar_df))
