"""
Author: Rhys Shaw
Date: 11-06-2025
"""

import cripser
import numpy as np
import polars as pl
from scipy.ndimage import label as scipy_label
from skimage import measure


def get_enclosing_mask_CPU(x, y, mask):
    """
    Returns the connected components inside the mask starting from the point (x, y).
    """
    labeled_mask, _ = scipy_label(mask)
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        label_at_pixel = labeled_mask[y, x]
        if label_at_pixel != 0:
            return labeled_mask == label_at_pixel
    return None


def _get_polygons_CPU(x1, y1, birth, death, image: np.ndarray):
    """
    Returns the polygon of the enclosed area of the point (x,y) in the mask.
    """
    image_padded = np.pad(image, pad_width=1, mode="constant", constant_values=0)
    mask = (image_padded <= birth) & (image_padded > death)
    enclosed_mask = get_enclosing_mask_CPU(int(y1) + 1, int(x1) + 1, mask)

    if enclosed_mask is None:
        return [0]

    contours = measure.find_contours(enclosed_mask, 0)
    if not contours:
        return [0]

    contour = contours[0]
    contour[:, 0] -= 1
    contour[:, 1] -= 1
    return contour


def get_mask_CPU(x1, y1, Birth, Death, img):
    mask = (img <= Birth) & (img > Death)
    return get_enclosing_mask_CPU(int(y1), int(x1), mask)


def make_point_enclosure_assoc_CPU(x1, y1, Birth, Death, x1_arr, y1_arr, ids, img):
    mask = get_mask_CPU(x1, y1, Birth, Death, img)
    if mask is None:
        return []

    # FIX: cripser returns x1 as the row (axis 0) and y1 as the col (axis 1).
    # Therefore we index the numpy array using mask[x1, y1].
    valid_coords = mask[x1_arr.astype(int), y1_arr.astype(int)]
    return ids[valid_coords].tolist()


def correct_first_destruction_pl(df: pl.DataFrame) -> pl.DataFrame:
    if "new_row" not in df.columns:
        df = df.with_columns(pl.lit(0, dtype=pl.Int8).alias("new_row"))

    islands_to_split = df.filter(pl.col("encloses").list.len() > 1)
    if islands_to_split.is_empty():
        return df

    new_rows_base = islands_to_split.join(
        df.select(["ID", "death"]),
        left_on=pl.col("encloses").list.get(0),
        right_on="ID",
        how="inner",
        suffix="_parent",
    )

    if new_rows_base.is_empty():
        return df

    max_id = df["ID"].max()
    num_new_rows = len(new_rows_base)
    new_ids = pl.int_range(
        start=max_id + 1,
        end=max_id + num_new_rows + 1,
        dtype=df.schema["ID"],
        eager=True,
    )

    new_rows = (
        new_rows_base.with_columns(
            ID=new_ids,
            death=pl.col("death_parent"),
            parent_tag=pl.col("ID_parent"),
            new_row=pl.lit(1, dtype=pl.Int8),
            encloses=pl.lit(None, dtype=df.schema["encloses"]),
        )
        .drop(["ID_parent", "death_parent"])
        .select(df.columns)
    )

    return pl.concat([df, new_rows], how="vertical")


def parent_tag_func_pl(df: pl.DataFrame) -> pl.DataFrame:
    parents = df.filter(pl.col("encloses").list.len() > 1).select(
        pl.col("ID").alias("parent_id"), pl.col("encloses")
    )

    mapping = (
        parents.explode("encloses")
        .rename({"encloses": "child_id"})
        .filter(pl.col("child_id") != pl.col("parent_id"))
    )

    df_with_parent_info = df.join(
        mapping, left_on="ID", right_on="child_id", how="left"
    )

    return df_with_parent_info.with_columns(
        parent_tag=pl.when(pl.col("parent_id").is_not_null())
        .then(pl.col("parent_id"))
        .otherwise(pl.col("ID"))
    ).drop("parent_id")


def compute_homology(
    img: np.ndarray,
    analysis_threshold: float,
    lifetime_limit: float = 0.0,
    lifetime_limit_fraction: float = 1.0,
    area_size_threshold: int = 2,
) -> pl.DataFrame:

    pd_data = cripser.computePH(-img, maxdim=0)
    columns = ["dim", "birth", "death", "x1", "y1", "z1", "x2", "y2", "z2"]
    polar_df = pl.DataFrame(pd_data, schema=columns).drop(["dim", "z1", "z2"])

    polar_df = polar_df.with_columns(
        [(-pl.col("birth")).alias("birth"), (-pl.col("death")).alias("death")]
    )

    polar_df = polar_df.with_columns(
        pl.when(pl.col("death") < analysis_threshold)
        .then(pl.lit(analysis_threshold))
        .otherwise(pl.col("death"))
        .alias("death")
    )

    polar_df = polar_df.with_columns(
        (abs(pl.col("death") - pl.col("birth"))).alias("lifetime"),
        (pl.col("birth") / pl.col("death")).alias("lifetimeFrac"),
    )

    polar_df = polar_df.filter(
        (pl.col("lifetimeFrac") > lifetime_limit_fraction)
        & (pl.col("lifetime") > lifetime_limit)
    )

    if polar_df.is_empty():
        return None

    polar_df = polar_df.with_columns(
        pl.when(pl.col("lifetime") == pl.col("lifetime").max())
        .then(pl.lit(0))
        .otherwise(pl.col("death"))
        .alias("death")
    )

    # Fast NumPy extraction to avoid iter_rows bottleneck
    births = polar_df["birth"].to_numpy()
    deaths = polar_df["death"].to_numpy()
    x1s = polar_df["x1"].to_numpy()
    y1s = polar_df["y1"].to_numpy()

    areas, min_ys, min_xs, max_ys, max_xs = [], [], [], [], []

    for b, d, x, y in zip(births, deaths, x1s, y1s):
        mask = get_mask_CPU(x, y, b, d, img)
        if mask is not None:
            rows, cols = np.where(mask)
            areas.append(mask.sum())
            min_ys.append(rows.min())
            min_xs.append(cols.min())
            max_ys.append(rows.max())
            max_xs.append(cols.max())
        else:
            areas.append(0)
            min_ys.append(np.nan)
            min_xs.append(np.nan)
            max_ys.append(np.nan)
            max_xs.append(np.nan)

    polar_df = polar_df.with_columns(
        [
            pl.Series("area", areas),
            pl.Series("bbox_min_y", min_ys),
            pl.Series("bbox_min_x", min_xs),
            pl.Series("bbox_max_y", max_ys),
            pl.Series("bbox_max_x", max_xs),
        ]
    )

    polar_df = polar_df.filter(pl.col("area") > area_size_threshold)
    if polar_df.is_empty():
        return None

    polar_df = polar_df.with_columns(pl.Series("ID", range(len(polar_df))))

    # ---> FIX: Re-extract arrays from the FILTERED DataFrame <---
    # This ensures ids, x1s, and y1s all have the exact same length
    ids = polar_df["ID"].to_numpy()
    filtered_x1s = polar_df["x1"].to_numpy()
    filtered_y1s = polar_df["y1"].to_numpy()

    encloses = [
        make_point_enclosure_assoc_CPU(x, y, b, d, filtered_x1s, filtered_y1s, ids, img)
        for b, d, x, y in zip(
            polar_df["birth"], polar_df["death"], polar_df["x1"], polar_df["y1"]
        )
    ]
    polar_df = polar_df.with_columns(pl.Series("encloses", encloses))

    polar_df = correct_first_destruction_pl(polar_df)
    polar_df = parent_tag_func_pl(polar_df)

    contours = [
        (
            list(map(tuple, _get_polygons_CPU(x, y, b, d, img)))
            if isinstance(_get_polygons_CPU(x, y, b, d, img), np.ndarray)
            else [0]
        )
        for b, d, x, y in zip(
            polar_df["birth"], polar_df["death"], polar_df["x1"], polar_df["y1"]
        )
    ]

    return polar_df.with_columns(pl.Series("contour", contours))
