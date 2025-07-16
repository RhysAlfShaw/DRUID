import polars as pl


def get_image_from_path(image_path):
    """
    Load an image from a file path.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        The loaded image as a NumPy array.
    """
    from astropy.io import fits

    with fits.open(image_path) as hdul:
        image = hdul[0].data

    # warn if the image is not 2D
    # reduce the image to 2D if it is not
    if image.ndim == 3:
        image = image[0, :, :]
    elif image.ndim == 4:
        image = image[0, 0, :, :]
    return image


def combine_polars_catalogs(catalogs: list):
    """
    Combine multiple polar catalogs into a single catalog.

    """
    if not catalogs:
        raise ValueError("No catalogs provided to combine.")

    combined_catalog = pl.concat(catalogs)

    # Ensure the 'id' column is unique
    if "id" in combined_catalog.columns:
        combined_catalog = combined_catalog.with_columns(
            pl.col("id").cast(pl.Int64)
        ).with_columns(pl.col("id").rank(method="dense").alias("id"))

    return combined_catalog
