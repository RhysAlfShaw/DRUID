import polars as pl
import numpy as np


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


def generate_2d_gaussian(A, shape, center, sigma_x, sigma_y, angle_deg=0, norm=True):
    """

    Generate a 2D elliptical Gaussian distribution on a 2D array.

    Parameters:

        shape (tuple): Shape of the output array (height, width).
        center (tuple): Center of the Gaussian distribution (x, y).
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        angle_deg (float): Rotation angle in degrees (default is 0).

    Returns:

        ndarray: 2D array containing the Gaussian distribution.

    """
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x_c, y_c = center
    angle_rad = np.radians(angle_deg)

    # Rotate coordinates

    x_rot = (x - x_c) * np.cos(angle_rad) - (y - y_c) * np.sin(angle_rad)
    y_rot = (x - x_c) * np.sin(angle_rad) + (y - y_c) * np.cos(angle_rad)

    # Calculate Gaussian values

    gaussian = A * np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))

    if norm:
        return gaussian / (2 * np.pi * sigma_x * sigma_y)
    else:
        return gaussian


def model_beam_func(peak_flux, shape, x, y, bmaj, bmin, bpa):
    model_beam = np.zeros(shape)
    model_beam = generate_2d_gaussian(
        peak_flux, shape, (x, y), bmaj, bmin, bpa, norm=False
    )
    return model_beam
