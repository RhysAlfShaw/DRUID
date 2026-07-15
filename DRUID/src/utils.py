import polars as pl
import numpy as np
from astropy.io import fits


def get_image_from_path(image_path):
    with fits.open(image_path) as hdul:
        image = hdul[0].data
        header = hdul[0].header

    if image.ndim == 3:
        image = image[0, :, :]
    elif image.ndim == 4:
        image = image[0, 0, :, :]
    return image, header


def combine_polars_catalogs(catalogs: list) -> pl.DataFrame:
    if not catalogs:
        raise ValueError("No catalogs provided to combine.")

    combined_catalog = pl.concat(catalogs)
    
    # Check for 'id' or 'ID' depending on your upstream schema
    if "id" in combined_catalog.columns:
        combined_catalog = combined_catalog.with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("id")
        )
    elif "ID" in combined_catalog.columns:
        combined_catalog = combined_catalog.with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("ID")
        )

    return combined_catalog

def generate_2d_gaussian(A, shape, center, sigma_x, sigma_y, angle_deg=0, norm=True):
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x_c, y_c = center
    angle_rad = np.radians(angle_deg)

    x_rot = (x - x_c) * np.cos(angle_rad) - (y - y_c) * np.sin(angle_rad)
    y_rot = (x - x_c) * np.sin(angle_rad) + (y - y_c) * np.cos(angle_rad)

    gaussian = A * np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))

    if norm:
        return gaussian / (2 * np.pi * sigma_x * sigma_y)
    return gaussian


def model_beam_func(peak_flux, shape, x, y, bmaj, bmin, bpa):
    return generate_2d_gaussian(peak_flux, shape, (x, y), bmaj, bmin, bpa, norm=False)
