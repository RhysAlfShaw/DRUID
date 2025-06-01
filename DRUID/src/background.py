import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.background import (
    Background2D,
    MedianBackground,
    StdBackgroundRMS,
    MADStdBackgroundRMS,
    BiweightLocationBackground,
    BiweightScaleBackgroundRMS,
    MMMBackground,
    MeanBackground,
    ModeEstimatorBackground,
    SExtractorBackground,
    BackgroundBase,
)

from photutils.segmentation import detect_sources


def make_source_mask(data, nsigma=3.0, kernel_size=3):
    """
    Create a mask for sources in the image data using sigma clipping.

    Parameters
    ----------
    data : numpy.ndarray
        The 2D image data.
    nsigma : float, optional
        The number of standard deviations to use for sigma clipping.
        The default is 3.0.
    kernel_size : int, optional
        The size of the convolution kernel for source detection.
        The default is 3.

    Returns
    -------
    mask : numpy.ndarray
        A boolean mask where True indicates a source pixel.
    """
    mean, median, std = sigma_clipped_stats(data, sigma=nsigma)
    threshold = median + nsigma * std

    # Detect sources using a simple thresholding method can add masked pixels e.g. known bad areas of image.
    segm = detect_sources(data, threshold, npixels=kernel_size**2)

    # Create a mask from the segmentation map
    mask = segm.data > 0

    return mask


def calculate_background_maps(
    image_path,
    bg_estimator="median",
    box_size=(50, 50),
    filter_size=(3, 3),
    nsigma=3.0,
    kernel_size=3,
):
    """
    Calculates background and background RMS maps from a FITS image
    similar to the style of PyBDSF.

    Parameters
    ----------
    image_path : str
        Path to the FITS image file.
    box_size : tuple of int, optional
        The size of the box to use for background estimation.
        The default is (50, 50).
    filter_size : tuple of int, optional
        The size of the median filter to apply to the background map.
        The default is (3, 3).
    nsigma : float, optional
        The number of standard deviations to use for sigma clipping
        when detecting sources to mask. The default is 3.0.
    bg_estimator : str or photutils.background.BackgroundBase, optional
        The background estimator to use. Options are 'median', 'std', 'mad_std',
        or a custom photutils background estimator object. The default is 'median'.
    kernel_size : int, optional
        The size of the convolution kernel for source detection.
        The default is 3.

    Returns
    -------
    background_map : numpy.ndarray
        The calculated background map.
    background_rms_map : numpy.ndarray
        The calculated background RMS map.
    """
    with fits.open(image_path) as hdul:
        data = hdul[0].data

    # Mask sources
    mask = make_source_mask(data, nsigma=nsigma, kernel_size=kernel_size)

    # calculate background and RMS Avalible background estimators
    available_estimators = {
        "median": MedianBackground,
        "std": StdBackgroundRMS,
        "mad_std": MADStdBackgroundRMS,
        "rms": StdBackgroundRMS,
        "BiweightLocation": BiweightLocationBackground,
        "BiweightScale": BiweightScaleBackgroundRMS,
        "MM": MMMBackground,
        "Mean": MeanBackground,
        "mode": ModeEstimatorBackground,
        "SEx": SExtractorBackground,
    }

    if isinstance(bg_estimator, str):
        bkg_estimator = available_estimators.get(
            bg_estimator.lower(), MedianBackground
        )()
    elif isinstance(bg_estimator, BackgroundBase):
        bkg_estimator = bg_estimator
    else:
        bkg_estimator = MedianBackground()

    bkg = Background2D(
        data,
        box_size,
        filter_size=filter_size,
        mask=mask,
        bkg_estimator=bkg_estimator,
    )

    return bkg.background, bkg.background_rms


def make_gaussian_sources_image(image_size, sources):
    """
    Create a 2D image with Gaussian sources.
    Parameters
    ----------
    image_size : tuple of int
        Size of the image (height, width).
    sources : list of dict
        List of sources, each defined by a dictionary with keys:
        'amplitude', 'x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta'.
    Returns
    -------
    numpy.ndarray
        2D array representing the image with Gaussian sources.
    """
    image = np.zeros(image_size)
    y, x = np.indices(image_size)
    for source in sources:
        amplitude = source["amplitude"]
        x_mean = source["x_mean"]
        y_mean = source["y_mean"]
        x_stddev = source["x_stddev"]
        y_stddev = source["y_stddev"]
        theta = source["theta"]
        a = (np.cos(theta) ** 2) / (2 * x_stddev**2) + (np.sin(theta) ** 2) / (
            2 * y_stddev**2
        )
        b = -np.sin(2 * theta) / (4 * x_stddev**2) + np.sin(2 * theta) / (
            4 * y_stddev**2
        )
        c = (np.sin(theta) ** 2) / (2 * x_stddev**2) + (np.cos(theta) ** 2) / (
            2 * y_stddev**2
        )
        gaussian = amplitude * np.exp(
            -(
                a * (x - x_mean) ** 2
                + 2 * b * (x - x_mean) * (y - y_mean)
                + c * (y - y_mean) ** 2
            )
        )
        image += gaussian
    return image


if __name__ == "__main__":
    # Example usage:
    # Create a dummy FITS file for demonstration
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord

    # Define image parameters
    image_size = (1000, 1000)
    pixel_scale = 0.1  # degrees per pixel
    center_coord = SkyCoord(ra=180, dec=30, unit="deg")

    # Create a dummy WCS
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [image_size[0] / 2, image_size[1] / 2]
    wcs.wcs.cdelt = np.array([-pixel_scale, pixel_scale])
    wcs.wcs.crval = [center_coord.ra.deg, center_coord.dec.deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Create dummy sources
    sources = [
        {
            "amplitude": 100,
            "x_mean": 500,
            "y_mean": 500,
            "x_stddev": 50,
            "y_stddev": 50,
            "theta": 0,
        },
        {
            "amplitude": 150,
            "x_mean": 150,
            "y_mean": 150,
            "x_stddev": 7,
            "y_stddev": 7,
            "theta": np.pi / 4,
        },
        {
            "amplitude": 200,
            "x_mean": 300,
            "y_mean": 300,
            "x_stddev": 10,
            "y_stddev": 10,
            "theta": np.pi / 2,
        },
        {
            "amplitude": 80,
            "x_mean": 700,
            "y_mean": 800,
            "x_stddev": 6,
            "y_stddev": 6,
            "theta": np.pi / 3,
        },
        {
            "amplitude": 120,
            "x_mean": 900,
            "y_mean": 200,
            "x_stddev": 8,
            "y_stddev": 8,
            "theta": np.pi / 6,
        },
        {
            "amplitude": 90,
            "x_mean": 400,
            "y_mean": 600,
            "x_stddev": 4,
            "y_stddev": 4,
            "theta": np.pi / 8,
        },
    ]

    # Create a dummy image with sources and background noise
    dummy_data = make_gaussian_sources_image(image_size, sources)
    dummy_data += np.random.normal(0, 5, size=image_size)  # Add some noise

    # Create a dummy FITS file
    hdu = fits.PrimaryHDU(dummy_data, header=wcs.to_header())
    dummy_fits_path = "DRUID/temp/dummy_image.fits"
    hdu.writeto(dummy_fits_path, overwrite=True)

    print(f"Dummy FITS file created at: {dummy_fits_path}")

    # Calculate background maps
    background_map, background_rms_map = calculate_background_maps(dummy_fits_path)
    # save the background maps to a FITS file
    background_hdu = fits.PrimaryHDU(background_map)
    background_rms_hdu = fits.PrimaryHDU(background_rms_map)
    background_hdu.writeto("DRUID/temp/background_map.fits", overwrite=True)
    background_rms_hdu.writeto("DRUID/temp/background_rms_map.fits", overwrite=True)

    print("Background map and background RMS map saved to FITS files.")
    print("Background map calculated.")
    print("Background RMS map calculated.")

    # You can optionally save the background maps to
    # plot the results with matplotlib or any other visualization library.
    from matplotlib import pyplot as plt

    # plot dummy data, background map, and background RMS map
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(dummy_data, origin="lower", cmap="gray", interpolation="nearest")
    plt.title("Dummy Image with Sources")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(background_map, origin="lower", cmap="gray", interpolation="nearest")
    plt.title("Background Map")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(background_rms_map, origin="lower", cmap="gray", interpolation="nearest")
    plt.title("Background RMS Map")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
