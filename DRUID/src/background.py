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
    mean, median, std = sigma_clipped_stats(data, sigma=nsigma)
    threshold = median + nsigma * std
    segm = detect_sources(data, threshold, n_pixels=kernel_size**2)
    if segm is None:
        return np.zeros(data.shape, dtype=bool)
    return segm.data > 0


def calculate_background_maps(
    image,
    bg_estimator="median",
    box_size=(50, 50),
    filter_size=(3, 3),
    nsigma=3.0,
    kernel_size=3,
):
    if isinstance(image, str):
        with fits.open(image) as hdul:
            data = hdul[0].data
    elif isinstance(image, np.ndarray):
        data = image
    else:
        raise TypeError("Image must be a path or a numpy array.")

    mask = make_source_mask(data, nsigma=nsigma, kernel_size=kernel_size)

    available_estimators = {
        "median": MedianBackground,
        "std": StdBackgroundRMS,
        "mad_std": MADStdBackgroundRMS,
        "rms": StdBackgroundRMS,
        "biweightlocation": BiweightLocationBackground,
        "biweightscale": BiweightScaleBackgroundRMS,
        "mm": MMMBackground,
        "mean": MeanBackground,
        "mode": ModeEstimatorBackground,
        "sex": SExtractorBackground,
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
