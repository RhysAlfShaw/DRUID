"""
Author: Rhys Shaw
Date: 08-09-2025
"""

import numpy as np
import polars as pl
from skimage import measure
from scipy.ndimage import label as scipy_label


def get_enclosing_mask_CPU(x, y, mask):
    labeled_mask, _ = scipy_label(mask)
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        label_at_pixel = labeled_mask[y, x]
        if label_at_pixel != 0:
            return labeled_mask == label_at_pixel
    return None


def calculate_radio_flux_error(background_rms, area, BMAJ, BMIN):
    if BMAJ is None or BMIN is None:
        return np.nan

    gfactor = 2 * np.sqrt(2 * np.log(2))
    Beam_area = 2 * np.pi * (BMAJ * BMIN) / gfactor
    return np.mean(background_rms) * np.sqrt(area / Beam_area)


def optical_flux_err(EFFRON, EFFGAIN, EXPTIME, Area, sky, Flux):
    try:
        RON_noise = np.sqrt(Area) * (EFFRON / EFFGAIN) * EXPTIME
    except Exception:
        RON_noise = 0
    return np.sqrt(RON_noise**2 + np.sqrt(sky) + np.sqrt(Flux))


def calculate_properties(
    cat,
    raw_image,
    smoothed_image,
    background,
    background_rms,
    position,
    analysis_threshold,
    mode,
    BMAJ=None,
    BMIN=None,
    EFFRON=None,
    EFFGAIN=None,
    EXPTIME=None,
):
    # Vectorized extraction to avoid Polars iter_rows
    births = cat["birth"].to_numpy()
    deaths = (
        cat["deaths"].to_numpy() if "deaths" in cat.columns else cat["death"].to_numpy()
    )
    x1s = cat["x1"].to_numpy()
    y1s = cat["y1"].to_numpy()
    areas = cat["area"].to_numpy()

    maj, min_ax, pa, centroid_lst, flux, flux_peak, bg, flux_err, snr = (
        [], [], [], [], [], [], [], [], []
    )

    for b, d, x, y, area in zip(births, deaths, x1s, y1s, areas):
        # mask boundaries determined by the SMOOTHED image
        mask = (smoothed_image <= b) & (smoothed_image > d)
        enclosed_mask = get_enclosing_mask_CPU(int(y), int(x), mask)

        if enclosed_mask is None:
            # Fallback for empty/invalid properties
            maj.append(np.nan)
            min_ax.append(np.nan)
            pa.append(np.nan)
            centroid_lst.append((np.nan, np.nan))
            flux.append(np.nan)
            flux_peak.append(np.nan)
            bg.append(np.nan)
            flux_err.append(np.nan)
            snr.append(np.nan)
            continue

        enclosed_mask_int = enclosed_mask.astype(int)
        
        # Geometries and intensities extracted from the RAW image
        props = measure.regionprops(enclosed_mask_int, intensity_image=raw_image)

        if props:
            p = props[0]
            maj.append(p.major_axis_length)
            min_ax.append(p.minor_axis_length)
            pa.append(p.orientation)
            centroid_lst.append(p.centroid)
        else:
            maj.append(np.nan)
            min_ax.append(np.nan)
            pa.append(np.nan)
            centroid_lst.append((np.nan, np.nan))

        # Flux summations computed on RAW data
        flux_tot = np.nansum(enclosed_mask_int * (raw_image - background))
        flux.append(flux_tot)
        flux_peak.append(np.nanmax(enclosed_mask_int * (raw_image - background)))

        bg_mean = np.mean(background * enclosed_mask_int)
        bg.append(bg_mean)

        if mode == "radio":
            f_err = calculate_radio_flux_error(background_rms, area, BMAJ, BMIN)
            flux_err.append(f_err)
            if f_err and not np.isnan(f_err):
                snr.append(flux_tot / f_err)
            else:
                snr.append(np.nan)

        elif mode == "optical":
            f_err = optical_flux_err(
                EFFRON,
                EFFGAIN,
                EXPTIME,
                area,
                np.nansum(background * enclosed_mask_int),
                flux_tot,
            )
            flux_err.append(f_err)
            snr.append(flux_tot / f_err if f_err else 0)
        else:
            flux_err.append(0)
            snr.append(0)

    return cat.with_columns(
        [
            pl.Series("maj", maj),
            pl.Series("min", min_ax),
            pl.Series("pa", pa),
            pl.Series("centroid_x", [c[1] for c in centroid_lst]),
            pl.Series("centroid_y", [c[0] for c in centroid_lst]),
            pl.Series("flux_peak", flux_peak),
            pl.Series("bg", bg),
            pl.Series("flux_err", flux_err),
            pl.Series("flux", flux),
            pl.Series("snr", snr),
        ]
    )