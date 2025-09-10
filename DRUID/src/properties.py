"""
Author: Rhys Shaw
Date: 08-09-2025
"""

from skimage import measure
import numpy as np
from skimage.draw import polygon
import polars as pl
import homology


def calculate_radio_flux_error(background_rms, area, BMAJ, BMIN):
    # adapted from https://github.com/mhardcastle/radioflux/blob/master/radioflux/radioflux.py

    gfactor = 2 * np.sqrt(2 * np.log(2))
    Beam_area = 2 * np.pi * (BMAJ * BMIN) / gfactor
    return np.mean(background_rms) * np.sqrt(area / Beam_area)


def get_region_properties(mask, image):
    # labeled_mask = measure.label(mask)
    properties = measure.regionprops(mask, intensity_image=image)
    return properties


def get_row_mask(row, image):
    mask = np.zeros_like(image, dtype=bool)
    mask = np.logical_or(mask, np.logical_and(img <= row["birth"], img > row["death"]))
    mask = homology.get_enclosing_mask_CPU(int(row["y1"]), int(row["x1"]), mask)
    mask = mask.astype(int)
    return mask


def ABmag(flux):
    return -2.5 * np.log10(flux)


def RONoise(EFFRON, EFFGAIN, EXPTIME, Area):
    return np.sqrt(Area) * (EFFRON / EFFGAIN) * EXPTIME


def SkyNoise(sky):
    return np.sqrt(sky)


def SourceNoise(Flux):
    return np.sqrt(Flux)


def optical_flux_err(EFFRON, EFFGAIN, EXPTIME, Area, sky, Flux):
    try:
        RON_noise = RONoise(EFFRON, EFFGAIN, EXPTIME, Area)
    except:
        print(
            "Error calculating RONoise (likely missing EFFORN, EFFGAIN or EXPTIME in header), setting to 0"
        )
        RON_noise = 0
    Sky_noise = SkyNoise(sky)
    Source_noise = SourceNoise(Flux)
    return np.sqrt(RON_noise**2 + Sky_noise + Source_noise)


def NOISE(row, local_ng):
    return np.sum(np.random.normal(row["mean_bg"], local_ng, int(row["Area"])))


def calculate_properties(
    cat,
    image,
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
    from matplotlib import pyplot as plt

    plt.imshow(image, cmap="gray", origin="lower")
    plt.show()
    print(cat)
    maj = []
    min = []
    pa = []
    centroid = []
    flux = []
    flux_peak = []
    bg = []
    flux_err = []
    snr = []
    for row in cat.iter_rows(named=True):
        # print(row)
        # create a mask of source based on birth and death
        mask = get_row_mask(row, image)
        props = get_region_properties(mask, image)

        maj.append(props[0].major_axis_length)
        min.append(props[0].minor_axis_length)
        pa.append(props[0].orientation)
        centroid.append(props[0].centroid)

        # calculate fluxes
        flux_tot = np.nansum(mask * (image - background))
        flux.append(flux_tot)
        flux_peak.append(np.nanmax(mask * (image - background)))
        bg.append(np.mean(background * mask))

        if mode == "radio":

            Flux_total_err = calculate_radio_flux_error(
                background_rms, row["area"], BMAJ, BMIN
            )
            flux_err.append(Flux_total_err)

        elif mode == "optical":
            Flux_total_err = optical_flux_err(
                EFFRON=EFFRON,
                EFFGAIN=EFFGAIN,
                EXPTIME=EXPTIME,
                Area=row["area"],
                sky=np.nansum(background * mask),
                Flux=np.nansum(mask * (image - background)),
            )
            flux_err.append(Flux_total_err)

        snr.append(flux_tot / Flux_total_err)

    cat = cat.with_columns(
        pl.Series("maj", maj),
        pl.Series("min", min),
        pl.Series("pa", pa),
        pl.Series("centroid", centroid),
        pl.Series("flux_peak", flux_peak),
        pl.Series("bg", bg),
        pl.Series("flux_err", flux_err),
        pl.Series("flux", flux),
        pl.Series("snr", snr),
    )

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
    iterable_images = zip(
        images_to_process,
        source_islands["positions"],
        source_islands["background"],
        source_islands["background_rms"],
    )
    i = 0

    for img, pos, back, back_rms in iterable_images:
        if i > 0:
            break
        cat = homology.compute_homology(
            img,
            analysis_threshold=3 * back_rms,
            lifetime_limit=0.0,
            lifetime_limit_fraction=1.4,
        )
        BMAJ = 0.35  # arcsec
        BMIN = 0.35  # arcsec
        cat_with_props = calculate_properties(
            cat,
            img,
            back,
            back_rms,
            pos,
            analysis_threshold=3,
            mode="radio",
            BMAJ=BMAJ,
            BMIN=BMIN,
        )
        # print(cat_with_props)

        i += 1
