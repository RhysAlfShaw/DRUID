version = "1.0"

import setproctitle

setproctitle.setproctitle("DRUID")

import numpy as np
import astropy.io.fits
from astropy.table import Table
import os
import ast
import sys
import time
import polars as pl
import polars.selectors as cs
from functools import partial
from multiprocessing import get_context
from multiprocessing import shared_memory
from rich.progress import Progress
import multiprocessing
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import logging

from .src import utils
from .src.utils import (
    TITLE,
    LINK,
    GOLD,
    RESET,
    BOLD,
    NOTICE,
    ERROR,
    WARNING,
    CODEBLOCK,
    GREEN,
    BLACK,
)
from .src import homology
from .src import background
from .src import source
from .src import properties

# Prevent Polars from thread oversubscription during multiprocessing
os.environ["POLARS_MAX_THREADS"] = "1"


DRUID_MESSAGE = rf"""  
{TITLE}
     _____  _____  _    _ _____ _____  
     |  __ \|  __ \| |  | |_   _|  __ \ 
     | |  | | |__) | |  | | | | | |  | |
     | |  | |  _  /| |  | | | | | |  | |
     | |__| | | \ \| |__| |_| |_| |__| |
     |_____/|_|  \_\\____/|_____|_____/
        
{RESET}

{BOLD}Detector of astRonomical soUrces in optIcal and raDio images{RESET}

{GOLD}Version{RESET}: {version}
For more information see:
{LINK}https://github.com/RhysAlfShaw/DRUID{RESET}
"""

global_image = None
global_smoothed_image = None
global_background_map = None
global_background_rms_map = None

# Keep shared memory objects alive in the worker
shm_img = None
shm_smooth = None
shm_bg = None
shm_rms = None


def _worker_init(
    shm_img_name,
    img_shape,
    img_dtype,
    shm_smooth_name,
    smooth_shape,
    smooth_dtype,
    shm_bg_name,
    bg_shape,
    bg_dtype,
    shm_rms_name,
    rms_shape,
    rms_dtype,
):
    """
    Initializer for multiprocessing pool.
    Attaches to shared memory blocks created by the main process.
    """
    global global_image, global_smoothed_image, global_background_map, global_background_rms_map
    global shm_img, shm_smooth, shm_bg, shm_rms

    from multiprocessing import shared_memory
    import numpy as np

    shm_img = shared_memory.SharedMemory(name=shm_img_name)
    global_image = np.ndarray(shape=img_shape, dtype=img_dtype, buffer=shm_img.buf)

    shm_smooth = shared_memory.SharedMemory(name=shm_smooth_name)
    global_smoothed_image = np.ndarray(
        shape=smooth_shape, dtype=smooth_dtype, buffer=shm_smooth.buf
    )

    shm_bg = shared_memory.SharedMemory(name=shm_bg_name)
    global_background_map = np.ndarray(
        shape=bg_shape, dtype=bg_dtype, buffer=shm_bg.buf
    )

    shm_rms = shared_memory.SharedMemory(name=shm_rms_name)
    global_background_rms_map = np.ndarray(
        shape=rms_shape, dtype=rms_dtype, buffer=shm_rms.buf
    )


def _worker(
    island_info,
    analysis_threshold,
    lifetime_limit,
    lifetime_limit_fraction,
    mode=None,
    BMAJ=None,
    BMIN=None,
    EFFRON=None,
    EFFGAIN=None,
    EXPTIME=None,
) -> pl.DataFrame:
    """
    Worker function to compute homology for a single source island.
    """
    bbox, position = island_info
    min_row, min_col, max_row, max_col = bbox

    # Cutouts extraction
    raw_image_cutout = global_image[min_row:max_row, min_col:max_col]
    smoothed_image_cutout = global_smoothed_image[min_row:max_row, min_col:max_col]
    bg_cutout = global_background_map[min_row:max_row, min_col:max_col]
    bg_rms_cutout = global_background_rms_map[min_row:max_row, min_col:max_col]

    local_threshold = bg_cutout + (analysis_threshold * bg_rms_cutout)

    # Island mask derived strictly from the smoothed cutout
    island_mask = smoothed_image_cutout > local_threshold
    smoothed_cutout_masked = np.where(island_mask, smoothed_image_cutout, 0)

    # Topology computed on smoothed data
    cat = homology.compute_homology(
        smoothed_cutout_masked,
        analysis_threshold=analysis_threshold * np.mean(bg_rms_cutout),
        lifetime_limit=lifetime_limit,
        lifetime_limit_fraction=lifetime_limit_fraction,
    )

    if cat is not None and not cat.is_empty():
        # Properties utilize BOTH raw and smoothed arrays
        cat = properties.calculate_properties(
            cat,
            raw_image=raw_image_cutout,
            smoothed_image=smoothed_cutout_masked,
            background=bg_cutout,
            background_rms=bg_rms_cutout,
            position=position,
            analysis_threshold=analysis_threshold,
            mode=mode,
            BMAJ=BMAJ,
            BMIN=BMIN,
            EFFRON=EFFRON,
            EFFGAIN=EFFGAIN,
            EXPTIME=EXPTIME,
        )

        cat = cat.with_columns(
            [
                pl.lit(position[0]).alias("Island_Y"),
                pl.lit(position[1]).alias("Island_X"),
            ]
        )

    return cat


class sf:
    def __init__(
        self,
        image: str | np.ndarray = None,
        mode: str = None,
        verbose: bool = True,
        area_limit: int = 0,
        max_area_limit: int = 10000,
        smooth_sigma: float = 0,
        num_threads: int = 1,
        chunksize: int = 10,
        header: astropy.io.fits.header.Header = None,
        working_directory: str = "./druid-working-dir",
        cache: bool = False,
        output_arg: str = "",
        no_message: bool = False,
    ):
        error_msg = f"""
            {ERROR}===================================================================={RESET}
            {BOLD}DRUID MULTIPROCESSING {ERROR}ERROR!{RESET}

            It looks like you are running DRUID with `num_threads > 1` without 
            protecting your execution code. 

            Because DRUID uses Python's robust multiprocessing, you must wrap your 
            top-level code in the `if __name__ == '__main__':` block.

            {BOLD}Please update your script to look like this:{RESET}

            {BLACK}from DRUID import sf

            def main():
                findmysource = sf(num_threads={num_threads}, ...)
                findmysource.set_background(...)
                findmysource.phsf(...)

            if __name__ == "__main__":
                main()
            {ERROR}===================================================================={RESET}
        """

        if multiprocessing.current_process().name != "MainProcess":
            raise RuntimeError(error_msg)
        if num_threads > 1 and multiprocessing.current_process().name == "MainProcess":
            try:
                import __main__

                if not hasattr(__main__, "__file__") or not os.path.exists(
                    __main__.__file__
                ):
                    raise RuntimeError(
                        f"{error_msg} (Cannot verify script safety in interactive environments)"
                    )

                with open(__main__.__file__, "r", encoding="utf-8") as f:
                    source_code = f.read()

                tree = ast.parse(source_code)

                is_protected = False
                for node in tree.body:
                    if isinstance(node, ast.If):
                        if isinstance(node.test, ast.Compare):
                            left = node.test.left
                            if isinstance(left, ast.Name) and left.id == "__name__":
                                is_protected = True
                                break

                if not is_protected:
                    raise RuntimeError(error_msg)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to validate safe multiprocessing execution: {e}"
                )

        self.no_message = no_message
        if not self.no_message:
            print(DRUID_MESSAGE)

        self.mode = mode
        self.verbose = verbose
        self.area_limit = area_limit
        self.max_area_limit = max_area_limit
        self.smooth_sigma = smooth_sigma
        self.num_threads = num_threads
        self.chunksize = chunksize
        self.header = header
        self.cache = cache
        self.output_arg = output_arg
        self.smoothed_image = None

        if image is None:
            raise ValueError(
                f"{ERROR}No image provided. Please provide a file path or a NumPy array.{RESET}"
            )

        if isinstance(image, str):
            try:
                self.image, self.header = utils.get_image_from_path(image)
            except Exception as e:
                raise ValueError(
                    f"{ERROR}Could not load image from path{RESET}: {image}"
                ) from e
        elif isinstance(image, np.ndarray):
            self.image = image
            self.header = header
        else:
            raise TypeError(
                f"{ERROR}Image must be a file path (str) or a NumPy array (np.ndarray).{RESET}"
            )
        self.working_directory = working_directory
        if self.working_directory:
            if not os.path.exists(working_directory):
                os.makedirs(working_directory)

        self.BMAJ, self.BMIN = None, None
        self.EFFRON, self.EFFGAIN, self.EXPTIME = None, None, None

        if self.mode == "radio" and self.header:
            try:
                self.BMAJ = self.header.get("BMAJ")
                self.BMIN = self.header.get("BMIN")
            except KeyError:
                print(
                    f"{WARNING}Warning: Could not find BMAJ or BMIN in header.{RESET}"
                )
        elif self.mode == "optical" and self.header:
            try:
                self.EFFRON = self.header.get("EFFRON")
                self.EFFGAIN = self.header.get("EFFGAIN")
                self.EXPTIME = self.header.get("EXPTIME")
            except KeyError:
                print(
                    f"{WARNING}Warning: Could not find EFFRON, EFFGAIN, or EXPTIME.{RESET}"
                )

    def phsf(self, lifetime_limit: float = 0.0, lifetime_limit_fraction: float = 1.0):
        if (
            getattr(self, "background_map", None) is None
            or getattr(self, "background_rms_map", None) is None
        ):
            raise ValueError(
                f"{ERROR}Background maps must be set before running source finding.{RESET}"
            )

        t0 = time.time()

        # Apply structural smoothing before thresholding
        if self.smooth_sigma > 0:
            if self.verbose:
                print(
                    f"{NOTICE}Applying Gaussian smoothing with sigma={self.smooth_sigma}...{RESET}"
                )
            self.smoothed_image = gaussian_filter(self.image, sigma=self.smooth_sigma)
        else:
            self.smoothed_image = self.image

        if self.verbose:
            print(f"{NOTICE}Thresholding to find source islands...{RESET}")

        t0 = time.time()
        source_islands = source.create_source_islands(
            self.smoothed_image,
            self.background_map,
            self.background_rms_map,
            detection_threshold=self.detection_threshold,
            analysis_threshold=self.analysis_threshold,
            area_limit=self.area_limit,
            max_area_limit=self.max_area_limit,
            verbose=self.verbose,
        )
        t1 = time.time()

        if self.verbose:
            print(f"{NOTICE}Thresholding took {t1 - t0:.2f} seconds.{RESET}")
            print(
                f"{NOTICE}Found {len(source_islands['bboxes'])} source islands.{RESET}"
            )

        iterable_islands = list(
            zip(source_islands["bboxes"], source_islands["positions"])
        )
        iterable_islands.sort(
            key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
            reverse=True,
        )

        if not iterable_islands:
            if self.verbose:
                print(
                    "{WARNING}Warning: No source islands found. Returning empty catalog.{RESET}"
                )
            self.catalog = pl.DataFrame()
            return

        t0 = time.time()

        worker_func = partial(
            _worker,
            analysis_threshold=self.analysis_threshold,
            lifetime_limit=lifetime_limit,
            lifetime_limit_fraction=lifetime_limit_fraction,
            mode=self.mode,
            BMAJ=self.BMAJ,
            BMIN=self.BMIN,
            EFFRON=self.EFFRON,
            EFFGAIN=self.EFFGAIN,
            EXPTIME=self.EXPTIME,
        )

        results = []
        if self.num_threads > 1:
            if self.verbose:
                print(
                    f"{NOTICE}Processing in parallel with {self.num_threads} threads.{RESET}"
                )
            optimal_chunksize = self.chunksize

            # Shared memory allocations
            shm_img = shared_memory.SharedMemory(create=True, size=self.image.nbytes)
            shm_smooth = shared_memory.SharedMemory(
                create=True, size=self.smoothed_image.nbytes
            )
            shm_bg = shared_memory.SharedMemory(
                create=True, size=self.background_map.nbytes
            )
            shm_rms = shared_memory.SharedMemory(
                create=True, size=self.background_rms_map.nbytes
            )

            np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=shm_img.buf)[
                :
            ] = self.image[:]
            np.ndarray(
                self.smoothed_image.shape,
                dtype=self.smoothed_image.dtype,
                buffer=shm_smooth.buf,
            )[:] = self.smoothed_image[:]
            np.ndarray(
                self.background_map.shape,
                dtype=self.background_map.dtype,
                buffer=shm_bg.buf,
            )[:] = self.background_map[:]
            np.ndarray(
                self.background_rms_map.shape,
                dtype=self.background_rms_map.dtype,
                buffer=shm_rms.buf,
            )[:] = self.background_rms_map[:]

            with get_context("spawn").Pool(
                self.num_threads,
                initializer=_worker_init,
                initargs=(
                    shm_img.name,
                    self.image.shape,
                    self.image.dtype,
                    shm_smooth.name,
                    self.smoothed_image.shape,
                    self.smoothed_image.dtype,
                    shm_bg.name,
                    self.background_map.shape,
                    self.background_map.dtype,
                    shm_rms.name,
                    self.background_rms_map.shape,
                    self.background_rms_map.dtype,
                ),
            ) as p:
                with Progress(disable=not self.verbose) as progress:
                    task = progress.add_task(
                        "[magenta]:mage: Computing...", total=len(iterable_islands)
                    )

                    results = []
                    for result in p.imap_unordered(
                        worker_func, iterable_islands, chunksize=optimal_chunksize
                    ):
                        results.append(result)
                        progress.advance(task)  # Update the progress bar incrementally

            # Flush memory
            shm_img.close()
            shm_img.unlink()
            shm_smooth.close()
            shm_smooth.unlink()
            shm_bg.close()
            shm_bg.unlink()
            shm_rms.close()
            shm_rms.unlink()
        else:
            global global_image, global_smoothed_image, global_background_map, global_background_rms_map
            global_image = self.image
            global_smoothed_image = self.smoothed_image
            global_background_map = self.background_map
            global_background_rms_map = self.background_rms_map

            # Single-threaded Rich progress bar implementation
            with Progress(disable=not self.verbose) as progress:
                task = progress.add_task(
                    f"[magenta]:mage: Computing...", total=len(iterable_islands)
                )

                for island in iterable_islands:
                    results.append(worker_func(island))
                    progress.advance(task)

        results = [res for res in results if res is not None and not res.is_empty()]
        if results:
            self.catalog = utils.combine_polars_catalogs(results)
            # calculate the ra and dec columns if the header is available
            # add island offsets to the centroid and contour coordinates
            self.catalog = self.catalog.with_columns(
                [
                    (pl.col("centroid_x") + pl.col("Island_X")).alias("centroid_x"),
                    (pl.col("centroid_y") + pl.col("Island_Y")).alias("centroid_y"),
                ]
            )
            # add island offsets to the contour coordinates
            self.catalog = (
                self.catalog.with_row_index("__row_id")
                .explode("contour")
                .with_columns(
                    pl.concat_list(
                        [
                            pl.col("contour").list.get(0) + pl.col("Island_X"),
                            pl.col("contour").list.get(1) + pl.col("Island_Y"),
                        ]
                    ).alias("contour")
                )
                .group_by("__row_id", maintain_order=True)
                .agg(
                    pl.all().exclude("contour").first(),
                    pl.col("contour"),
                )
                .drop("__row_id")
            )
            if self.header is not None:
                self.catalog = utils.calculate_radec(self.catalog, self.header)

            else:
                print(
                    f"{WARNING}Warning{RESET}: No FITS header provided. RA and Dec columns will not be calculated."
                )
                self.catalog = self.catalog.with_columns(
                    pl.lit(None).alias("ra"), pl.lit(None).alias("dec")
                )
            desired_order = [
                "ID",
                "ra",
                "dec",
                "centroid_x",
                "centroid_y",
                "flux",
                "flux_peak",
                "flux_err",
                "bg",
                "snr",
                "maj",
                "min",
                "pa",
                "area",
                "contour",
                "lifetime",
                "birth",
                "death",
                "x1",
                "y1",
                "x2",
                "y2",
                "encloses",
                "new_row",
                "parent_tag",
                "class",
                "lifetimeFrac",
                "bbox_min_y",
                "bbox_min_x",
                "bbox_max_y",
                "bbox_max_x",
                "Island_X",
                "Island_Y",
            ]
            self.catalog = self.catalog.select(
                *desired_order, cs.all().exclude(desired_order)
            )

            # save catalog to working directory
            catalog_file = os.path.join(
                self.working_directory,
                f"druid_source_catalog_{self.output_arg}",
            )
            self.catalog.write_parquet(f"{catalog_file}.parquet")

            print(f"{NOTICE}Catalog saved to {catalog_file}.parquet{RESET}")
        else:
            self.catalog = pl.DataFrame()

        t1 = time.time()
        if self.verbose:
            print(f"{NOTICE}Homology computation took {t1 - t0:.2f} seconds.{RESET}")
            print(f"{GREEN}---------------CATALOG SUMMARY---------------------{RESET}")
            print(f"Total sources detected: {self.catalog.height}")
            print(
                f"Number of large sources (area > {self.max_area_limit}): {self.catalog.filter(pl.col('area') > self.max_area_limit).height}"
            )
            print("Average Background: ", self.background_map.mean())
            print("Average Background RMS: ", self.background_rms_map.mean())
            print(f"{GREEN}---------------------------------------------------{RESET}")

    def set_background(
        self,
        method: str = "rms",
        detection_threshold: int = 5,
        analysis_threshold: int = 3,
        box_size: tuple = (50, 50),
        filter_size: tuple = (3, 3),
        kernel_size: int = 3,
    ):
        if self.verbose:
            print(f"{NOTICE}Calculating background map and RMS map...{RESET}")
        t0 = time.time()
        self.detection_threshold = detection_threshold
        self.analysis_threshold = analysis_threshold

        bg_file = os.path.join(self.working_directory or "", "background_map.npy")
        rms_file = os.path.join(self.working_directory or "", "background_rms_map.npy")

        if self.cache and os.path.exists(bg_file) and os.path.exists(rms_file):
            if self.verbose:
                print(f"{NOTICE}Background maps exist. Loading from disk.{RESET}")
            self.background_map = np.load(bg_file)
            self.background_rms_map = np.load(rms_file)
        else:
            self.background_map, self.background_rms_map = (
                background.calculate_background_maps(
                    self.image,
                    bg_estimator=method,
                    box_size=box_size,
                    filter_size=filter_size,
                    nsigma=detection_threshold,
                    kernel_size=kernel_size,
                )
            )
            if self.cache:
                np.save(bg_file, self.background_map)
                np.save(rms_file, self.background_rms_map)

        t1 = time.time()
        if self.verbose:
            print(f"{NOTICE}Background calculation took {t1 - t0:.2f} seconds.{RESET}")
