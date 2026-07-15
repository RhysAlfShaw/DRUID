version = "1.0"

import setproctitle

setproctitle.setproctitle("DRUID")

import numpy as np
import astropy.io.fits
import os
import sys
import time
import polars as pl
from functools import partial
from multiprocessing import get_context
from multiprocessing import shared_memory
import multiprocessing
from tqdm import tqdm

from .src import utils
from .src import homology
from .src import background
from .src import source
from .src import properties

# Prevent Polars from thread oversubscription during multiprocessing
os.environ["POLARS_MAX_THREADS"] = "1"

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"
DRUID_MESSAGE = rf"""  
{RED}#############################################{RESET}
{GREEN}
_______   _______          _________ ______  
(  __  \ (  ____ )|\     /|\__   __/(  __  \ 
| (  \  )| (    )|| )   ( |   ) (   | (  \  )
| |   ) || (____)|| |   | |   | |   | |   ) |
| |   | ||     __)| |   | |   | |   | |   | |
| |   ) || (\ (   | |   | |   | |   | |   ) |
| (__/  )| ) \ \__| (___) |___) (___| (__/  )
(______/ |/   \__/(_______)\_______/(______/ 
        
{RESET}
{RED}#############################################{RESET}

{BOLD}Detector of astRonomical soUrces in optIcal and raDio images{RESET}

Version: {version}
For more information see:
{BLUE}https://github.com/RhysAlfShaw/DRUID{RESET}
"""

# Global variables for worker processes to avoid IPC memory overhead
global_image = None
global_background_map = None
global_background_rms_map = None

# Keep shared memory objects alive in the worker
shm_img = None
shm_bg = None
shm_rms = None

def _worker_init(
    shm_img_name, img_shape, img_dtype,
    shm_bg_name, bg_shape, bg_dtype,
    shm_rms_name, rms_shape, rms_dtype
):
    """
    Initializer for multiprocessing pool.
    Attaches to shared memory blocks created by the main process.
    """
    global global_image, global_background_map, global_background_rms_map
    global shm_img, shm_bg, shm_rms
    
    from multiprocessing import shared_memory
    import numpy as np
    
    # 1. Attach and map the main image
    shm_img = shared_memory.SharedMemory(name=shm_img_name)
    global_image = np.ndarray(shape=img_shape, dtype=img_dtype, buffer=shm_img.buf)
    
    # 2. Attach and map the background map
    shm_bg = shared_memory.SharedMemory(name=shm_bg_name)
    global_background_map = np.ndarray(shape=bg_shape, dtype=bg_dtype, buffer=shm_bg.buf)
    
    # 3. Attach and map the background RMS map
    shm_rms = shared_memory.SharedMemory(name=shm_rms_name)
    global_background_rms_map = np.ndarray(shape=rms_shape, dtype=rms_dtype, buffer=shm_rms.buf)

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
    Reads from global arrays to minimize memory serialization.
    """
    bbox, position = island_info
    min_row, min_col, max_row, max_col = bbox

    raw_image_cutout = global_image[min_row:max_row, min_col:max_col]
    bg_cutout = global_background_map[min_row:max_row, min_col:max_col]
    bg_rms_cutout = global_background_rms_map[min_row:max_row, min_col:max_col]

    local_threshold = bg_cutout + (analysis_threshold * bg_rms_cutout)
    island_mask = raw_image_cutout > local_threshold

    image_cutout = np.where(island_mask, raw_image_cutout, 0)

    cat = homology.compute_homology(
        image_cutout,
        analysis_threshold=analysis_threshold * np.mean(bg_rms_cutout),
        lifetime_limit=lifetime_limit,
        lifetime_limit_fraction=lifetime_limit_fraction,
    )

    if cat is not None and not cat.is_empty():
        cat = properties.calculate_properties(
            cat,
            image_cutout,
            bg_cutout,
            bg_rms_cutout,
            position,
            analysis_threshold,
            mode,
            BMAJ,
            BMIN,
            EFFRON,
            EFFGAIN,
            EXPTIME,
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
        cashe: bool = False,
        no_message: bool = False,
    ):
        error_msg = f"""
            {RED}===================================================================={RESET}
            {BOLD}DRUID MULTIPROCESSING ERROR{RESET}

            It looks like you are running DRUID with `num_threads > 1` without 
            protecting your execution code. 

            Because DRUID uses Python's robust multiprocessing, you must wrap your 
            top-level code in the `if __name__ == '__main__':` block.

            {BLUE}Please update your script to look like this:{RESET}

            from DRUID import sf

            def main():
                findmysource = sf(num_threads={num_threads}, ...)
                findmysource.set_background(...)
                findmysource.phsf(...)

            if __name__ == "__main__":
                main()
            {RED}===================================================================={RESET}
        """

        if multiprocessing.current_process().name != "MainProcess":
            raise RuntimeError(error_msg)

        if num_threads > 1 and multiprocessing.current_process().name == "MainProcess":
            try:
                import __main__

                if hasattr(__main__, "__file__") and os.path.exists(__main__.__file__):
                    with open(__main__.__file__, "r") as f:
                        script_content = f.read()

                    clean_script = script_content.replace(" ", "").replace("'", '"')

                    if 'if__name__=="__main__":' not in clean_script:
                        raise RuntimeError(error_msg)
            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise e

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
        self.cashe = cashe

        if image is None:
            raise ValueError(
                "No image provided. Please provide a file path or a NumPy array."
            )

        if isinstance(image, str):
            try:
                self.image, self.header = utils.get_image_from_path(image)
            except Exception as e:
                raise ValueError(f"Could not load image from path: {image}") from e
        elif isinstance(image, np.ndarray):
            self.image = image
            self.header = header
        else:
            raise TypeError(
                "Image must be a file path (str) or a NumPy array (np.ndarray)."
            )

        if self.cashe:
            if not os.path.exists(working_directory):
                os.makedirs(working_directory)
            self.working_directory = working_directory
        else:
            self.working_directory = None

        self.BMAJ, self.BMIN = None, None
        self.EFFRON, self.EFFGAIN, self.EXPTIME = None, None, None

        if self.mode == "radio" and self.header:
            try:
                self.BMAJ = self.header.get("BMAJ")
                self.BMIN = self.header.get("BMIN")
            except KeyError:
                print("Warning: Could not find BMAJ or BMIN in header.")
        elif self.mode == "optical" and self.header:
            try:
                self.EFFRON = self.header.get("EFFRON")
                self.EFFGAIN = self.header.get("EFFGAIN")
                self.EXPTIME = self.header.get("EXPTIME")
            except KeyError:
                print("Warning: Could not find EFFRON, EFFGAIN, or EXPTIME.")

    def phsf(self, lifetime_limit: float = 0.0, lifetime_limit_fraction: float = 1.0):
        if (
            getattr(self, "background_map", None) is None
            or getattr(self, "background_rms_map", None) is None
        ):
            raise ValueError(
                "Background maps must be set before running source finding."
            )

        if self.verbose:
            print("Thresholding to find source islands...")

        t0 = time.time()
        source_islands = source.create_source_islands(
            self.image,
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
            print(f"Thresholding took {t1 - t0:.2f} seconds.")
            print(f"Found {len(source_islands['positions'])} source islands.")

        iterable_islands = list(
            zip(source_islands["bboxes"], source_islands["positions"])
        )

        iterable_islands.sort(
            key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
            reverse=True,
        )

        if not iterable_islands:
            if self.verbose:
                print("No source islands to process.")
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
                print(f"Processing in parallel with {self.num_threads} threads.")

            optimal_chunksize = self.chunksize 
            
            # 1. Create shared memory blocks for all three arrays
            shm_img = shared_memory.SharedMemory(create=True, size=self.image.nbytes)
            shm_bg = shared_memory.SharedMemory(create=True, size=self.background_map.nbytes)
            shm_rms = shared_memory.SharedMemory(create=True, size=self.background_rms_map.nbytes)

            # 2. Copy the data into the shared memory buffers
            np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=shm_img.buf)[:] = self.image[:]
            np.ndarray(self.background_map.shape, dtype=self.background_map.dtype, buffer=shm_bg.buf)[:] = self.background_map[:]
            np.ndarray(self.background_rms_map.shape, dtype=self.background_rms_map.dtype, buffer=shm_rms.buf)[:] = self.background_rms_map[:]

            with get_context("spawn").Pool(
                self.num_threads,
                initializer=_worker_init,
                initargs=(
                    shm_img.name, self.image.shape, self.image.dtype,
                    shm_bg.name, self.background_map.shape, self.background_map.dtype,
                    shm_rms.name, self.background_rms_map.shape, self.background_rms_map.dtype
                ) 
            ) as p:
                
                results = list(
                    tqdm(
                        p.imap_unordered(
                            worker_func, 
                            iterable_islands, 
                            chunksize=optimal_chunksize
                        ),
                        total=len(iterable_islands),
                        disable=not self.verbose,
                        desc="Computing Homology",
                        dynamic_ncols=True
                    )
                )
            
            # 3. Clean up shared memory in the main process
            shm_img.close()
            shm_img.unlink()
            shm_bg.close()
            shm_bg.unlink()
            shm_rms.close()
            shm_rms.unlink()
        else:
            if self.verbose:
                print("Processing sequentially.")
            
            # Safely bind module-level globals for single-threaded execution
            global global_image, global_background_map, global_background_rms_map
            global_image = self.image
            global_background_map = self.background_map
            global_background_rms_map = self.background_rms_map
            
            for island in tqdm(
                iterable_islands, 
                disable=not self.verbose, 
                desc="Computing Homology", 
                dynamic_ncols=True
            ):
                results.append(worker_func(island))

        results = [res for res in results if res is not None and not res.is_empty()]
        if results:
            self.catalog = utils.combine_polars_catalogs(results)
        else:
            self.catalog = pl.DataFrame()

        t1 = time.time()
        if self.verbose:
            print(f"Homology computation took {t1 - t0:.2f} seconds.")  
            print("---------------CATALOG SUMMARY---------------------")
            print(f"Total sources detected: {self.catalog.height}")
            print(f"Number of large sources (area > {self.max_area_limit}): {self.catalog.filter(pl.col('area') > self.max_area_limit).height}")
            print("Average Background: ", self.background_map.mean())
            print("Average Background RMS: ", self.background_rms_map.mean())
            print("---------------------------------------------------")
            

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
            print("Calculating background map and RMS map...")
        t0 = time.time()
        self.detection_threshold = detection_threshold
        self.analysis_threshold = analysis_threshold

        bg_file = os.path.join(self.working_directory or "", "background_map.npy")
        rms_file = os.path.join(self.working_directory or "", "background_rms_map.npy")

        if self.cashe and os.path.exists(bg_file) and os.path.exists(rms_file):
            if self.verbose:
                print("Background maps exist. Loading from disk.")
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
            if self.cashe:
                np.save(bg_file, self.background_map)
                np.save(rms_file, self.background_rms_map)

        t1 = time.time()
        if self.verbose:
            print(f"Background calculation took {t1 - t0:.2f} seconds.")
