version = "1.0"

import setproctitle

setproctitle.setproctitle("DRUID")

import numpy as np
import astropy
import os

# this prevent polars from using all available threads.
# Especially for multithreaded homology computation. otherwise we will spawn nested threads.
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl

import time

from multiprocessing import get_context
from tqdm import tqdm

from .src import utils
from .src import homology
from .src import background
from .src import source

DRUID_MESSAGE = """                 
#############################################
_______   _______          _________ ______  
(  __  \ (  ____ )|\     /|\__   __/(  __  \ 
| (  \  )| (    )|| )   ( |   ) (   | (  \  )
| |   ) || (____)|| |   | |   | |   | |   ) |
| |   | ||     __)| |   | |   | |   | |   | |
| |   ) || (\ (   | |   | |   | |   | |   ) |
| (__/  )| ) \ \__| (___) |___) (___| (__/  )
(______/ |/   \__/(_______)\_______/(______/ 
        
        
#############################################

Detector of astRonomical soUrces in optIcal and raDio images

Version: {}

For more information see:
https://github.com/RhysAlfShaw/DRUID
        """.format(
    version
)


def _worker(image: np.ndarray) -> "pl.DataFrame":
    """
    Worker function to compute homology for a single source island.
    """
    return homology.compute_homology(image)


class sf:
    def __init__(
        self,
        image: str | np.ndarray = None,
        mode: str = None,
        verbose: bool = True,
        area_limit: int = 0,
        smooth_sigma: float = 0,
        num_threads: int = 1,
        header: astropy.io.fits.header.Header = None,
        working_directory: str = "DRUID/temp",
    ):
        """

        Initialise DRUID and preform some basic checks.

        """

        print(DRUID_MESSAGE)

        self.mode = mode
        self.verbose = verbose
        self.area_limit = area_limit
        self.smooth_sigma = smooth_sigma
        self.num_threads = num_threads
        self.header = header

        if image is None:
            raise ValueError(
                "No image provided. Please provide a file path or a NumPy array."
            )

        if isinstance(image, str):
            try:
                self.image = utils.get_image_from_path(image)
            except Exception as e:
                raise ValueError(f"Could not load image from path: {image}") from e
        elif isinstance(image, np.ndarray):
            self.image = image
        else:
            raise TypeError(
                "Image must be a file path (str) or a NumPy array (np.ndarray)."
            )

        # check if there are files in the working directory
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
        self.working_directory = working_directory

    def phsf(self, lifetime_limit: float = 0.0, lifetime_limit_fraction: float = 2):
        """
        Runs the source findin algorithm on the image.

        Requires that the background has first been calculated.

        """
        if self.background_map is None or self.background_rms_map is None:
            raise ValueError(
                "Background map and RMS map must be set before running source finding."
                "Please call set_background() first. or assign them manually."
            )

        if self.verbose:
            print("Thresholding to find source islands...")
        # this function is rather slow.
        t0 = time.time()
        source_islands = source.create_source_islands(
            self.image,
            self.background_map,
            self.background_rms_map,
            detection_threshold=self.detection_threshold,
            analysis_threshold=self.analysis_threshold,
            area_limit=self.area_limit,
            verbose=self.verbose,
        )
        t1 = time.time()
        print(f"Thresholding took {t1 - t0:.2f} seconds.")
        t0 = time.time()
        if self.verbose:
            print(
                f"Found {len(source_islands['positions'])} source islands in the image with area limit {self.area_limit}."
            )

        images_to_process = source_islands["island_image"]

        if not images_to_process:
            if self.verbose:
                print("No source islands to process.")
            self.catalog = pl.DataFrame()
            return
        print(self.num_threads)

        if self.num_threads > 1:
            if self.verbose:
                print(
                    f"Processing {len(images_to_process)} source islands in parallel. with {self.num_threads} threads."
                )
            print("images to process:", len(images_to_process))
            batch_size = len(images_to_process) // self.num_threads
            print(f"Batch size: {batch_size}")
            with get_context("spawn").Pool(self.num_threads) as p:
                results = p.map(_worker, images_to_process, chunksize=batch_size)

        else:
            print(f"Processing {len(images_to_process)} source islands sequentially.")
            results = []
            for image in tqdm(images_to_process):
                results.append(homology.compute_homology(image))

            # combine the results catalogs to a single catalog

        if results:
            self.catalog = utils.combine_polars_catalogs(results)

        t1 = time.time()
        print(f"Homology computation took {t1 - t0:.2f} seconds.")

    def set_background(
        self,
        method: str = "rms",
        detection_threshold: int = 5,
        analysis_threshold: int = 3,
        box_size: tuple = (50, 50),  # kernal size for background calculation
        filter_size: tuple = (3, 3),  # size of median filter for background map
        kernel_size: int = 3,  # size of kernel for sigma clipping.
    ):
        """
        Calculate the background map of the image.
        This is required before running the source finding algorithm.
        """
        # Check if background maps already exist in the working directory.

        if self.verbose:
            print("Calculating background map and RMS map...")
        t0 = time.time()
        self.detection_threshold = detection_threshold
        self.analysis_threshold = analysis_threshold

        if os.path.exists(self.working_directory + "/background_map.npy"):
            if os.path.exists(self.working_directory + "/background_rms_map.npy"):
                if self.verbose:
                    print(
                        "Background map and RMS map already exist. Loading from disk."
                    )
                self.background_map = np.load(
                    self.working_directory + "/background_map.npy"
                )
                self.background_rms_map = np.load(
                    self.working_directory + "/background_rms_map.npy"
                )

        else:
            if self.verbose:
                print("Calculating background map and RMS map from image.")
            self.background_map, self.background_rms_map = (
                background.calculate_background_maps(
                    self.image,
                    bg_estimator=method,
                    box_size=box_size,
                    filter_size=(3, 3),
                    nsigma=detection_threshold,
                    kernel_size=3,
                )
            )
            # Save the background maps to disk for future use.
            np.save(self.working_directory + "/background_map.npy", self.background_map)
            np.save(
                self.working_directory + "/background_rms_map.npy",
                self.background_rms_map,
            )
        t1 = time.time()
        print(f"Background calculation took {t1 - t0:.2f} seconds.")

        if self.verbose:
            print("Background map and RMS map calculated.")
