version = "1.0"

import setproctitle


setproctitle.setproctitle("DRUID")

import numpy as np
import astropy
from multiprocessing import Pool


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

        source_islands = source.create_source_islands(
            self.image,
            self.background_map,
            self.background_rms_map,
            detection_threshold=self.detection_threshold,
            analysis_threshold=self.analysis_threshold,
            area_limit=self.area_limit,
            verbose=self.verbose,
        )

        if self.verbose:
            print(
                f"Found {len(source_islands['positions'])} source islands in the image with area limit {self.area_limit}."
            )

        images_to_process = source_islands["island_image"]

        if not images_to_process:
            if self.verbose:
                print("No source islands to process.")
            # Create an empty catalog if no islands are found
            import polars as pl

            self.catalog = pl.DataFrame()
            return

        if self.num_threads > 1:
            if self.verbose:
                print(
                    f"Processing {len(images_to_process)} source islands in parallel. with {self.num_threads} threads."
                )

            with Pool(self.num_threads) as p:
                results = p.map(_worker, images_to_process)
        else:
            results = []
            for image in images_to_process:
                results.append(_worker(image))

        # combine the results catalogs to a single catalog
        if results:
            self.catalog = utils.combine_polars_catalogs(results)

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
        if self.verbose:
            print("Calculating background map and RMS map...")
        self.detection_threshold = detection_threshold
        self.analysis_threshold = analysis_threshold

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

        if self.verbose:
            print("Background map and RMS map calculated.")
