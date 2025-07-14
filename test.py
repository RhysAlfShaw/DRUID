from DRUID import sf


def create_dummy_image():
    """
    Create a dummy FITS image for testing purposes.
    """
    from astropy.io import fits
    import numpy as np

    # Create a dummy image with random data
    data = np.random.normal(size=(10000, 10000)).astype(np.float32)
    # add many bright sources
    for _ in range(500):
        x = np.random.randint(0, 10000)
        y = np.random.randint(0, 10000)
        data[x, y] += np.random.uniform(500, 10000)

    # convolve the image with a Gaussian kernel to simulate a more realistic image
    from scipy.ndimage import gaussian_filter

    data = gaussian_filter(data, sigma=5)

    # Create a FITS file
    hdu = fits.PrimaryHDU(data)
    hdu.writeto("DRUID/temp/dummy_image.fits", overwrite=True)

    # # plot the image to verify
    # import matplotlib.pyplot as plt

    # img_size = [2000, 3000, 5000, 10000]
    # bg_time = [0.79, 1.7, 4.35, 17.36]
    # thresh_time = [5.74, 16.2, 50, 60 * 7]
    # # fit exponential curves to the data
    # from scipy.optimize import curve_fit

    # def exp_func(x, a, b):
    #     return a * np.exp(b * x)

    # popt_bg, _ = curve_fit(exp_func, img_size, bg_time)
    # popt_thresh, _ = curve_fit(exp_func, img_size, thresh_time)

    # plt.plot(img_size, bg_time, label="Background Calculation Time")
    # plt.plot(img_size, thresh_time, label="Thresholding Time")
    # extrapolated_img_size = np.linspace(0, 20000, 100)
    # plt.plot(
    #     extrapolated_img_size,
    #     exp_func(np.array(extrapolated_img_size), *popt_bg),
    #     linestyle="--",
    #     color="blue",
    # )

    # plt.plot(
    #     extrapolated_img_size,
    #     exp_func(np.array(extrapolated_img_size), *popt_thresh),
    #     linestyle="--",
    #     color="orange",
    # )
    # plt.legend()
    # plt.xlabel("Image Size (pixels)")
    # plt.ylabel("Time (seconds)")
    # plt.title("Background Calculation and Thresholding Time vs Image Size")
    # plt.show()


def main():
    create_dummy_image()  # Create a dummy image for testing
    image_path = "DRUID/temp/dummy_image.fits"
    # image_path = "/Users/rs17612/Documents/Optical_IR_Data/EUCLID/EUC_MER_BGSUB-MOSAIC-VIS_TILE101158277-BB647A_20240122T115602.395130Z_00.00.fits"
    findmysource = sf(image=image_path, mode="optical", area_limit=5, num_threads=1)
    findmysource.set_background()
    findmysource.phsf()

    # pint the catalog
    catalog = findmysource.catalog
    print("Catalog:", catalog)


if __name__ == "__main__":
    main()
