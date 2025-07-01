from DRUID import sf


def main():
    image_path = "DRUID/temp/dummy_image.fits"
    test_optical_image = "/Users/rs17612/Documents/Optical_IR_Data/EUCLID/EUC_MER_BGSUB-MOSAIC-VIS_TILE101158277-BB647A_20240122T115602.395130Z_00.00.fits"
    findmysource = sf(
        image=test_optical_image, mode="optical", area_limit=5, num_threads=2
    )
    findmysource.set_background()
    findmysource.phsf()

    # pint the catalog
    catalog = findmysource.catalog
    print("Catalog:", catalog)


if __name__ == "__main__":
    main()
