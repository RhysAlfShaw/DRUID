from DRUID import sf


def main():
    path = "/Users/rs17612/Documents/Optical_IR_Data/EUCLID/EUC_MER_BGSUB-MOSAIC-DES-Z_TILE102026098-321DAA_20240407T183900.016572Z_00.00.fits"
    working_dir = "DRUID/temp_new"
    findmysource = sf(
        image=path,
        mode="optical",
        area_limit=15,
        num_threads=2,
        working_directory=working_dir,
        cashe=False,
    )
    findmysource.set_background(
        detection_threshold=5,
        analysis_threshold=3,
        box_size=50,
    )
    findmysource.phsf(lifetime_limit_fraction=1.4)
    print(findmysource.catalog)


if __name__ == "__main__":
    main()
