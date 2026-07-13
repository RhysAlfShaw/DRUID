from DRUID import sf
import matplotlib.pyplot as plt
import numpy as np


def main():
    working_dir = "DRUID/temp"
    image_paths = [
        "/Users/rs17612/Documents/Radio_Data/3CRR/3C401",
        "/Users/rs17612/Documents/Radio_Data/3CRR/3C295",
        "/Users/rs17612/Documents/Radio_Data/3CRR/3C438",
        "/Users/rs17612/Documents/Radio_Data/3CRR/3C452",
        "/Users/rs17612/Documents/Radio_Data/3CRR/3C76P1",
    ]

    n_images = len(image_paths)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    i = 1
    if n_images == 1:
        axes = [axes]

    for ax, image_path in zip(axes, image_paths):
        findmysource = sf(
            image=image_path,
            mode="radio",
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

        catalog = findmysource.catalog
        # save polars catalog to temp folder
        if i == 1:
            catalog.write_parquet(
                f"{working_dir}/catalog_{image_path.split('/')[-1]}.parquet"
            )
            # save image as numpy array
            np.save(
                f"{working_dir}/image_{image_path.split('/')[-1]}.npy",
                findmysource.image,
            )
            # save background and background rms as numpy array
            np.save(
                f"{working_dir}/background_{image_path.split('/')[-1]}.npy",
                findmysource.background_map,
            )
            np.save(
                f"{working_dir}/background_rms_{image_path.split('/')[-1]}.npy",
                findmysource.background_rms_map,
            )
            i += 1

        print(catalog)
        ax.imshow(findmysource.image, cmap="gray", origin="lower")
        ax.scatter(
            catalog["y1"] + catalog["Island_Y"],
            catalog["x1"] + catalog["Island_X"],
            s=1,
            c="red",
            label="Source Islands",
        )
        contours = catalog["contour"].to_list()
        Island_X = catalog["Island_X"].to_list()
        Island_Y = catalog["Island_Y"].to_list()
        for i, contour in enumerate(contours):
            contour = np.array(contour)
            Island_X_val = Island_X[i]
            Island_Y_val = Island_Y[i]
            ax.plot(
                contour[:, 1] + Island_Y_val,
                contour[:, 0] + Island_X_val,
                # color="red",
                alpha=1,
                linewidth=1,
            )
        ax.set_title(f"{image_path.split('/')[-1]}")
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{working_dir}/all_source_islands_on_images.png")
    # plt.show()


if __name__ == "__main__":
    main()
