from DRUID import sf
from astropy.table import Table
import matplotlib.pyplot as plt

def main(data_path, out_path, facet):
    
    findmysources = sf(image=None, image_path=data_path,
                   pb_path=None, mode='Radio',
                   cutup=True, cutup_size=1000,cutup_buff=50,
                   output=True,GPU=False) # feild is not dense enough to need GPU. as the acceleration helps with many matrix operations.

    findmysources.set_background(detection_threshold=5,analysis_threshold=2,
                                 bg_map_bool=True, box_size=100, mode='mad_std')
    # plot the background map
    #print(findmysources.local_bg)
    
    findmysources.phsf(lifetime_limit_fraction=1.1)
    findmysources.source_characterising(use_gpu=False)
    
    catalogue = findmysources.catalogue

    # apply flux scale from paper to compare with.
    flux_scale_from_paper = 1.21 # +/- 0.19
    flux_scale_from_paper_err = 0.19
    
    catalogue['Flux_total'] = catalogue['Flux_total'] * flux_scale_from_paper
    catalogue['flux_total_err'] = catalogue['Flux_total'] * flux_scale_from_paper_err/flux_scale_from_paper
    catalogue['Flux_peak'] = catalogue['Flux_peak'] * flux_scale_from_paper
    catalogue['flux_peak_err'] = catalogue['Flux_peak'] * flux_scale_from_paper_err/flux_scale_from_paper
    catalogue_astropy_table = Table.from_pandas(catalogue)
    catalogue_astropy_table.write(out_path + 'facet_' + facet + '_catalogue_rmsnoise.fits', overwrite=True)
    findmysources.create_polygons_fast() # some are not linking this and instead it will fail. But we do create the catalogue either way.
    findmysources.save_polygons_to_ds9(out_path + 'facet_' + facet + '_polygons.reg')
    
arguments=[
    "facet 01 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_01_beamcorrected_kernel25_mgain0p5_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 02 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_02_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 03 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_03_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 04 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_04_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 05 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_05_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 06 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_06_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 07 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_07_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 08 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_08_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 09 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_09_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 10 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_10_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 11 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_11_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 12 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_12_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    #"facet 13 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_13_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 14 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_14_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 15 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_15_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 16 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_16_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 17 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_17_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 18 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_18_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 19 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_19_kernel25-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 20 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_20_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 21 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_21_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 22 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_22_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 23 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_23_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 24 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_24_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
    # "facet 25 data_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_25_beamcorrected_kernel25_mgain0p6_pd8192_nmiter10_automask5-MFS-image-pb.fits.trimmed.scaled.fits out_path /data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/"
]

# Split each argument string into a list of individual arguments
split_arguments = [arg.split() for arg in arguments]

# Convert each list of arguments into a dictionary
argument_dicts = [
    {split_arguments[i][j]: split_arguments[i][j + 1] for j in range(0, len(split_arguments[i]), 2)}
    for i in range(len(split_arguments))
]


i = 0
main(argument_dicts[i]['data_path'], argument_dicts[i]['out_path'], argument_dicts[i]['facet'])