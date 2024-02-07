from DRUID import sf
import argparse

from astropy.table import Table


parser = argparse.ArgumentParser(description='Process Lockman facet.')

parser.add_argument('--facet', type=str, default='0',
                    help='facet number')

parser.add_argument('--data_path', type=str, default='/data/lockman/',
                    help='data path')

parser.add_argument('--out_path', type=str, default='/data/lockman/',
                    help='output path')

args = parser.parse_args()

def main():
    findmysources = sf(image=None, image_path=args.data_path,
                   pb_path=None, mode='Radio',
                   cutup=True, cutup_size=1000,cutup_buff=50,
                   output=False,GPU=False) # feild is not dense enough to need GPU. as the acceleration helps with many matrix operations.

    findmysources.set_background(detection_threshold=5,analysis_threshold=2,
                                 bg_map_bool=True, box_size=50, mode='mad_std')
    findmysources.phsf(lifetime_limit_fraction=1.1)
    findmysources.source_characterising(use_gpu=False)
    
    catalogue = findmysources.catalogue

    # apply flux scale from paper to compare with.
    flux_scale_from_paper = 1.21 # +/- 0.19
    flux_scale_from_paper_err = 0.19
    # drop the contours and enclosed_i columns, to prevent object pickling error.
    catalogue = catalogue.drop(columns=['contour','enclosed_i'])
    catalogue['Flux_total'] = catalogue['Flux_total'] * flux_scale_from_paper
    catalogue['flux_total_err'] = catalogue['Flux_total'] * flux_scale_from_paper_err/flux_scale_from_paper
    catalogue['Flux_peak'] = catalogue['Flux_peak'] * flux_scale_from_paper
    catalogue['flux_peak_err'] = catalogue['Flux_peak'] * flux_scale_from_paper_err/flux_scale_from_paper
    catalogue_astropy_table = Table.from_pandas(catalogue)
    catalogue_astropy_table.write(args.out_path + 'facet_' + args.facet + '_catalogue_rmsnoise.fits', overwrite=True)
    
    
if __name__ == "__main__":
    main()