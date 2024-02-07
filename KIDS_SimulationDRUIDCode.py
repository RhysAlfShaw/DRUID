from DRUID import sf
from astropy.io import fits
from matplotlib import colors
import time


PATH_image = 'Sim_imageKIds_combined_2.fits'

hdullist = fits.open(PATH_image)

image = hdullist[0].data
# crop the image


header = hdullist[0].header

findmysources = sf(image=image,image_path=None,mode='optical',
                pb_path = None, cutup = True, cutup_size = 1000, cutup_buff=100,
                output = True,
                area_limit=5,smooth_sigma=1.5,
                nproc=1,GPU=True, header=header,Xoff=None, Yoff=None)

findmysources.set_background(detection_threshold=2,
                             analysis_threshold=2,
                             mode='mad_std')

findmysources.phsf(lifetime_limit_fraction=2)

findmysources.source_characterising(use_gpu=True)

#findmysources.save_catalogue('DRUID_KIDS_Cat_SIMULATIUED.fits',filetype='fits',overwrite=True)

findmysources.catalogue = findmysources.catalogue[(findmysources.catalogue['Class'] == 1) | (findmysources.catalogue['Class'] == 4) | (findmysources.catalogue['Class'] == 0)]
findmysources.catalogue = findmysources.catalogue.drop(columns=['contour','enclosed_i'])

# Prehaps we should get the ra and dec from the Peak Position. 

def get_rad_dec_from_xy(x,y,header):
    """
    Convert x,y pixel coordinates to RA and Dec using the WCS information in the header.
    """
    from astropy.wcs import WCS
    wcs = WCS(header)
    ra,dec =  wcs.all_pix2world(x,y,0)
    return ra,dec

image_header = fits.getheader('Sim_imageKIds_combined_2.fits')
Peak_x = findmysources.catalogue['y1'] #+ findmysources.catalogue['X0_cutout']
Peak_y = findmysources.catalogue['x1'] #+ findmysources.catalogue['Y0_cutout']

findmysources.catalogue['RA'],findmysources.catalogue['DEC'] = get_rad_dec_from_xy(Peak_x,Peak_y,image_header)
print(findmysources.catalogue)
print(findmysources.catalogue.describe())
print(findmysources.catalogue.columns)
findmysources.save_catalogue('DRUID_KIDS_Cat_SIMS_peak_SEED.fits',filetype='fits',overwrite=True)

# contours = {}

# for i in range(len(findmysources.catalogue)):
#
#     contours[findmysources.catalogue['ID'][i]] = findmysources.catalogue['contour'][i]

# save the coutours dict to hdf5 file
# only have Class=0,1,4 in the catalogue
