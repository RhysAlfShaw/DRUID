from DRUID import sf
from astropy.io import fits
from matplotlib import colors
import time

PATH_image = '/data/typhon2/Rhys/data/KiDS/ADP.2019-02-11T13:02:26.713.fits'
#save_path = '/home/rs17612/DPS_Comparison/'

hdullist = fits.open(PATH_image)
image = hdullist[0].data
header = hdullist[0].header

xmin = 10600
ymin = 10000
ymax = 12000
xmax = 12600
#print(image.shape)
image = image[xmin:xmax,ymin:ymax]
t0 = time.time()
findmysources = sf(image=image,image_path=None,mode='optical',
                pb_path = None, cutup = False, cutup_size = None, cutup_buff=None,
                output = True,
                area_limit=5, smooth_sigma=1,
                nproc=1, GPU=True, header=header, Xoff=None, Yoff=None)
print()
findmysources.set_background(detection_threshold=2,
                             analysis_threshold=2,
                             mode='Radio')
findmysources.phsf(lifetime_limit_fraction=2)
# filter catalogue based on Class 
#findmysources.catalogue = findmysources.catalogue[(findmysources.catalogue['Class'] == 1) | (findmysources.catalogue['Class'] == 4) | (findmysources.catalogue['Class'] == 0)]
findmysources.source_characterising(use_gpu=True)

print("Completed in {} seconds".format(time.time()-t0))
findmysources.plot_sources(cmap="gray",figsize=(20,20),
                           norm=colors.LogNorm(clip=True,vmin=1E-13,vmax=1E-9),
                           save_path='test_gpu_.png')

# # put the contours into a dictionary with the source ID as the key
# contours = {}
# for i in range(len(findmysources.catalogue)):
#     contours[findmysources.catalogue['ID'][i]] = findmysources.catalogue['contour'][i]
# # save the coutours dict to hdf5 file
# import h5py
# hf = h5py.File(save_path+'DRUID_Contours_CLASS014_reg.h5', 'w')
# for key in contours.keys():
#     hf.create_dataset(str(key), data=contours[key])
# hf.close()

# findmysources.catalogue = findmysources.catalogue.drop(columns=['contour','enclosed_i'])
# # cut the catalogue down to just the sources we want
# findmysources.catalogue = findmysources.catalogue[(findmysources.catalogue['Class'] == 1) | (findmysources.catalogue['Class'] == 4) | (findmysources.catalogue['Class'] == 0)]
# findmysources.save_catalogue(save_path+'DRUID_KIDS_Cat_CLASS014.fits',filetype='fits',overwrite=True)
# findmysources.save_polygons_to_ds9(save_path+'DRUID_Polygons_CLASS014.reg')