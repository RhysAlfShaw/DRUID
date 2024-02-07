from DRUID import sf
from astropy.io import fits
from matplotlib import colors
import time

PATH_image = '/data/typhon2/Rhys/data/KiDS/ADP.2019-02-11T13:02:26.713.fits'
out_path = '/home/rs17612/DPS_Comparion/'
# open and crop the image
hdullist = fits.open(PATH_image)
header = hdullist[0].header

image = hdullist[0].data

# make sure we are recording cpu compute time

t0 = time.time()
findmysources = sf(image=image,image_path=None,mode='optical',
                pb_path = None, cutup = True, cutup_size = 1000, cutup_buff=200,
                output = False,
                area_limit=5, smooth_sigma=1.5,
                nproc=1, GPU=True, header=header, Xoff=None, Yoff=None)

findmysources.set_background(detection_threshold=2,
                             analysis_threshold=2,
                             mode='mad_std')

findmysources.phsf(lifetime_limit_fraction=2)
findmysources.source_characterising(use_gpu=True)
t1 = time.time()
print("DRUID Finished! Completed in {} seconds".format(t1-t0))

# import matplotlib.pyplot as plt

# print("Class statsic")
# print(findmysources.catalogue['Class'].value_counts())

catalogue = findmysources.catalogue 

# print(catalogue.columns)
# # remove CLASS 2 and 3
catalogue = catalogue[catalogue['Class']!=2]
catalogue = catalogue[catalogue['Class']!=3]

# remove enclosed_i and the contours.
catalogue = catalogue.drop('enclosed_i',axis=1)
catalogue = catalogue.drop('contour',axis=1)

# save the catalogue
findmysources.catalogue = catalogue
findmysources.save_catalogue(out_path+'DRUID_KIDS_Cat_CLASS014.fits')

# plt.figure(figsize=(10,10))
# plt.imshow(image, cmap='gray', norm=colors.LogNorm(clip=True,vmin=1E-13,vmax=1E-9))
# plt.scatter(catalogue['Xc'],catalogue['Yc'],s=1,c='r',marker='x')
# for con in catalogue['contour']:
#     plt.plot(con[:,1],con[:,0])
# for con in catalogue_2['contour']:
#     plt.plot(con[:,1],con[:,0],c='b',linestyle='--',linewidth=0.5)
# plt.legend()
# plt.savefig('test.png')
# import numpy as np
# # check if there are duplicate polygons
# Polygons = findmysources.catalogue['contour']