from DRUID import sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

Path_NEW_EROSITA = #'/Users/rs17612/Documents/Xray_Data/A3158/TEST_image_bin64.fits'

from astropy.io import fits

hdulist = fits.open(Path_NEW_EROSITA)
hdulist.info()
header = hdulist[0].header
# crop the nans from the image

data = hdulist[0].data[~np.isnan(hdulist[0].data).all(axis=1)]
data = data[:,~np.isnan(data).all(axis=0)]

findmysources = sf(image=data,mode='optical',area_limit=5,
                   smooth_sigma=1.5,GPU=False,
                   header=header)
findmysources.set_background(detection_threshold=5,
                             analysis_threshold=5,
                             mode='mad_std')
findmysources.phsf(lifetime_limit=findmysources.local_bg,lifetime_limit_fraction=1.5)
findmysources.source_characterising(use_gpu=False)

plt.imshow(data,vmin=0,vmax=50)
for con in findmysources.catalogue['contour']:
    plt.plot(con[:,1],con[:,0])
plt.show()