## script that tests DRUID using MIGHTEE data (simplier image size)

from DRUID import sf
import matplotlib.pyplot as plt

PATH_MIGHTEE_IMAGE = '/data/typhon2/Rhys/data/MIGHTEE/MIGHTEE_Continuum_Early_Science_COSMOS_r0p0.app.restored.circ.fits'

findmysources = sf(image=None, image_path=PATH_MIGHTEE_IMAGE,
                   pb_path=None, mode='Radio',
                   cutup=True, cutup_size=1000,cutup_buff=100,
                   output=True,GPU=True)
findmysources.set_background(detection_threshold=5,analysis_threshold=2,mode='mad_std',
                             bg_map_bool=False, box_size=1000)

# plot the background map
plt.figure(figsize=(10,10))
plt.imshow(findmysources.local_bg)
plt.savefig('MIGHTEE_DRUID_TEST_BG.png')

findmysources.phsf(lifetime_limit_fraction=1.1)

findmysources.source_characterising(use_gpu=False) 
# save catalogue to file
# remove contours and enclosed_i from catalogue
findmysources.catalogue = findmysources.catalogue.drop(columns=['contour','enclosed_i'])

findmysources.save_catalogue('MIGHTEE_DRUID_TEST_Cat.fits',filetype='fits',overwrite=True)
#print(findmysources.catalogue)
# save catalogue to file
# plot source positions on image
plt.figure(figsize=(10,10))
plt.imshow(findmysources.image, vmax=0.001,vmin=0)
plt.scatter(findmysources.catalogue['Xc'],findmysources.catalogue['Yc'],s=1,c='r')
contours = findmysources.catalogue['contour']
for con in contours:
    plt.plot(con[:,1],con[:,0])
plt.xlim(2000,3000)
plt.ylim(2000,3000)
plt.savefig('MIGHTEE_DRUID_TEST.png')