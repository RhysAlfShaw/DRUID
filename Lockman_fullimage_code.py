from DRUID import sf

data_path = '/data/typhon2/Rhys/data/LoFAR/Lockman_hole/LockmanHole_ILT_mosaic-matched-MFS-image-pb.fits'
save_path = '/data/typhon2/Rhys/data/LoFAR/Lockman_hole/'

findmysources = sf(image=None, image_path=data_path,
                   pb_path=None, mode='Radio',
                   cutup=True, cutup_size=2000,cutup_buff=None,
                   output=True,GPU=True)

findmysources.set_background(detection_threshold=5,analysis_threshold=2,
                                bg_map_bool=True, box_size=100,mode='Radio')
# plot the background map
#print(findmysources.local_bg)

findmysources.phsf(lifetime_limit_fraction=1.1)
findmysources.source_characterising(gpu=True)
catalogue = findmysources.catalogue

# save contours to file hdf5 file

contours = {}
for i in range(len(findmysources.catalogue)):
    contours[findmysources.catalogue['ID'][i]] = findmysources.catalogue['contour'][i]
# save the coutours dict to hdf5 file
import h5py
hf = h5py.File(save_path+'DRUID_Contours_LOCAKMANreg.h5', 'w')
for key in contours.keys():
    hf.create_dataset(str(key), data=contours[key])
hf.close()

# save both version of the catalogue

# remove contours and enclosed_i from catalogue
findmysources.catalogue = findmysources.catalogue.drop(columns=['contour','enclosed_i'])
findmysources.save_catalogue(save_path+'DRUID_LockmanHole_Cat.fits',filetype='fits',overwrite=True)

# select only class 014 sources
findmysources.catalogue = findmysources.catalogue[(findmysources.catalogue['Class'] == 1) | (findmysources.catalogue['Class'] == 4) | (findmysources.catalogue['Class'] == 0)]
findmysources.save_catalogue(save_path+'DRUID_LockmanHole_Cat_CLASS014.fits',filetype='fits',overwrite=True)