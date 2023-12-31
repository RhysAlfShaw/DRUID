from DRUID import sf
from astropy.io import fits
from matplotlib import colors

PATH_image = '/data/typhon2/Rhys/data/KiDS/ADP.2019-02-11T13:02:26.713.fits'

hdullist = fits.open(PATH_image)
image = hdullist[0].data
header = hdullist[0].header

xmin = 10600
xmax = 15600
ymin = 10000
ymax = 15000

image = image[xmin:xmax,ymin:ymax]
print(type(colors.LogNorm(clip=True,vmin=1E-13,vmax=1E-9)))

findmysources = sf(image=image,image_path=None,mode='optical',
                  area_limit=5,smooth_sigma=1.5,nproc=1,GPU=True,
                   header=header,Xoff=xmin,Yoff=ymin)

findmysources.set_background(detection_threshold=2,analysis_threshold=2,mode='Radio')
findmysources.phsf()
findmysources.source_characterising(use_gpu=True)
findmysources.plot_sources(cmap="gray",figsize=(10,10),
                           norm=colors.LogNorm(clip=True,vmin=1E-13,vmax=1E-9),
                           save_path='test_gpu.png')
findmysources.save_catalogue('Test_catlogue.fits',filetype='fits',overwrite=True)
print(findmysources.catalogue)