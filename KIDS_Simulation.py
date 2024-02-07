import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import numpy as np
from tqdm import tqdm
import pandas as pd

np.random.seed(42)

# create a Simular image with known sources and see if DRUID can find them.
image_path = '/data/typhon2/Rhys/data/KiDS/ADP.2019-02-11T13:02:26.713.fits'
Catalogue_Path = '/data/typhon2/Rhys/data/KiDS/ADP.2019-02-11T13:02:26.716.fits'

def model_gaussian(shape,xc,yc,amp,sigmax,sigmay,theta):
    
    """
    
    Function to create a 2d gaussian model
    
    """
    
    x = np.arange(0,shape[1],1)
    y = np.arange(0,shape[0],1)
    x,y = np.meshgrid(x,y)
    a = (np.cos(theta)**2)/(2*sigmax**2) + (np.sin(theta)**2)/(2*sigmay**2)
    b = -(np.sin(2*theta))/(4*sigmax**2) + (np.sin(2*theta))/(4*sigmay**2)
    c = (np.sin(theta)**2)/(2*sigmax**2) + (np.cos(theta)**2)/(2*sigmay**2)
    z = amp*np.exp(-(a*((x-xc)**2) + 2*b*(x-xc)*(y-yc) + c*((y-yc)**2)))
    
    return z


def generate_2d_gaussian(A,shape, center, sigma_x, sigma_y, angle_deg=0,norm=True):
    """
    
    Generate a 2D elliptical Gaussian distribution on a 2D array.

    Parameters:
    
        shape (tuple): Shape of the output array (height, width).
        center (tuple): Center of the Gaussian distribution (x, y).
        sigma_x (float): Standard deviation along the x-axis.
        sigma_y (float): Standard deviation along the y-axis.
        angle_deg (float): Rotation angle in degrees (default is 0).

    Returns:
    
        ndarray: 2D array containing the Gaussian distribution.
    
    """
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x_c, y_c = center
    angle_rad = np.radians(angle_deg)

    # Rotate coordinates
    
    x_rot = (x - x_c) * np.cos(angle_rad) - (y - y_c) * np.sin(angle_rad)
    y_rot = (x - x_c) * np.sin(angle_rad) + (y - y_c) * np.cos(angle_rad)

    # Calculate Gaussian values
    
    gaussian = A *np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))

    if norm:
        return gaussian / (2 * np.pi * sigma_x * sigma_y) 
    else:
        return gaussian
    
def mag_to_flux(mag):
    # vega mag tp flux

    return 10**(-0.4*(mag))

def flux_to_peak(flux,sigma):
    return flux/(2*np.pi*sigma**2)

hdullist = fits.open(image_path)
image = hdullist[0].data
header = hdullist[0].header

#image = image[10600:12600,10000:12000]

from astropy.table import Table
catalogue = Table.read(Catalogue_Path)
background_mean = -9.624E-14
background_std = 1.445E-12

# loop though th catalogue and get the sources peak flux using its position and the image

    
#mean_peak_flux = np.mean(peak_flux)
#std_peak_flux = np.std(peak_flux)

N = 55_000 # number of sources to simulate half the number of sources in the true catalogue

# psf is fwhm 8.348750000000E-01 arcsecs
# 1 pixel = 0.2 arcsecs
# fwhm = 1.1774 pixels
# FWHM = 2.355 sigma

psf_fwhm_arsec = 8.348750000000E-01
psf_fwhm_pixels = psf_fwhm_arsec/0.2
psf_sigma = psf_fwhm_pixels/2.355

# sigma = 1.7744 pixels

sigma = psf_sigma

# create a new array of peak fluxes with a gaussian distribution

mags = np.random.choice(catalogue["MAG_AUTO"],N)

# remove sources lower than 0

# remove any mags larfer than 30

mags = mags[(mags > 0) & (mags < 30)]
N = len(mags)
print("Number of sources: {}".format(N))

# create a image made up of these sources using the model_gaussian function

model = np.zeros_like(image)
params = []
xc = np.random.uniform(0,image.shape[0],N)
yc = np.random.uniform(0,image.shape[1],N)

for i in tqdm(range(N),desc='Creating Sim Image',total=N):    
    Flux = mag_to_flux(mags[i])
    amp = flux_to_peak(Flux,sigma)
    model[int(xc[i]),int(yc[i])] = Flux
    params.append([xc[i],yc[i],amp,sigma,Flux,mags[i]])
    
# convolve the model with a gaussian kernel

from scipy.ndimage import gaussian_filter
print(sigma)
model = gaussian_filter(model,sigma=sigma)

# save image
hdu = fits.PrimaryHDU(model)
hdu.header = header
hdu.writeto('Sim_imageKIds_model_2.fits',overwrite=True)


hdu = fits.PrimaryHDU(image+model)
hdu.header = header
hdu.writeto('Sim_imageKIds_combined_2.fits',overwrite=True)


# make params a pandas dataframe
params = np.array(params)
params = pd.DataFrame(params,columns=['yc','xc','amp','sigma','Flux','MAG'])
# save as fits file table
# using the header and WCS from the original image create the RA and Dec columns
from astropy.wcs import WCS
wcs = WCS(header)
RA,DEC = wcs.all_pix2world(yc,xc,0)
params['RA'] = RA
params['DEC'] = DEC
# save as fits file
Paraks = Table.from_pandas(params)
Paraks.write('Sim_imageKIds_params_2.fits',overwrite=True)

print("Saved Simulated Image and Parameters")
print("Creating Histogram")

#plt.imshow(image, cmap='gray_r',vmin=0,vmax=1E-10)
bins = np.linspace(13,40,100)
plt.hist(Paraks['MAG'], bins=bins, label='Simulated',histtype='step')
plt.hist(catalogue['MAG_AUTO'], bins=bins, label='Provided',histtype='step')
plt.xlabel('Magnitude')
plt.ylabel('Number of objects')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('Sim_imageKIds_2.png')

# need to generate a list of gaussian sources
