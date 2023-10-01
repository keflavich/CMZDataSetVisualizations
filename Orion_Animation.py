# wip
import numpy as np
import regions
from astropy import coordinates
from astropy import units as u, constants
import astropy.visualization.wcsaxes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset
import os
from astropy.utils.data import download_file
from astropy.io import fits
from spectral_cube import SpectralCube
from astropy.visualization import simple_norm
import shutil
import reproject
from reproject import reproject_from_healpix, reproject_to_healpix

from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization.wcsaxes.frame import EllipticalFrame
import astropy.visualization.wcsaxes

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation



from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u, constants
from astropy.convolution import convolve_fft, Gaussian2DKernel, Gaussian1DKernel
import healpy
import PIL
import pyavm
import regions

import pylab as pl
import numpy as np
pl.rcParams['image.interpolation'] = 'none'

def fix_nans(img):
    kernel = Gaussian2DKernel(2)
    sm = convolve_fft(img, kernel)
    img[np.isnan(img)] = sm[np.isnan(img)]
    return img


target_header = fits.Header.fromstring("""
NAXIS   =                    2
NAXIS1  =                 1920
NAXIS2  =                  960
CTYPE1  = 'GLON-MOL'
CRPIX1  =                960.5
CRVAL1  =                  0.0
CDELT1  =             -0.16875
CUNIT1  = 'deg     '
CTYPE2  = 'GLAT-MOL'
CRPIX2  =                480.5
CRVAL2  =                  0.0
CDELT2  =              0.16875
CUNIT2  = 'deg     '
COORDSYS= 'icrs    '
""", sep='\n')

target_header_innergal = fits.Header.fromstring("""
NAXIS   =                    2
NAXIS1  =                 3840
NAXIS2  =                  960
CTYPE1  = 'GLON-MOL'
CRPIX1  =               1920.5
CRVAL1  =                  0.0
CDELT1  =                -0.05
CUNIT1  = 'deg     '
CTYPE2  = 'GLAT-MOL'
CRPIX2  =                480.5
CRVAL2  =                  0.0
CDELT2  =                 0.05
CUNIT2  = 'deg     '
COORDSYS= 'icrs    '
""", sep='\n')

target_header_orion = fits.Header.fromstring("""
NAXIS   =                    2
NAXIS1  =                 1920
NAXIS2  =                  960
CTYPE1  = 'RA---MOL'
CRPIX1  =                960.5
CRVAL1  =                83.82
CDELT1  =             -0.16875
CUNIT1  = 'deg     '
CTYPE2  = 'DEC--MOL'
CRPIX2  =                480.5
CRVAL2  =                -5.39
CDELT2  =              0.16875
CUNIT2  = 'deg     '
COORDSYS= 'icrs    '
""", sep='\n')


fn_ThermalDust = "COM_CompMap_ThermalDust-commander_2048_R2.00.fits"
if not os.path.exists(fn_ThermalDust):
    planck_ThermalDust = download_file(f"https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/{fn_ThermalDust}")
    shutil.move(planck_ThermalDust, fn_ThermalDust)
planck_ThermalDust = fits.open(fn_ThermalDust)


fn_CO21 = "COM_CompMap_CO21-commander_2048_R2.00.fits"
if not os.path.exists(fn_CO21):
    planck_CO21 = download_file(f"https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/{fn_CO21}")
    shutil.move(planck_CO21, fn_CO21)
planck_CO21 = fits.open(fn_CO21)

fn_HI4PI = 'NHI_HPX.fits'
if not os.path.exists(fn_HI4PI):
    HI4PI = download_file(f"https://lambda.gsfc.nasa.gov/data/foregrounds/HI4PI/{fn_HI4PI}")
    shutil.move(HI4PI, fn_HI4PI)
HI4PI = fits.open(fn_HI4PI)


img_CO21, footprint = reproject_from_healpix(fn_CO21, target_header)
img_CO21 = fix_nans(img_CO21)
img_HI4PI, footprint = reproject_from_healpix(fn_HI4PI, target_header, field=5)
img_ThermalDust, footprint = reproject_from_healpix(fn_ThermalDust, target_header)

# not neceessarily needed; could be useful if we want higher-res starting point or no Mollweide
img_HI4PI_innergal, footprint = reproject_from_healpix(fn_HI4PI, target_header_innergal, field=5)
img_ThermalDust_innergal, footprint = reproject_from_healpix(fn_ThermalDust, target_header_innergal)
img_CO21_innergal, footprint = reproject_from_healpix(fn_CO21, target_header_innergal)

img_HI4PI_orion, footprint = reproject_from_healpix(fn_HI4PI, target_header_orion, field=5)
img_CO21_orion, footprint = reproject_from_healpix(fn_CO21, target_header_orion)
img_CO21_orion = fix_nans(img_CO21_orion)
img_ThermalDust_orion, footprint = reproject_from_healpix(fn_ThermalDust, target_header_orion)


if False:
    rgb = np.array([simple_norm(img_HI4PI_innergal,       min_percent=0.01, max_percent=99.99, log_a=1e1, stretch='log')(img_HI4PI_innergal),
                    simple_norm(img_CO21_innergal,        min_percent=0.01, max_percent=99.99, log_a=3e1, stretch='log')(img_CO21_innergal),
                    simple_norm(img_ThermalDust_innergal, min_percent=0.01, max_percent=99.99, log_a=5e2, stretch='log')(img_ThermalDust_innergal)]).T.swapaxes(0,1)
    hsv = rgb_to_hsv(rgb)
    hsv[:,:,0] += -0.35  # 0.25 = 90/360
    hsv[:,:,0] = hsv[:,:,0] % 1
    rgb_scaled = hsv_to_rgb(hsv)
    rgb_scaled[rgb_scaled > 1] = 1
    rgb_scaled[rgb_scaled < 0] = 0

rgb_full = np.array([simple_norm(img_HI4PI,       min_percent=0.01, max_percent=99.99, log_a=2e1, stretch='log')(img_HI4PI),
                     simple_norm(img_CO21,        min_percent=0.01, max_percent=99.90, log_a=2e1, stretch='log')(img_CO21),
                     simple_norm(img_ThermalDust, min_percent=0.01, max_percent=99.90, log_a=5e2, stretch='log')(img_ThermalDust)]).T.swapaxes(0,1)
hsv = rgb_to_hsv(rgb_full)
hsv[:,:,0] += -0.35  # 0.25 = 90/360
hsv[:,:,0] = hsv[:,:,0] % 1
rgb_full_scaled = hsv_to_rgb(hsv)
rgb_full_scaled[rgb_full_scaled > 1] = 1
rgb_full_scaled[rgb_full_scaled < 0] = 0

fig = pl.figure(figsize=(10,5), dpi=200, frameon=False)
ax = pl.subplot(1,1,1, projection=WCS(target_header),
                frame_class=EllipticalFrame)
ax.imshow(rgb_full_scaled)
ax.axis('off')

nframes = 420
# zoom for 300 frames, then pan
zoomfac = np.hstack([np.geomspace(1, 1/1920., 300),
                     np.ones(120, dtype='float')/1920.])
raise NotImplementedError

sgrb2 = SkyCoord.from_name('Sgr B2')
sgra = SkyCoord.from_name('Sgr A')
sgrc = SkyCoord.from_name('Sgr C')

# pan
cx, cy = ax.wcs.world_to_pixel(sgrb2)
cx2, cy2 = ax.wcs.world_to_pixel(sgrc)
cxs = np.hstack([np.ones(300)*cx, np.linspace(cx, cx2, 120)])
cys = np.hstack([np.ones(300)*cy, np.linspace(cy, cy2, 120)])

dy0 = rgb_full_scaled.shape[0]
dx0 = rgb_full_scaled.shape[1]

rgbcmz = np.array(PIL.Image.open('gc_fullres_6.jpg'))[::-1,:,:]
wwcmz = WCS(fits.Header.fromtextfile('gc_fullres_6.wcs'))

aces7m = fits.open('/orange/adamginsburg/ACES/mosaics/7m_continuum_mosaic.fits')
aces7mwcs = WCS(aces7m[0].header)
aces7m[0].data[aces7m[0].data < -0.0005] = 0
#aces7mrepr,_ = reproject.reproject_interp(aces7m, ww, shape_out=rgb.shape[:2])
#aces7mrepr[aces7mrepr < 0] = np.nan

aces12m = fits.open('/orange/adamginsburg/ACES/mosaics/12m_continuum_mosaic.fits')
aces12mwcs = WCS(aces12m[0].header)
aces12m[0].data[aces12m[0].data < -0.0005] = 0

atlasgal = fits.open('/orange/adamginsburg/galactic_plane_surveys/atlasgal/MOSAICS/apex_planck.fits')

cmap = pl.cm.Oranges
orange_transparent = cmap(np.arange(cmap.N))
orange_transparent[:,-1] = np.linspace(0, 1, cmap.N)
orange_transparent = ListedColormap(orange_transparent)

transorange = pl.cm.Oranges.copy()
transorange.set_under((0,0,0,0))


def animate(n, nframes=nframes):
    if n == 40:
        ax.imshow(atlasgal[0].data,
          norm=simple_norm(atlasgal[0].data, min_percent=5, max_percent=99.9, stretch='asinh'),
          transform=ax.get_transform(WCS(atlasgal[0].header)),
          cmap=orange_transparent)
    if n == 80:
        ax.imshow(rgbcmz,
          transform=ax.get_transform(wwcmz))

    if n == 180:
        ax.imshow(aces7m[0].data,
                  norm=simple_norm(aces7m[0].data, stretch='log', max_percent=99.96, min_percent=1),
                  transform=ax.get_transform(aces7mwcs),
                  cmap=orange_transparent)

    if n == 240:
        ax.imshow(aces12m[0].data,
                     norm=simple_norm(aces12m[0].data, stretch='log',
                                      min_percent=None, max_percent=None,
                                      min_cut=-0.0005, max_cut=0.05,),
                     cmap='gray',
                  transform=ax.get_transform(aces12mwcs),
                    )
        ax.imshow(aces12m[0].data,
                     norm=simple_norm(aces12m[0].data, stretch='asinh',
                                      min_percent=None, max_percent=99.99,
                                      min_cut=0.05,),
                     cmap=transorange,
                  transform=ax.get_transform(aces12mwcs), )



    dy = dy0 * zoomfac[n]
    dx = dx0 * zoomfac[n]
    cx, cy = cxs[n], cys[n]

    ax.axis((cx-dx, cx+dx, cy-dy, cy+dy))

    print(f'{n}|', end='', flush=True)

    return fig,

anim = animation.FuncAnimation(fig, animate, frames=nframes, repeat_delay=5000,
                               interval=50)
anim.save('zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35.gif')
