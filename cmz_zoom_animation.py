# zoom into the CMZ _even more_
import subprocess
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
import functools

from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization.wcsaxes.frame import EllipticalFrame
import astropy.visualization.wcsaxes

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.colors as mcolors




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

print("Done importing")

def fix_nans(img):
    kernel = Gaussian2DKernel(2)
    sm = convolve_fft(img, kernel)
    img[np.isnan(img)] = sm[np.isnan(img)]
    return img

def fixed_imshow(ax, data, **kwargs):
    im = ax.imshow(data, **kwargs)
    if kwargs.get('transform'):
        w = data.shape[1]
        h = data.shape[0]
        path = mp.path.Path([[-0.5,-0.5], [w-0.5,-0.5], [w-0.5,h-0.5], [-0.5,h-0.5], [-0.5,-0.5]])
        im.set_clip_path(path, transform=kwargs.get('transform'))



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


acesMUSTANGfeather = fits.open('/orange/adamginsburg/ACES/mosaics/continuum/12m_continuum_commonbeam_circular_reimaged_mosaic_MUSTANGfeathered.fits')
acesMUSTANGfeatherwcs = WCS(acesMUSTANGfeather[0].header)
#acesMUSTANGfeather[0].data[acesMUSTANGfeather[0].data < -0.0005] = 0


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

fn_co21repr = 'Planck_CO21_reproject_mollweide.fits'
fn_HI4PIrepr = 'HI4PI_reproject_mollweide.fits'
fn_ThermalDustrepr = 'Planck_ThermalDust_reproject_mollweide.fits'
if not os.path.exists(fn_ThermalDustrepr):
    print("Reprojecting CO, HI, dust")
    img_CO21, footprint = reproject_from_healpix(fn_CO21, target_header)
    img_CO21 = fix_nans(img_CO21)
    fits.PrimaryHDU(data=img_CO21, header=target_header).writeto(fn_co21repr)
    img_HI4PI, footprint = reproject_from_healpix(fn_HI4PI, target_header, field=5)
    fits.PrimaryHDU(data=img_HI4PI, header=target_header).writeto(fn_HI4PIrepr)
    img_ThermalDust, footprint = reproject_from_healpix(fn_ThermalDust, target_header)
    fits.PrimaryHDU(data=img_ThermalDust, header=target_header).writeto(fn_ThermalDustrepr)
else:
    print("Loading reprojected CO, HI, Dust")
    img_ThermalDust = fits.getdata(fn_ThermalDustrepr)
    img_CO21 = fits.getdata(fn_co21repr)
    img_HI4PI = fits.getdata(fn_HI4PIrepr)



cmap = pl.cm.Oranges
orange_transparent = cmap(np.arange(cmap.N))
orange_transparent[:,-1] = np.linspace(0, 1, cmap.N)
orange_transparent = ListedColormap(orange_transparent)

transorange = pl.cm.Oranges.copy()
transorange.set_under((0,0,0,0))
transoranges_r = pl.cm.Oranges_r.copy()
transoranges_r.set_under((0,0,0,0))
transoranges_r.set_bad((0,0,0,0))



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
# https://stackoverflow.com/a/15883620/814354
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

ax = pl.subplot(1,1,1, projection=WCS(target_header),
                frame_class=EllipticalFrame)
ax.imshow(rgb_full_scaled)
ax.axis('off')


nframes = 180
zoomfac = np.geomspace(1, 1/180., nframes)


dy0 = np.abs(np.diff(ax.get_ylim()))/2
dx0 = np.abs(np.diff(ax.get_xlim()))/2
dxdy = [dx0, dy0]
cx = np.mean(ax.get_xlim()) # - 2.6 # zoom in between Sgr A and Sgr B2
cy = np.mean(ax.get_ylim())

rgbcmz = np.array(PIL.Image.open('gc_fullres_6.jpg'))[::-1,:,:]
wwcmz = WCS(fits.Header.fromtextfile('gc_fullres_6.wcs'))

colors1 = pl.cm.gray_r(np.linspace(0., 1, 128))
colors2 = pl.cm.hot(np.linspace(0, 1, 128))

colors = np.vstack((colors1, colors2))
grey_hot = mcolors.LinearSegmentedColormap.from_list('grey_hot', colors)

atlasgal = fits.open('/orange/adamginsburg/galactic_plane_surveys/atlasgal/MOSAICS/apex_planck.fits')
lm20,_ = map(int, WCS(atlasgal[0].header).world_to_pixel(SkyCoord(-20*u.deg, 0*u.deg, frame='galactic')))
lp20,_ = map(int, WCS(atlasgal[0].header).world_to_pixel(SkyCoord( 20*u.deg, 0*u.deg, frame='galactic')))
agaldata = atlasgal[0].data
agalwcs = WCS(atlasgal[0].header)
# print(f"ATLASGAL from {atlasgal[0].data.shape} -> {agaldata.shape}")

def animate(n, nframes=nframes):
    dx0, dy0 = dxdy
    if n == 40:
        ax.imshow(atlasgal[0].data,
          norm=simple_norm(atlasgal[0].data, min_percent=5, max_percent=99.9, stretch='asinh'),
          transform=ax.get_transform(WCS(atlasgal[0].header)),
          cmap=orange_transparent, zorder=1)
        ax.imshow(atlasgal[0].data,
          norm=simple_norm(atlasgal[0].data, min_percent=99.9, max_percent=99.9999, stretch='linear'),
          transform=ax.get_transform(WCS(atlasgal[0].header)),
          cmap=transoranges_r, zorder=2)
    if n == 150:
        ax.imshow(acesMUSTANGfeather[0].data,
                  norm=simple_norm(acesMUSTANGfeather[0].data, stretch='log',
                                   vmin=0.0001, vmax=1.5,),
                  transform=ax.get_transform(acesMUSTANGfeatherwcs),
                  cmap=grey_hot,
                  zorder=180,
                 )


    dy = dy0 * zoomfac[n]
    dx = dx0 * zoomfac[n]

    ax.axis((cx-dx, cx+dx, cy-dy, cy+dy))
    print(f'{n}:{zoomfac[n]:0.1f}|', end='')

    return fig,

anim = animation.FuncAnimation(fig, animate, frames=nframes, repeat_delay=5000,
                               interval=50)
anim.save('zoom_anim_cmz_linear_HI-CO-Dust_m0.35.gif')

subprocess.check_call("""convert zoom_anim_cmz_linear_HI-CO-Dust_m0.35.gif \( -clone 0 -set delay 500 \) \( -clone 1-179 \) -delete 0-179 \( +clone -set delay 500 \) +swap +delete \( -clone 1--1 -reverse \) -loop 0 zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.gif""".split())

subprocess.check_call('ffmpeg -i zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.mp4'.split())

"""
convert zoom_anim_cmz_linear_HI-CO-Dust_m0.35.gif \( -clone 0 -set delay 500 \) \( -clone 1-179 \) -delete 0-179 \( +clone -set delay 500 \) +swap +delete \( -clone 1--1 -reverse \) -loop 0 zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.gif
ffmpeg -i zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.mp4
"""