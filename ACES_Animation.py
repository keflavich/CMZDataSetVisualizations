# zoom into the CMZ _even more_
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

if False:
    # not necessarily needed; could be useful if we want higher-res starting point or no Mollweide
    img_HI4PI_innergal, footprint = reproject_from_healpix(fn_HI4PI, target_header_innergal, field=5)
    img_ThermalDust_innergal, footprint = reproject_from_healpix(fn_ThermalDust, target_header_innergal)
    img_CO21_innergal, footprint = reproject_from_healpix(fn_CO21, target_header_innergal)

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

nframes = 540
# zoom for 300 frames, then pan
zoomfac = np.hstack([np.geomspace(1, 1/1920., 300),
                     np.ones(240, dtype='float')/1920.])


sgrb2 = SkyCoord.from_name('Sgr B2')
sgra = SkyCoord.from_name('Sgr A')
sgrc = SkyCoord.from_name('Sgr C')
sgrc = SkyCoord(359.49154*u.deg, -0.09789*u.deg, frame='galactic')

# pan
cx, cy = WCS(target_header).world_to_pixel(sgrb2)
cx2, cy2 = WCS(target_header).world_to_pixel(sgrc)
cxs = np.hstack([np.ones(300)*cx, np.linspace(cx, cx2, 240)])
cys = np.hstack([np.ones(300)*cy, np.linspace(cy, cy2, 240)])

dy0 = rgb_full_scaled.shape[0]/2
dx0 = rgb_full_scaled.shape[1]/2

rgbcmz = np.array(PIL.Image.open('gc_fullres_6.jpg'))[::-1,:,:]
wwcmz = WCS(fits.Header.fromtextfile('gc_fullres_6.wcs'))

aces7m = fits.open('/orange/adamginsburg/ACES/mosaics/7m_continuum_mosaic.fits')
aces7mwcs = WCS(aces7m[0].header)
aces7m[0].data[aces7m[0].data < -0.0005] = 0

aces12m = fits.open('/orange/adamginsburg/ACES/mosaics/12m_continuum_mosaic.fits')
aces12mwcs = WCS(aces12m[0].header)
aces12m[0].data[aces12m[0].data < -0.0005] = 0

sgrb2_269 = fits.open('/orange/adamginsburg/sgrb2/2013.1.00269.S/continuum/SgrB2_selfcal_full_TCTE7m_try2_selfcal6_ampphase_deeper_mask1.5mJy.image.tt0.pbcor.fits')
sgrb2_269wcs = WCS(sgrb2_269[0].header)

atlasgal = fits.open('/orange/adamginsburg/galactic_plane_surveys/atlasgal/MOSAICS/apex_planck.fits')
lm20,_ = map(int, WCS(atlasgal[0].header).world_to_pixel(SkyCoord(-20*u.deg, 0*u.deg, frame='galactic')))
lp20,_ = map(int, WCS(atlasgal[0].header).world_to_pixel(SkyCoord( 20*u.deg, 0*u.deg, frame='galactic')))
agaldata = atlasgal[0].data[:, lp20:lm20]
agalwcs = WCS(atlasgal[0].header)[:, lp20:lm20]
print(f"ATLASGAL from {atlasgal[0].data.shape} -> {agaldata.shape}")


cmap = pl.cm.Oranges
orange_transparent = cmap(np.arange(cmap.N))
orange_transparent[:,-1] = np.linspace(0, 1, cmap.N)
orange_transparent = ListedColormap(orange_transparent)

transorange = pl.cm.Oranges.copy()
transorange.set_under((0,0,0,0))


def animate(n, nframes=nframes, start=0, fig=None):
    ax = fig.gca()
    n0 = n
    n = start + n0
    if n == 40 or n0 == 0 and start > 40 and start < 240:
        print("Triggered n>=40")
        ax.imshow(agaldata,
          norm=simple_norm(agaldata, min_percent=5, max_percent=99.9, stretch='asinh'),
          transform=ax.get_transform(agalwcs),
          cmap=orange_transparent,
                 zorder=5)
    if n == 120 or n0 == 0 and start > 120 and start < 300:
        print("Triggered n>=120")
        ax.imshow(rgbcmz,
              transform=ax.get_transform(wwcmz),
                  zorder=120,
                 )

    if n == 180 or n0 == 0 and start > 180 and start < 300:
        print("Triggered n>=180")
        ax.imshow(aces7m[0].data,
                  norm=simple_norm(aces7m[0].data, stretch='log',
                                   max_percent=99.96, min_percent=1),
                  transform=ax.get_transform(aces7mwcs),
                  cmap=orange_transparent,
                  zorder=180,
                 )

    if n == 240 or n0 == 0 and start > 240:
        print("Triggered n>=240")
        ax.imshow(aces12m[0].data,
                     norm=simple_norm(aces12m[0].data, stretch='log',
                                      min_percent=None, max_percent=None,
                                      min_cut=-0.0005, max_cut=0.05,),
                     cmap='gray',
                  transform=ax.get_transform(aces12mwcs),
                  zorder=240
                    )
        aces12m[0].data[aces12m[0].data < 0.05] = np.nan
        ax.imshow(aces12m[0].data,
                     norm=simple_norm(aces12m[0].data, stretch='asinh',
                                      min_percent=None, max_percent=99.99,
                                      min_cut=0.05,),
                     cmap=transorange,
                  zorder=241,
                  transform=ax.get_transform(aces12mwcs), )

        ax.imshow(sgrb2_269[0].data,
                     norm=simple_norm(sgrb2_269[0].data, stretch='log',
                                      min_percent=None, max_percent=None,
                                      min_cut=-0.0005, max_cut=0.05,),
                     cmap='gray',
                  transform=ax.get_transform(sgrb2_269wcs),
                  zorder=242,
                    )
        sgrb2_269[0].data[sgrb2_269[0].data < 0.05] = np.nan
        ax.imshow(sgrb2_269[0].data,
                     norm=simple_norm(sgrb2_269[0].data, stretch='asinh',
                                      min_percent=None, max_percent=99.99,
                                      min_cut=0.05,),
                     cmap=transorange,
                  zorder=243,
                  transform=ax.get_transform(sgrb2_269wcs), )


    dy = dy0 * zoomfac[n]
    dx = dx0 * zoomfac[n]
    cx, cy = cxs[n], cys[n]

    ax.axis((cx-dx, cx+dx, cy-dy, cy+dy))

    print(f'{n}|', end='', flush=True)

    return fig,




if __name__ == "__main__":

    fig = pl.figure(figsize=(10,5), dpi=200, frameon=False)
    ax = pl.subplot(1,1,1, projection=WCS(target_header),
                    frame_class=EllipticalFrame)
    print("Showing full all-sky image")
    ax.imshow(rgb_full_scaled)
    ax.axis('off')


    if False:
        print("Beginning animation steps")
        anim_seg1 = functools.partial(animate, start=0, fig=fig)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg1, frames=nframes, repeat_delay=5000,
                                       interval=50, cache_frame_data=False)
        anim.save('zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment1.gif')

        print("Starting segment 2")
        anim_seg2 = functools.partial(animate, start=60, fig=fig)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg2, frames=nframes, repeat_delay=5000,
                                       interval=50, cache_frame_data=False)
        anim.save('zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment2.gif')

        print("Starting segment 3")
        anim_seg3 = functools.partial(animate, start=120, fig=fig)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg3, frames=nframes, repeat_delay=5000,
                                       interval=50, cache_frame_data=False)
        anim.save('zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment3.gif')

        print("Starting segment 4")
        anim_seg4 = functools.partial(animate, start=180, fig=fig)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg4, frames=nframes, repeat_delay=5000,
                                       interval=50, cache_frame_data=False)
        anim.save('zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment4.gif')

        print("Starting segment 5")
        anim_seg5 = functools.partial(animate, start=240, fig=fig)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg5, frames=nframes, repeat_delay=5000,
                                       interval=50, cache_frame_data=False)
        anim.save('zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment5.gif')

#    if True:
        print("Starting segment 6 afresh")

        fig = pl.figure(figsize=(10,5), dpi=200, frameon=False)
        ax = pl.subplot(1,1,1, projection=WCS(target_header),
                        frame_class=EllipticalFrame)

        anim_seg6 = functools.partial(animate, start=300, fig=fig)
        nframes = 240
        anim = animation.FuncAnimation(fig, anim_seg6, frames=nframes, repeat_delay=5000,
                                       interval=50, cache_frame_data=False)
        anim.save('zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment6.gif')

        # old version
        #anim = animation.FuncAnimation(fig, animate, frames=nframes, repeat_delay=5000,
        #                               fargs={'start': 240},
        #                               interval=50)
        #anim.save('zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35.gif')
