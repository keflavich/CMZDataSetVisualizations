# pan across the CMZ
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
import matplotlib
import matplotlib as mp



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
        path = matplotlib.path.Path([[-0.5,-0.5], [w-0.5,-0.5], [w-0.5,h-0.5], [-0.5,h-0.5], [-0.5,-0.5]])
        im.set_clip_path(path, transform=kwargs.get('transform'))



nframes = 540
# zoom for 300 frames, then pan
zoomfac = np.hstack([np.ones(540, dtype='float')/8.])


sgrb2 = SkyCoord.from_name('Sgr B2')
sgra = SkyCoord.from_name('Sgr A')
sgrc = SkyCoord.from_name('Sgr C')
sgrc = SkyCoord(359.49154*u.deg, -0.09789*u.deg, frame='galactic')

aces12m = fits.open('/orange/adamginsburg/ACES/mosaics/12m_continuum_mosaic.fits')
ww = aces12mwcs = WCS(aces12m[0].header)
aces12m[0].data[aces12m[0].data < -0.0005] = 0


# pan
cx, cy = ww.world_to_pixel(sgrb2)
cx2, cy2 = ww.world_to_pixel(sgrc)
cxs = np.linspace(cx, cx2, nframes)
cys = np.linspace(cy, cy2, nframes)

dy0 = aces12m[0].data.shape[0]/2
dx0 = aces12m[0].data.shape[1]/2


sgrb2_269 = fits.open('/orange/adamginsburg/sgrb2/2013.1.00269.S/continuum/SgrB2_selfcal_full_TCTE7m_try2_selfcal6_ampphase_deeper_mask1.5mJy.image.tt0.pbcor.fits')
sgrb2_269wcs = WCS(sgrb2_269[0].header)


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

    # fixed_imshow(ax, sgrb2_269[0].data,
    #              norm=simple_norm(sgrb2_269[0].data, stretch='log',
    #                               min_percent=None, max_percent=None,
    #                               min_cut=-0.0005, max_cut=0.001,),
    #              cmap='gray',
    #           transform=ax.get_transform(sgrb2_269wcs),
    #           zorder=242,
    #             )
    # sgrb2_269[0].data[sgrb2_269[0].data < 0.05] = np.nan
    # fixed_imshow(ax, sgrb2_269[0].data,
    #              norm=simple_norm(sgrb2_269[0].data, stretch='asinh',
    #                               min_percent=None, max_percent=99.999,
    #                               min_cut=0.001,),
    #              cmap=transorange,
    #           zorder=243,
    #           transform=ax.get_transform(sgrb2_269wcs), )


    dy = dy0 * zoomfac[n]
    dx = dx0 * zoomfac[n]
    dx = 750
    dy = 750
    cx, cy = cxs[n], cys[n]

    ax.axis((cx-dx, cx+dx, cy-dy, cy+dy))

    print(f'{n}|', end='', flush=True)

    return fig,




if __name__ == "__main__":

    fig = pl.figure(figsize=(10,10), dpi=200, frameon=False)
    # https://stackoverflow.com/a/15883620/814354
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None,
                        hspace=None)
    ax = pl.subplot(1,1,1, projection=ww,
                    frame_class=EllipticalFrame)
    print("Showing full all-sky image")

    aces12m = fits.open('/orange/adamginsburg/ACES/mosaics/12m_continuum_mosaic.fits')
    cut = 0.002
    fixed_imshow(ax, aces12m[0].data,
                 norm=simple_norm(aces12m[0].data, stretch='log',
                                  min_percent=None, max_percent=None,
                                  min_cut=-0.0005, max_cut=cut*5,),
                 cmap='gray',
              transform=ax.get_transform(aces12mwcs),
              zorder=240
                )
    aces12m[0].data[aces12m[0].data < cut] = np.nan
    fixed_imshow(ax, aces12m[0].data,
                 norm=simple_norm(aces12m[0].data, stretch='log',
                                  min_percent=None, max_percent=99.995,
                                  min_cut=cut,),
                 cmap=transorange,
              zorder=241,
              transform=ax.get_transform(aces12mwcs), )


    ax.axis('off')


    if True:
        print("Beginning animation steps")
        anim_seg1 = functools.partial(animate, start=0, fig=fig)
        nframes = nframes
        anim = animation.FuncAnimation(fig, anim_seg1, frames=nframes, repeat_delay=5000,
                                       interval=50, cache_frame_data=False)
        anim.save('pan_anim_ACES.gif')
        anim.save('pan_anim_ACES.mp4')

    #if True:
    #    anim_seg1 = functools.partial(animate, start=0, fig=fig)(0)
