# zoom into the CMZ _even more_
import subprocess
import numpy as np
import time
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

rgbcmz = np.array(PIL.Image.open('gc_fullres_6.jpg'))[::-1,:,:]
wwcmz = WCS(fits.Header.fromtextfile('gc_fullres_6.wcs'))

cmap = pl.cm.Oranges
orange_transparent = cmap(np.arange(cmap.N))
orange_transparent[:,-1] = np.linspace(0, 1, cmap.N)
orange_transparent = ListedColormap(orange_transparent)

transorange = pl.cm.Oranges.copy()
transorange.set_under((0,0,0,0))

aces7m = fits.open('/orange/adamginsburg/ACES/mosaics/7m_continuum_mosaic.fits')
aces7mwcs = WCS(aces7m[0].header)
aces7m[0].data[aces7m[0].data < -0.0005] = 0

aces12m = fits.open('/orange/adamginsburg/ACES/mosaics/12m_continuum_mosaic.fits')
aces12mwcs = WCS(aces12m[0].header)
aces12m[0].data[aces12m[0].data < -0.0005] = 0

sgrb2m = SkyCoord.from_name('Sgr B2 Main')
sgrb2m = SkyCoord('17:47:20.174 -28:23:04.81', frame='icrs', unit=(u.h, u.deg))
sgrb2n = SkyCoord.from_name('NAME Sgr B2 (North)')
sgrb2n = SkyCoord('17:47:19.886 -28:22:18.20', frame='icrs', unit=(u.h, u.deg))
sgrb2s = SkyCoord.from_name('NAME Sgr B2 (South)')
print("Found Sgr B2 N, M, and S")

sgrb2_269 = fits.open('/orange/adamginsburg/sgrb2/2013.1.00269.S/continuum/SgrB2_selfcal_full_TCTE7m_try2_selfcal6_ampphase_deeper_mask1.5mJy.image.tt0.pbcor.fits')
sgrb2_269wcs = WCS(sgrb2_269[0].header)

#fh4 = fits.open('/orange/adamginsburg/sgrb2/NB/NB.sgr_b2.M.B3.cont.pb0.1.r0.5.clean500k0.1mjy.pcal2.image.tt0.pbcor.fits')
#fh3 = fits.open('/orange/adamginsburg/sgrb2/2017.1.00114.S/imaging_results/Sgr_B2_DS_B6_uid___A001_X1290_X46_continuum_merged_12M_robust0_selfcal4_finaliter.image.tt0.pbcor.fits')
#fh2 = fits.open('/orange/adamginsburg/sgrb2/NB/NB.sgr_b2.N.B3.cont.pb0.1.r0.5.clean500k0.1mjy.pcal2.image.tt0.pbcor.fits')
#fh4 = fits.open('/orange/adamginsburg/sgrb2/NB/final_imaging/sgr_b2.M.B3.cont.r0.5.1m0.05mJy.cal3.short_spacing_model.image.tt0.pbcor.fits')
#fh2 = fits.open('/orange/adamginsburg/sgrb2/NB/final_imaging/sgr_b2.N.B3.cont.r0.5.1m0.05mJy.cal3.short_spacing_model.image.tt0.pbcor.fits')
fh2= fits.open('/orange/adamginsburg/sgrb2/2016.1.00550.S/FITS/SgrB2_B3_NM_mosaic_withshortspacing.fits')



# make it square
dy0 = np.max(rgbcmz.shape)/2
dx0 = np.max(rgbcmz.shape)/2

print("Done opening files")

t0 = [time.time()]

def animate(n, nframes=0, start=0, fig=None, zoomfac=None, cxs=None, cys=None, verbose=False):
    ax = fig.gca()
    n0 = n
    n = start + n0
    if verbose:
        print(f"Start={start} n={n} n0={n0}")

    if n0 == 0:
        print(f"Triggered n0=0 (n={n}, n0={n0}, start={start})")
        ax.cla()
        fixed_imshow(ax, rgbcmz, zorder=1)
        ax.axis('off')


    if (n == 40 and start <= 40) or n0 == 0 and start > 40:
        print(f"Triggered n=40 (n={n}, n0={n0}, start={start})")
        fixed_imshow(ax, aces7m[0].data,
                  norm=simple_norm(aces7m[0].data, stretch='log',
                                   max_percent=99.96, min_percent=1),
                  transform=ax.get_transform(aces7mwcs),
                  cmap=orange_transparent,
                  zorder=40
                 )

    if (n == 100 and start <= 100) or n0 == 0 and start > 100:
        print(f"Triggered n=100 (n={n}, n0={n0}, start={start})")
        # fixed_imshow(ax, aces12m[0].data,
        #              norm=simple_norm(aces12m[0].data, stretch='log',
        #                               min_percent=None, max_percent=None,
        #                               min_cut=-0.0005, max_cut=0.05,),
        #              cmap='gray',
        #           transform=ax.get_transform(aces12mwcs),
        #           zorder=80
        #             )
        # aces12m[0].data[aces12m[0].data < 0.05] = np.nan
        # fixed_imshow(ax, aces12m[0].data,
        #              norm=simple_norm(aces12m[0].data, stretch='asinh',
        #                               min_percent=None, max_percent=99.99,
        #                               min_cut=0.05,),
        #              cmap=transorange,
        #           zorder=81,
        #           transform=ax.get_transform(aces12mwcs), )

        sgrb2_269 = fits.open('/orange/adamginsburg/sgrb2/2013.1.00269.S/continuum/SgrB2_selfcal_full_TCTE7m_try2_selfcal6_ampphase_deeper_mask1.5mJy.image.tt0.pbcor.fits')

        cut = 0.002
        fixed_imshow(ax, sgrb2_269[0].data,
                     norm=simple_norm(sgrb2_269[0].data, stretch='log',
                                      min_percent=None, max_percent=None,
                                      min_cut=-cut, max_cut=cut*2,),
                     cmap='gray',
                  transform=ax.get_transform(sgrb2_269wcs),
                  zorder=82,
                    )
        sgrb2_269[0].data[sgrb2_269[0].data < cut] = np.nan
        fixed_imshow(ax, sgrb2_269[0].data,
                     norm=simple_norm(sgrb2_269[0].data, stretch='log',
                                      min_percent=None, max_percent=99.99,
                                      min_cut=cut,),
                     cmap=transorange,
                  zorder=83,
                  transform=ax.get_transform(sgrb2_269wcs), )

    if (n == 150 and start <= 150) or n0 == 0 and start > 150:
        print(f"Triggered n=150 (n={n}, n0={n0}, start={start})")
        #fixed_imshow(ax, fh4[0].data,
        #             norm=simple_norm(fh4[0].data, stretch='log',
        #                              min_percent=None, max_percent=None,
        #                              min_cut=-0.0005, max_cut=0.05,),
        #             cmap='gray',
        #          transform=ax.get_transform(WCS(fh4[0].header)),
        #          zorder=120,
        #            )
        #fh4[0].data[fh4[0].data < 0.05] = np.nan
        #fixed_imshow(ax, fh4[0].data,
        #             norm=simple_norm(fh4[0].data, stretch='asinh',
        #                              min_percent=None, max_percent=99.99,
        #                              min_cut=0.05,),
        #             cmap=transorange,
        #          zorder=121,
        #          transform=ax.get_transform(WCS(fh4[0].header)), )
        fh2= fits.open('/orange/adamginsburg/sgrb2/2016.1.00550.S/FITS/SgrB2_B3_NM_mosaic_withshortspacing.fits')

        cut = 0.0015
        fixed_imshow(ax, fh2[0].data,
                     norm=simple_norm(fh2[0].data, stretch='log',
                                      min_percent=None, max_percent=None,
                                      min_cut=-0.0005, max_cut=cut*2,),
                     cmap='gray',
                  zorder=122,
                  transform=ax.get_transform(WCS(fh2[0].header)),
                    )
        fh2[0].data[fh2[0].data < cut] = np.nan
        fixed_imshow(ax, fh2[0].data,
                     norm=simple_norm(fh2[0].data, stretch='log',
                                      min_percent=None, max_percent=99.99,
                                      min_cut=cut,),
                     cmap=transorange,
                  zorder=123,
                  transform=ax.get_transform(WCS(fh2[0].header)), )
        # fixed_imshow(ax, fh3[0].data.squeeze(),
        #              norm=simple_norm(fh3[0].data.squeeze(), stretch='log',
        #                               min_percent=None, max_percent=None,
        #                               min_cut=-0.0005, max_cut=0.05,),
        #              cmap='gray',
        #           zorder=124,
        #           transform=ax.get_transform(WCS(fh3[0].header).celestial),
        #             )
        # fh3[0].data[fh3[0].data < 0.05] = np.nan
        # fixed_imshow(ax, fh3[0].data.squeeze(),
        #              norm=simple_norm(fh3[0].data.squeeze(), stretch='asinh',
        #                               min_percent=None, max_percent=99.99,
        #                               min_cut=0.05,),
        #              cmap=transorange,
        #           zorder=125,
        #           transform=ax.get_transform(WCS(fh3[0].header).celestial), )




    dy = dy0 * zoomfac[n]
    dx = dx0 * zoomfac[n]
    cx, cy = cxs[n], cys[n]

    ax.axis((cx-dx, cx+dx, cy-dy, cy+dy))

    print(f'{n}:{time.time()-t0[-1]:0.1f}|', end='', flush=True)
    t0.append(time.time())

    return fig,




if __name__ == "__main__":
    nframes = 420
    zoomfac = np.hstack([np.geomspace(1, 1/200., 240),
                         np.ones(180, dtype='float')/200.])
    cx, cy = wwcmz.world_to_pixel(sgrb2n)
    cx2, cy2 = wwcmz.world_to_pixel(sgrb2m)
    cxs = np.hstack([np.ones(240)*cx, np.linspace(cx, cx2, 180)])
    cys = np.hstack([np.ones(240)*cy, np.linspace(cy, cy2, 180)])


    print("Making figure")
    fig = pl.figure(figsize=(10, 10), dpi=200, frameon=False)
    # https://stackoverflow.com/a/15883620/814354
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    ax = pl.subplot(1,1,1, projection=wwcmz)
    fixed_imshow(ax, rgbcmz, zorder=1)
    ax.axis('off')

    print("Beginning animation steps")

    animname = 'cmz_to_sgrb2_zoomier_square'
    if True:

        anim_seg1 = functools.partial(animate, start=0, fig=fig, zoomfac=zoomfac, cxs=cxs, cys=cys)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg1, frames=nframes, repeat_delay=5000,
                                       interval=50)
        anim.save(f'{animname}_segment1.gif')

    #if False:
        print("Starting segment 2")
        anim_seg2 = functools.partial(animate, start=60, fig=fig, zoomfac=zoomfac, cxs=cxs, cys=cys)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg2, frames=nframes, repeat_delay=5000,
                                       interval=50)
        anim.save(f'{animname}_segment2.gif')

    #if False:
        print("Starting segment 3")
        anim_seg3 = functools.partial(animate, start=120, fig=fig, zoomfac=zoomfac, cxs=cxs, cys=cys)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg3, frames=nframes, repeat_delay=5000,
                                       interval=50)
        anim.save(f'{animname}_segment3.gif')

        print("Starting segment 4")
        anim_seg4 = functools.partial(animate, start=180, fig=fig, zoomfac=zoomfac, cxs=cxs, cys=cys)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg4, frames=nframes, repeat_delay=5000,
                                       interval=50)
        anim.save(f'{animname}_segment4.gif')

        print("Starting segment 5")
        anim_seg5 = functools.partial(animate, start=240, fig=fig, zoomfac=zoomfac, cxs=cxs, cys=cys)
        nframes = 60
        anim = animation.FuncAnimation(fig, anim_seg5, frames=nframes, repeat_delay=5000,
                                       interval=50)
        anim.save(f'{animname}_segment5.gif')

        print("Starting segment 6")
        anim_seg6 = functools.partial(animate, start=300, fig=fig, zoomfac=zoomfac, cxs=cxs, cys=cys)
        nframes = 120
        anim = animation.FuncAnimation(fig, anim_seg6, frames=nframes, repeat_delay=5000,
                                       interval=50)
        anim.save(f'{animname}_segment6.gif')

        subprocess.check_call("""
convert cmz_to_sgrb2_zoomier_square_segment[0-9].gif \( -clone 0 -set delay 200 \) \( -clone 1-239 \) \( +clone -set delay 200 \) \( -clone 241-419 \) -delete 0-419  \( +clone -set delay 500 \) +swap +delete \( -clone 1--1 -reverse \) -loop 0 cmz_to_sgrb2_zoomier_combined_square.gif
""".split())
