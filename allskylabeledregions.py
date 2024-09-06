"""
All-sky labeled locations
"""
import pylab as pl
import numpy as np
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

from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u, constants
from astropy.convolution import convolve_fft, Gaussian2DKernel, Gaussian1DKernel
import healpy
import PIL
import pyavm
import regions

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


def fix_nans(img):
    kernel = Gaussian2DKernel(2)
    sm = convolve_fft(img, kernel)
    img[np.isnan(img)] = sm[np.isnan(img)]
    return img

def h_rot(rgb, rot):
    hsv = rgb_to_hsv(rgb)
    hsv[:,:,0] += rot  # 0.25 = 90/360
    hsv[:,:,0] = hsv[:,:,0] % 1 
    rgb_scaled = hsv_to_rgb(hsv)
    return rgb_scaled

def make_rgb(rgb, basename, hsv_rotation=0, layernames='RGB',
             do_layers=True,
             header=target_header,
             axlims=None, frame_class=EllipticalFrame):
    plt.figure(figsize=(10,5), dpi=200)
    ax = plt.subplot(1,1,1, projection=WCS(header),
                     frame_class=frame_class)
    #ax.coords.grid(color='white')
    try:
        ax.coords['glat'].set_ticklabel(visible=False)
        ax.coords['glon'].set_ticklabel(visible=False)
        ax.coords['glat'].set_ticks_visible(False)
        ax.coords['glon'].set_ticks_visible(False)
    except Exception:
        ax.coords['ra'].set_ticklabel(visible=False)
        ax.coords['dec'].set_ticklabel(visible=False)
        ax.coords['ra'].set_ticks_visible(False)
        ax.coords['dec'].set_ticks_visible(False)
    if do_layers:
        for ind, color in zip((0, 1, 2), (layernames)):
            rgbc = np.zeros_like(rgb)
            rgbc[:, :, ind] = rgb[:, :, ind]
            if hsv_rotation != 0:
                rgbc = h_rot(rgbc, hsv_rotation)
            ax.imshow(rgbc)
            pl.savefig(f'{basename}_{color}.png', bbox_inches='tight', transparent=True)
    if hsv_rotation != 0:
        rgb = h_rot(rgb, hsv_rotation)
    ax.imshow(rgb)
    if axlims is not None:
        ax.axis(axlims)
    pl.savefig(f'{basename}_RGB.png', bbox_inches='tight', transparent=True)
    return ax



fn_HI4PI = 'NHI_HPX.fits'
if not os.path.exists(fn_HI4PI):
    HI4PI = download_file(f"https://lambda.gsfc.nasa.gov/data/foregrounds/HI4PI/{fn_HI4PI}")
    shutil.move(HI4PI, fn_HI4PI)
HI4PI = fits.open(fn_HI4PI)

img_HI4PI, footprint = reproject_from_healpix(fn_HI4PI, target_header, field=5)


fn_CO21 = "COM_CompMap_CO21-commander_2048_R2.00.fits"
if not os.path.exists(fn_CO21):
    planck_CO21 = download_file(f"https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/{fn_CO21}")
    shutil.move(planck_CO21, fn_CO21)
planck_CO21 = fits.open(fn_CO21)

img_CO21, footprint = reproject_from_healpix(fn_CO21, target_header)
img_CO21 = fix_nans(img_CO21)

fn_ThermalDust = "COM_CompMap_ThermalDust-commander_2048_R2.00.fits"
if not os.path.exists(fn_ThermalDust):
    planck_ThermalDust = download_file(f"https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/{fn_ThermalDust}")
    shutil.move(planck_ThermalDust, fn_ThermalDust)
planck_ThermalDust = fits.open(fn_ThermalDust)

img_ThermalDust, footprint = reproject_from_healpix(fn_ThermalDust, target_header)


# different colors, overwrites previous
rgb = np.array([simple_norm(img_HI4PI,       min_percent=0.01, max_percent=99.99, log_a=2e1, stretch='log')(img_HI4PI),
                simple_norm(img_CO21,        min_percent=0.01, max_percent=99.90, log_a=2e1, stretch='log')(img_CO21),
                simple_norm(img_ThermalDust, min_percent=0.01, max_percent=99.90, log_a=5e2, stretch='log')(img_ThermalDust)]).T.swapaxes(0,1)
ax = make_rgb(rgb, 'HI-CO-Dust_m0.35', -0.35, layernames=('HI', 'CO', 'Dust'), do_layers=False)
rect = regions.RectangleSkyRegion(
    coordinates.SkyCoord(0*u.deg, 0*u.deg, frame='galactic'),
    width=np.abs((ax.get_xlim()[1] - ax.get_xlim()[0])*target_header['CDELT1'])*u.deg,
    height=(ax.get_ylim()[1] - ax.get_ylim()[0])*target_header['CDELT2']*u.deg,
)
prect = rect.to_pixel(ax.wcs)
prect.visual['edgecolor'] = 'w'
innergalplot = prect.plot(ax=ax)
pl.gcf().set_facecolor('none')
pl.savefig("HI-CO-Dust_m0.35_RGB_innergal_zoombox.png", bbox_inches='tight')

innergaltext = ax.text(0, 25, "Galactic Plane", transform=ax.get_transform('world'),
        color='w', horizontalalignment='center')

cmzrect = regions.RectangleSkyRegion(
    coordinates.SkyCoord(0*u.deg, 0*u.deg, frame='galactic'),
    width=12*u.deg,
    height=4.8*u.deg,
)
pcmzrect = cmzrect.to_pixel(ax.wcs)
pcmzrect.visual['edgecolor'] = 'w'
cmzplot = pcmzrect.plot(ax=ax)

cmztext = ax.text(0, 5, "Central Molecular Zone", transform=ax.get_transform('world'),
        color='w', horizontalalignment='center')


orionrect = regions.RectangleSkyRegion(
    coordinates.SkyCoord(209*u.deg, -19*u.deg, frame='galactic'),
    width=15*u.deg,
    height=15*u.deg,
)
porionrect = orionrect.to_pixel(ax.wcs)
porionrect.visual['edgecolor'] = 'w'
oriplot = porionrect.plot(ax=ax)
oritext = ax.text(209, -7, "Orion", transform=ax.get_transform('world'),
        color='w', horizontalalignment='center')

pl.savefig("HI-CO-Dust_m0.35_RGB_labeled_zones.png", bbox_inches='tight')

innergalplot.set_visible(False)
cmzplot.set_visible(False)
innergaltext.set_visible(False)
cmztext.set_visible(False)

pl.savefig("HI-CO-Dust_m0.35_RGB_labeled_orion.png", bbox_inches='tight')

oritext.set_visible(False)
oriplot.set_visible(False)

m31 = coordinates.SkyCoord.from_name('M31')
andromedarect = regions.RectangleSkyRegion(
    m31,
    width=5*u.deg,
    height=5*u.deg,
)
pandromedarect = andromedarect.to_pixel(ax.wcs)
pandromedarect.visual['edgecolor'] = 'w'
andromedaplot = pandromedarect.plot(ax=ax)
andromedatext = ax.text(m31.galactic.l.deg, m31.galactic.b.deg+4,
                        "Andromeda", transform=ax.get_transform('world'),
        color='w', horizontalalignment='center')

pl.savefig("HI-CO-Dust_m0.35_RGB_labeled_andromeda.png", bbox_inches='tight')

andromedatext.set_visible(False)
andromedaplot.set_visible(False)


taurus = coordinates.SkyCoord.from_name('taurus')
taurusrect = regions.RectangleSkyRegion(
    taurus,
    width=15*u.deg,
    height=15*u.deg,
)
ptaurusrect = taurusrect.to_pixel(ax.wcs)
ptaurusrect.visual['edgecolor'] = 'w'
taurusplot = ptaurusrect.plot(ax=ax)
taurustext = ax.text(taurus.galactic.l.deg-5, taurus.galactic.b.deg+10,
                        "Taurus", transform=ax.get_transform('world'),
        color='w', horizontalalignment='center')

pl.savefig("HI-CO-Dust_m0.35_RGB_labeled_taurus.png", bbox_inches='tight')

taurustext.set_visible(False)
taurusplot.set_visible(False)

W51 = coordinates.SkyCoord.from_name('W51')
W51rect = regions.RectangleSkyRegion(
    W51,
    width=15*u.deg,
    height=15*u.deg,
)
pW51rect = W51rect.to_pixel(ax.wcs)
pW51rect.visual['edgecolor'] = 'w'
W51plot = pW51rect.plot(ax=ax)
W51text = ax.text(W51.galactic.l.deg-5, W51.galactic.b.deg+10,
                        "W51", transform=ax.get_transform('world'),
        color='w', horizontalalignment='center')

pl.savefig("HI-CO-Dust_m0.35_RGB_labeled_W51.png", bbox_inches='tight')

W51text.set_visible(False)
W51plot.set_visible(False)


for clustername in ('NGC 3603', 'Westerlund 1', 'Westlerlund 2'):

    cluster = coordinates.SkyCoord.from_name(clustername)
    clusterrect = regions.RectangleSkyRegion(
        cluster,
        width=5*u.deg,
        height=5*u.deg,
    )
    pclusterrect = clusterrect.to_pixel(ax.wcs)
    pclusterrect.visual['edgecolor'] = 'w'
    clusterplot = pclusterrect.plot(ax=ax)
    clustertext = ax.text(cluster.galactic.l.deg-5, cluster.galactic.b.deg+10,
                          clustername, transform=ax.get_transform('world'),
            color='w', horizontalalignment='center')

    pl.savefig(f"HI-CO-Dust_m0.35_RGB_labeled_{clustername.replace(' ', '')}.png", bbox_inches='tight')

    clustertext.set_visible(False)
    clusterplot.set_visible(False)


smacs0723 = coordinates.SkyCoord('07h 23m 19.5s −73d 27m 15.6s', unit=(u.h, u.deg), frame='fk5')
smacs0723rect = regions.RectangleSkyRegion(
    smacs0723,
    width=1*u.deg,
    height=1*u.deg,
)
psmacs0723rect = smacs0723rect.to_pixel(ax.wcs)
psmacs0723rect.visual['edgecolor'] = 'w'
smacs0723plot = psmacs0723rect.plot(ax=ax)
smacs0723text = ax.text(smacs0723.galactic.l.deg-5, smacs0723.galactic.b.deg+10,
                        "SMACS0723", transform=ax.get_transform('world'),
        color='w', horizontalalignment='center')

pl.savefig("HI-CO-Dust_m0.35_RGB_labeled_smacs0723.png", bbox_inches='tight')

smacs0723text.set_visible(False)
smacs0723plot.set_visible(False)



saltdiskcoords = {
    'W33': coordinates.SkyCoord.from_name('W33'),
    'NGC6334': coordinates.SkyCoord.from_name('NGC6334'),
    'G17': coordinates.SkyCoord(17.64*u.deg, 0.16*u.deg, frame='galactic'),
    'G351': coordinates.SkyCoord(351.77*u.deg, -0.51*u.deg, frame='galactic'),
    'I16547': coordinates.SkyCoord.from_name('IRAS 16547-4247')
}
elts = []
for ii, (name, coord) in enumerate(saltdiskcoords.items()):
    saltdisksrect = regions.RectangleSkyRegion(
        coord,
        width=1*u.deg,
        height=1*u.deg,
    )
    psaltdisksrect = saltdisksrect.to_pixel(ax.wcs)
    psaltdisksrect.visual['edgecolor'] = 'r'
    saltdisksplot = psaltdisksrect.plot(ax=ax)
    saltdiskstext = ax.text(coord.galactic.l.deg-5, coord.galactic.b.deg+5 + 5*ii,
                            name, transform=ax.get_transform('world'),
            color='w', horizontalalignment='center')
    elts.append(saltdisksplot)
    elts.append(saltdiskstext)

pl.savefig("HI-CO-Dust_m0.35_RGB_labeled_saltdisks.png", bbox_inches='tight')

#for elt in elts:
#    elt.set_visible(False)