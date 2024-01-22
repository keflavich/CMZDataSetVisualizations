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

import matplotlib.colors as mcolors
import numpy as np
from astropy.visualization import simple_norm

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb 

from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u, constants
from astropy.convolution import convolve_fft, Gaussian2DKernel, Gaussian1DKernel
import healpy
import PIL
import pyavm
import regions

import regions
from astropy import coordinates
from astropy import units as u, constants
import astropy.visualization.wcsaxes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import pylab as pl
import pylab as plt

from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector, Path
from astropy import wcs
from matplotlib.transforms import Bbox
from matplotlib.patches import Polygon, PathPatch, ConnectionPatch

def mark_inset_generic(axins, parent_ax, data, loc1=1, loc2=3,
                       bl=None, br=None, tl=None, tr=None,
                       loc1in=None, loc2in=None, edgecolor='b', zorder1=100, zorder2=100, polyzorder=1):
    if bl is None:
        bl = axins.wcs.pixel_to_world(0, 0)
    if br is None:
        br = axins.wcs.pixel_to_world(data.shape[1], 0)
    if tl is None:
        tl = axins.wcs.pixel_to_world(0, data.shape[0]) # x,y not y,x
    if tr is None:
        tr = axins.wcs.pixel_to_world(data.shape[1], data.shape[0]) # x,y not y,x
    
    fig = parent_ax.get_figure()
    
    #frame = parent_ax.wcs.wcs.radesys.lower()
    #frame = ax.wcs.world_axis_physical_types[0].split(".")[1]
    frame = wcs.utils.wcs_to_celestial_frame(ax.wcs)
    
    blt = bl.transform_to(frame)
    brt = br.transform_to(frame)
    tlt = tl.transform_to(frame)
    trt = tr.transform_to(frame)
    xys = [parent_ax.wcs.wcs_world2pix([[crd.spherical.lon.deg,
                                         crd.spherical.lat.deg]],0)[0]
           for crd in (trt, tlt, blt, brt, trt)]
    
    markinkwargs = dict(fc='none', ec=edgecolor)
    ppoly = Polygon(xys, fill=False, zorder=polyzorder, **markinkwargs)
    parent_ax.add_patch(ppoly)
    
    corners = parent_ax.transData.inverted().transform(xys)
    
    axcorners = [(1,1), (0,1), (0,0), (1,0)]
    corners = [(crd.spherical.lon.deg, crd.spherical.lat.deg)
               for crd in (trt, tlt, blt, brt)]
    
    if loc1in is None:
        loc1in = loc1
    if loc2in is None:
        loc2in = loc2
    
    con1 = ConnectionPatch(xyA=corners[loc1-1], coordsA=axins.get_transform('world'), axesA=axins,
                           xyB=corners[loc1in-1], coordsB=parent_ax.get_transform('world'), axesB=parent_ax,
                           linestyle='-', color=edgecolor, zorder=zorder1)
    con2 = ConnectionPatch(xyA=corners[loc2-1], coordsA=axins.get_transform('world'), axesA=axins,
                           xyB=corners[loc2in-1], coordsB=parent_ax.get_transform('world'), axesB=parent_ax,
                           linestyle='-', color=edgecolor, zorder=zorder2)    
    fig.add_artist(con1)
    fig.add_artist(con2)
    
    return con1, con2, ppoly

aces12m = fits.open('/orange/adamginsburg/ACES/mosaics/continuum/12m_continuum_mosaic.fits')
aces12mwcs = WCS(aces12m[0].header)
#aces12m[0].data[aces12m[0].data < -0.0005] = 0

fig = plt.figure(figsize=(10,5), frameon=False)
ax = plt.subplot(1,1,1, projection=aces12mwcs)


colors1 = pl.cm.gray_r(np.linspace(0., 1, 128))
colors2 = pl.cm.inferno(np.linspace(0, 1, 128))

colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)


ax.coords['glat'].set_ticklabel(visible=False)
ax.coords['glon'].set_ticklabel(visible=False)
ax.coords['glat'].set_ticks_visible(False)
ax.coords['glon'].set_ticks_visible(False)
ax.axis('off')


### Region Setup
sgrb2m = regions.CircleSkyRegion(
    coordinates.SkyCoord(000.6672*u.deg,  -00.03640*u.deg, frame='galactic'),
    radius=25*u.arcsec)
sgrb2n = regions.CircleSkyRegion(
    coordinates.SkyCoord(000.6773*u.deg,  -00.029*u.deg, frame='galactic'),
    radius=25*u.arcsec)
sgrb2ds = regions.RectangleSkyRegion(
    coordinates.SkyCoord('17:47:21.0856811259 -28:24:48.9796510714', frame='icrs', unit=(u.h, u.deg)),
    width=96*u.arcsec,
    height=2.77*u.arcmin,
    #angle=58*u.deg,
)
brick = regions.RectangleSkyRegion(
    coordinates.SkyCoord(0.2499811648, 0.0187129377, frame='galactic', unit=(u.deg, u.deg)).fk5,
    width=2.46*u.arcmin,
    height=5.9*u.arcmin,
    #angle=58*u.deg
)
sgrb2m.visual['edgecolor'] = 'r'
sgrb2n.visual['edgecolor'] = 'r'
sgrb2ds.visual['edgecolor'] = 'r'
brick.visual['edgecolor'] = 'r'

psgrb2m = sgrb2m.to_pixel(ax.wcs)
msgrb2m = psgrb2m.to_mask()
slcs_sgrb2m,_ = msgrb2m.get_overlap_slices(aces12m[0].data.shape)

psgrb2n = sgrb2n.to_pixel(ax.wcs)
psgrb2n.visual['edgecolor'] = 'r'
msgrb2n = psgrb2n.to_mask()
slcs_sgrb2n,_ = msgrb2n.get_overlap_slices(aces12m[0].data.shape)

merge = psgrb2n | psgrb2m
merge_mask = merge.to_mask()
slcs_merge,_ = merge_mask.get_overlap_slices(aces12m[0].data.shape)

dsmask = sgrb2ds.to_pixel(ax.wcs).to_mask()
slcs_ds,_ = dsmask.get_overlap_slices(aces12m[0].data.shape)
brick_mask = brick.to_pixel(ax.wcs).to_mask()
slcs_brick,_ = brick_mask.get_overlap_slices(aces12m[0].data.shape)


sgrb2ds_tl = SkyCoord(sgrb2ds.center.ra + sgrb2ds.width / 2 / np.cos(sgrb2ds.center.dec), sgrb2ds.center.dec + sgrb2ds.height / 2, frame='icrs')
sgrb2ds_tr = SkyCoord(sgrb2ds.center.ra - sgrb2ds.width / 2 / np.cos(sgrb2ds.center.dec), sgrb2ds.center.dec + sgrb2ds.height / 2, frame='icrs')
sgrb2ds_bl = SkyCoord(sgrb2ds.center.ra + sgrb2ds.width / 2 / np.cos(sgrb2ds.center.dec), sgrb2ds.center.dec - sgrb2ds.height / 2, frame='icrs')
sgrb2ds_br = SkyCoord(sgrb2ds.center.ra - sgrb2ds.width / 2 / np.cos(sgrb2ds.center.dec), sgrb2ds.center.dec - sgrb2ds.height / 2, frame='icrs')

brick_tl = SkyCoord(brick.center.ra + brick.width / 2 / np.cos(brick.center.dec), brick.center.dec + brick.height / 2, frame='icrs')
brick_tr = SkyCoord(brick.center.ra - brick.width / 2 / np.cos(brick.center.dec), brick.center.dec + brick.height / 2, frame='icrs')
brick_bl = SkyCoord(brick.center.ra + brick.width / 2 / np.cos(brick.center.dec), brick.center.dec - brick.height / 2, frame='icrs')
brick_br = SkyCoord(brick.center.ra - brick.width / 2 / np.cos(brick.center.dec), brick.center.dec - brick.height / 2, frame='icrs')




# Basic Image Setup
ax.imshow(aces12m[0].data,
             norm=simple_norm(aces12m[0].data, stretch='log',
                              min_percent=None, max_percent=None,
                              min_cut=-0.0005, max_cut=0.09,),
             cmap=mymap,
            )



bbox = [0.01, 0, 0.2, 1.05]

# during debugging, False'd this for Sgr B2, which behaved fine
if True:

    point_sgrb2m = psgrb2m.plot(ax=ax)
    point_sgrb2n = psgrb2n.plot(ax=ax)

    axins1 = zoomed_inset_axes(ax, 30, loc='upper left', bbox_to_anchor=bbox,
                            bbox_transform=fig.transFigure,
                            axes_class=astropy.visualization.wcsaxes.core.WCSAxes,
                            axes_kwargs=dict(wcs=ax.wcs[slcs_sgrb2m])
                            )
    axins1.axis('off')
    axins1.imshow(msgrb2m.cutout(aces12m[0].data),
                norm=simple_norm(aces12m[0].data, stretch='log',
                                min_percent=None, max_percent=None,
                                min_cut=-0.0005, max_cut=0.5,),
                cmap=mymap,
                )
    #axins1.axis(msgrb2m.bbox.extent)
    mark1 = mark_inset_generic(axins1, ax, msgrb2m.cutout(aces12m[0].data), loc1=1, loc2=3, edgecolor='r')
    point_sgrb2m_in = sgrb2m.to_pixel(axins1.wcs).plot(ax=axins1)
    axins1.coords['glat'].set_ticklabel(visible=False)
    axins1.coords['glon'].set_ticklabel(visible=False)
    axins1.coords['glat'].set_ticks_visible(False)
    axins1.coords['glon'].set_ticks_visible(False)


    pl.savefig("aces_mark_sgrb2m.png", bbox_inches='tight', dpi=200)

    point_sgrb2m.set_visible(False)
    point_sgrb2m_in.set_visible(False)
    #axins1.set_visible(False)
    for pp in mark1:
        pp.set_visible(False)
    axins1.set_visible(False)

    axins2 = zoomed_inset_axes(ax, 30, loc='upper left', bbox_to_anchor=bbox, bbox_transform=fig.transFigure,
                            axes_class=astropy.visualization.wcsaxes.core.WCSAxes,
                            axes_kwargs=dict(wcs=ax.wcs[slcs_sgrb2n]))
    axins2.imshow(msgrb2n.cutout(aces12m[0].data),
                norm=simple_norm(aces12m[0].data, stretch='log',
                                min_percent=None, max_percent=None,
                                min_cut=-0.0005, max_cut=0.5,),
                cmap=mymap,
                )
    #axins1.axis(msgrb2n.bbox.extent)
    mark2 = mark_inset_generic(axins2, ax, msgrb2n.cutout(aces12m[0].data), loc1=1, loc2=3, edgecolor='r')
    point_sgrb2n_in = sgrb2n.to_pixel(axins2.wcs).plot(ax=axins2)
    axins2.coords['glat'].set_ticklabel(visible=False)
    axins2.coords['glon'].set_ticklabel(visible=False)
    axins2.coords['glat'].set_ticks_visible(False)
    axins2.coords['glon'].set_ticks_visible(False)

    pl.savefig("aces_mark_sgrb2n.png", bbox_inches='tight', dpi=200)

    point_sgrb2n.set_visible(False)
    point_sgrb2n_in.set_visible(False)
    for pp in mark2:
        pp.set_visible(False)
    axins2.set_visible(False)


    axins3 = zoomed_inset_axes(ax, 15, loc='upper left',
                            bbox_to_anchor=[0.05, 0, 0.2, 1], bbox_transform=fig.transFigure,
                            axes_class=astropy.visualization.wcsaxes.core.WCSAxes,
                            axes_kwargs=dict(wcs=ax.wcs[slcs_merge]))
    axins3.axis('off')
    axins3.imshow(merge_mask.multiply(aces12m[0].data),
                norm=simple_norm(aces12m[0].data, stretch='log',
                                min_percent=None, max_percent=None,
                                min_cut=-0.0005, max_cut=0.5,),
                cmap=mymap
                )
    #axins3.axis(merge_mask.bbox.extent)
    mark3 = mark_inset_generic(axins3, ax, merge_mask.cutout(aces12m[0].data), loc1=1, loc2=3, edgecolor='r')
    point_sgrb2n_in = sgrb2n.to_pixel(axins3.wcs).plot(ax=axins3)
    point_sgrb2m_in = sgrb2m.to_pixel(axins3.wcs).plot(ax=axins3)
    axins3.coords['glat'].set_ticklabel(visible=False)
    axins3.coords['glon'].set_ticklabel(visible=False)
    axins3.coords['glat'].set_ticks_visible(False)
    axins3.coords['glon'].set_ticks_visible(False)

    pl.savefig("aces_mark_sgrb2mANDn.png", bbox_inches='tight', dpi=200)


    point_sgrb2n.set_visible(False)
    point_sgrb2m.set_visible(False)
    point_sgrb2m_in.set_visible(False)
    point_sgrb2n_in.set_visible(False)
    for pp in mark3:
        pp.set_visible(False)
    axins3.set_visible(False)


axins4 = zoomed_inset_axes(ax, 8, loc='upper left',
                        bbox_to_anchor=[0.05, 0, 0.2, 1], bbox_transform=fig.transFigure,
                        axes_class=astropy.visualization.wcsaxes.core.WCSAxes,
                        axes_kwargs=dict(wcs=ax.wcs[slcs_ds]))
axins4.axis('off')
mul = dsmask.multiply(aces12m[0].data)
mul[mul==0] = np.nan
axins4.imshow(mul,
            norm=simple_norm(aces12m[0].data[slcs_ds], stretch='asinh',
                            min_percent=None, max_percent=None,
                            min_cut=-0.0008, max_cut=0.02,),
            cmap=mymap,
            zorder=150,
            )
mark4 = mark_inset_generic(axins4, ax, dsmask.cutout(aces12m[0].data), loc1=2, loc2=4, edgecolor='r',
bl=sgrb2ds_bl,
tr=sgrb2ds_tr,
br=sgrb2ds_br,
tl=sgrb2ds_tl,
)
point_sgrb2ds_in = sgrb2ds.to_pixel(axins4.wcs).plot(ax=axins4)
axins4.coords['glat'].set_ticklabel(visible=False)
axins4.coords['glon'].set_ticklabel(visible=False)
axins4.coords['glat'].set_ticks_visible(False)
axins4.coords['glon'].set_ticks_visible(False)

print("Sgr B2 DS done: saving")
pl.savefig("aces_mark_sgrb2ds.png", bbox_inches='tight', dpi=200)
print("Sgr B2 DS saved")




point_sgrb2ds_in.set_visible(False)
for pp in mark4:
    pp.set_visible(False)
axins4.set_visible(False)


axins5 = zoomed_inset_axes(ax, 5, loc='upper left',
                        bbox_to_anchor=[0.11, 0.04, 0.33, 1.05], bbox_transform=fig.transFigure,
                        axes_class=astropy.visualization.wcsaxes.core.WCSAxes,
                        axes_kwargs=dict(wcs=ax.wcs[slcs_brick]))
axins5.axis('off')
mul = brick_mask.multiply(aces12m[0].data)
mul[mul == 0] = np.nan
axins5.imshow(mul,
            norm=simple_norm(aces12m[0].data[slcs_brick], stretch='asinh',
                            min_percent=None, max_percent=None,
                            min_cut=-0.0005, max_cut=0.02,),
            cmap=mymap,
            zorder=150,
            )
#axins5.axis(brick_mask.bbox.extent)
mark5 = mark_inset_generic(axins5, ax, brick_mask.cutout(aces12m[0].data), loc1=4, loc2=3, edgecolor='r',
bl=brick_bl,
tr=brick_tr,
br=brick_br,
tl=brick_tl,
)
point_brick_in = brick.to_pixel(axins5.wcs).plot(ax=axins5)
axins5.coords['glat'].set_ticklabel(visible=False)
axins5.coords['glon'].set_ticklabel(visible=False)
axins5.coords['glat'].set_ticks_visible(False)
axins5.coords['glon'].set_ticks_visible(False)

print("Brick done: Saving")
pl.savefig("aces_mark_brick.png", bbox_inches='tight', dpi=200)
print("Brick done: Saved")