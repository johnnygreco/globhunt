
import os, sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from astropy.table import Table, hstack
from .utils import mag_to_flux, check_random_state, pixel_xmatch
from . import logger
from . import project_dir
data_dir = os.path.join(project_dir, 'data')


__all__ = ['GlobColors', 'random_gc_mags', 'grid_positions', 
           'make_star_image', 'recovery', 'rgb_images', 'inject']


def _get_blom_gc_data():
    table = Table.read(os.path.join(data_dir, 'blom-2012-gcs.fits'))
    table.rename_column('zmag', 'z_mag')
    mask = np.ones(len(table), dtype=bool)
    for b in 'griz':
        mask &= ~np.isnan(table[b+'_mag'])
    table = table[mask]
    return table


class GlobColors(object):

    _gc_color_data= dict(
        blom = _get_blom_gc_data
    )

    def __init__(self, ref='blom'):
        self.ref = ref
        self.data = self._gc_color_data[ref]()

    def get_color(self, color):
        c1 = color.split('-')[0]
        c2 = color.split('-')[1]
        return self.data[c1 + '_mag'] - self.data[c2 + '_mag']

    def __call__(self, color):
        return self.get_color(color)


def random_gc_mags(mag_low=20, mag_high=27, nstars=100, bands='griz', 
                   color_cat='blom', ref_band='g', random_state=None, 
                   **kwargs):

    rng = check_random_state(random_state)
    observed_colors = GlobColors(color_cat)

    if ref_band not in bands:
        logger.exception(ref_band + ' is the reference band!')
        sys.exit(1)

    mags = Table()
    mags[ref_band] = rng.uniform(mag_low, mag_high, nstars)
    mags[ref_band].unit = 'mag'
    idx = np.arange(len(observed_colors('g-i')))
    rand_idx = rng.choice(idx, size=nstars, replace=True)

    for b in bands:
        if b != ref_band:
            gc_colors = observed_colors(ref_band + '-' + b)
            mags[b] = mags[ref_band] - gc_colors[rand_idx]

    return mags


def grid_positions(image_shape=(501, 501), nstars=100, edge_buffer=20, 
                   max_rand_shift=5, random_state=None, randomness=True, 
                   **kwargs):

    assert edge_buffer > 0, 'edge buffer must be greater than 0'
    
    if randomness:
        rng = check_random_state(random_state)
        edge = rng.randint(edge_buffer, edge_buffer + 10)
        shift = rng.randint(0, max_rand_shift, size=2)
    else:
        edge = edge_buffer
        shift = 0
        
    ydim, xdim = image_shape[0] - edge, image_shape[1] - edge
    sep = np.sqrt((xdim * ydim)/nstars)

    xsep = np.ceil(xdim/sep).astype(int)
    ysep = np.ceil(ydim/sep).astype(int)
    x_grid = np.linspace(edge, xdim, xsep)
    y_grid = np.linspace(edge, ydim, ysep)

    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    xy_grid = np.vstack((x_grid.ravel(), y_grid.ravel())).T - shift

    return xy_grid.astype(int)


def random_positions(image_shape=(501, 501), nstars=100, edge_buffer=20, 
                     random_state=None, **kwargs):

     rng = check_random_state(random_state)
     ydim, xdim = image_shape[0] - edge_buffer, image_shape[1] - edge_buffer
     x = rng.randint(edge_buffer, image_shape[1] - edge_buffer, nstars)
     y = rng.randint(edge_buffer, image_shape[0] - edge_buffer, nstars)
     xy_grid = np.vstack([x, y]).T

     return xy_grid


def make_star_catalog(nstars, position_type, random_state=None, **kwargs):
    rng = check_random_state(random_state)

    if position_type == 'grid':
        xy_grid = grid_positions(nstars=nstars, random_state=rng, **kwargs)
    elif position_type == 'random':
        xy_grid = random_positions(nstars=nstars, random_state=rng,**kwargs)
    else:
        logger.exception('invalid position type')
        sys.exit(1)
    mags = random_gc_mags(nstars=len(xy_grid), random_state=rng, **kwargs)
    cat = Table([xy_grid[:, 0], xy_grid[:, 1]], names=['x', 'y'])
    cat = hstack([cat, mags])
    return cat


def make_star_image(xy, mags, image_shape, psf):
    if type(xy)==Table:
        positions = np.array([xy['y'], xy['x']])
    else:
        positions = np.array([xy[:, 1], xy[:, 0]])
    ravel_idx = np.ravel_multi_index(positions, image_shape)
    count = np.bincount(ravel_idx, weights=mag_to_flux(mags))
    pad_width = (0, image_shape[0]*image_shape[1] - len(count))
    star_image = np.pad(count, pad_width, 'constant').reshape(*image_shape)

    psf /= psf.sum()
    star_image = signal.fftconvolve(star_image, psf, 'same')

    return star_image


def recovery(cat, synth_cat, max_sep=1):
    recover_idx, inject_mask = pixel_xmatch(cat, synth_cat,
                                            cols_1=['x_fit', 'y_fit'],
                                            cols_2=['x', 'y'],
                                            max_sep=max_sep)
    injected = synth_cat[inject_mask].copy()
    recovered = cat[recover_idx].copy()
    return injected, recovered


def rgb_images(mbi, mbi_fakes, phot_pipe, stretch=0.5, figsize=(18, 6),
               titles=True, save_fn=None):

    fig, ax = plt.subplots(1, 3, figsize=figsize,
                           subplot_kw=dict(xticks=[], yticks=[]))

    fig.subplots_adjust(wspace=0.01)
    ax[0].imshow(mbi.make_rgb(stretch=stretch), origin='lower')
    ax[1].imshow(mbi_fakes.make_rgb(stretch=stretch), origin='lower')
    ax[2].imshow(phot_pipe.make_rgb('residual_image_forced', stretch=stretch),
                 origin='lower')

    if titles:
        ax[0].set_title('Original Image', fontsize=25, y=1.01)
        ax[1].set_title('Image with Synths', fontsize=25, y=1.01)
        ax[2].set_title('Residual Image', fontsize=25, y=1.01)

    if save_fn is not None:
        fig.savefig(save_fn)

    return fig, ax


def inject(mbi, nstars, mag_range=[20, 28], ref_band='i',
           bands='griz', position_type='random', **kwargs):

    logger.info('***** injecting synthetic stars into image *****')
    logger.info(ref_band + '-band is set as the synth color reference')

    mbi_fakes = mbi.copy()
    image_shape = mbi.shape[1:]

    kws = dict(image_shape=image_shape,
               ref_band=ref_band,
               mag_low=mag_range[0],
               mag_high=mag_range[1])

    kws.update(kwargs)

    cat = make_star_catalog(nstars, position_type, **kws)

    for b in bands:
        star_image = make_star_image(cat['x', 'y'],
                                     cat[b],
                                     image_shape,
                                     mbi_fakes.psf[b])
        mbi_fakes.image[b] += star_image

    return cat, mbi_fakes
