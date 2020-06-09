from __future__ import division, print_function

import os
import yaml
from time import time
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.spatial import cKDTree
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma
from functools import wraps
from collections import namedtuple
from sfdmap import SFDMap
from .lsststruct import LsstStruct
from .log import logger


__all__ = ['hsc_mag_zpt', 'project_dir', 'ext_coeff', 'measure_psf_fwhm', 
           'get_logger', 'sky_cone', 'check_random_state', 
           'embed_slices', 'imfit_subimage_string', 'check_kwargs_defaults', 
           'mag_to_flux', 'mkdir_if_needed', 'check_run_dir', 'pixel_xmatch', 
           'dustmap', 'load_config', 'hsc_pixscale', 'load_hsc_lsbg_cat', 
           'get_hsc_lsbg_params', 'func_timer']


hsc_pixscale = 0.168
hsc_mag_zpt = 27.0
project_dir = os.path.dirname(os.path.dirname(__file__))
dustmap = SFDMap()
mag_to_flux= lambda mag_: 10**(0.4 * (hsc_mag_zpt - mag_))

# Extinction correction factor for HSC
# A_lambda = Coeff * E(B-V)
ExtCoeff = namedtuple('ExtCoeff', 'g r i z y')
ext_coeff = ExtCoeff(g=3.233, r=2.291, i=1.635, z=1.261, y=1.076)


def measure_psf_fwhm(psf, fwhm_guess):
    """
    Fit a 2D Gaussian to observed psf to estimate the FWHM

    Parameters
    ----------
    psf : ndarray
        Observed psf
    fwhm_guess : float
        Guess for fwhm in pixels

    Returns
    -------
    mean_fwhm : float
        Mean x & y FWHM 
    """
    sigma = fwhm_guess * gaussian_fwhm_to_sigma
    g_init = models.Gaussian2D(psf.max()*0.3, 
                               psf.shape[1]/2, 
			       psf.shape[0]/2, 
                               sigma)
    fit_g = fitting.LevMarLSQFitter()
    xx, yy = np.meshgrid(np.arange(psf.shape[1]), np.arange(psf.shape[0]))
    best_fit = fit_g(g_init, xx, yy, psf)
    mean_fwhm = np.mean([best_fit.x_fwhm, best_fit.y_fwhm])
    return mean_fwhm


def get_logger(level='info'):
    logger.setLevel(level.upper())
    return logger


def sky_cone(ra_c, dec_c, theta, steps=50, include_center=True):
    """
    Get ra and dec coordinates of a cone on the sky.
    
    Parameters
    ----------
    ra_c, dec_c: float
        Center of cone in degrees.
    theta: astropy Quantity, float, or int
        Angular radius of cone. Must be in arcsec
        if not a Quantity object.
    steps: int, optional
        Number of steps in the cone.
    include_center: bool, optional
        If True, include center point in cone.
    
    Returns
    -------
    ra, dec: ndarry
        Coordinates of cone.
    """
    from spherical_geometry.polygon import SphericalPolygon
    if type(theta)==float or type(theta)==int:
        theta = theta*u.arcsec
    cone = SphericalPolygon.from_cone(
        ra_c, dec_c, theta.to('deg').value, steps=steps)
    ra, dec = list(cone.to_lonlat())[0]
    ra = np.mod(ra - 360., 360.0)
    if include_center:
        ra = np.concatenate([ra, [ra_c]])
        dec = np.concatenate([dec, [dec_c]])
    return ra, dec


def check_random_state(seed):
    """
    Turn seed into a `numpy.random.RandomState` instance.
    Parameters
    ----------
    seed : `None`, int, list of ints, or `numpy.random.RandomState`
        If ``seed`` is `None`, return the `~numpy.random.RandomState`
        singleton used by ``numpy.random``.  If ``seed`` is an `int`,
        return a new `~numpy.random.RandomState` instance seeded with
        ``seed``.  If ``seed`` is already a `~numpy.random.RandomState`,
        return it.  Otherwise raise ``ValueError``.
    Returns
    -------
    random_state : `numpy.random.RandomState`
        RandomState object.
    Notes
    -----
    This routine is adapted from scikit-learn.  See
    http://scikit-learn.org/stable/developers/utilities.html#validation-tools.
    """
    import numbers

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if type(seed)==list:
        if type(seed[0])==int:
            return np.random.RandomState(seed)

    raise ValueError('{0!r} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def embed_slices(center, arr_shape, img_shape):
    """
    Get slices to embed smaller array into larger image.
    Parameters
    ----------
    center : ndarray
        Center of array as row, column.
    arr_shape : tuple 
        Shape of the array to embed (dimensions must be odd).
    img_shape : tuple
        Shape of the main image array.  
    Returns
    -------
    img_slice, arr_slice : tuples of slices
        Slicing indices. To embed array in image, 
        use the following:
        img[img_slice] = arr[arr_slice]
    """
    arr_shape = np.asarray(arr_shape)
    img_shape = np.asarray(img_shape)

    assert np.alltrue(arr_shape%2 != np.array([0,0]))

    imin = center - arr_shape//2
    imax = center + arr_shape//2 

    amin = ((imin < np.array([0,0]))*(-imin)).astype(int)
    amax = (arr_shape*(imax<=img_shape-1) +\
           (arr_shape-(imax-(img_shape-1)))*(imax>img_shape-1)).astype(int)

    imin = np.maximum(imin, np.array([0, 0])).astype(int)
    imax = np.minimum(imax, np.array(img_shape)-1).astype(int)
    imax += 1

    img_slice = np.s_[imin[0]:imax[0], imin[1]:imax[1]]
    arr_slice = np.s_[amin[0]:amax[0], amin[1]:amax[1]]

    return img_slice, arr_slice


def imfit_subimage_string(size, shape):
    assert size % 2 != 0, 'size must be odd'
    nrows, ncols = shape
    hsize = size//2  + 1
    slices = [nrows//2-hsize+1, 
              nrows//2+hsize+1, 
              ncols//2-hsize+1, 
              ncols//2+hsize+1]
    section = '[{}:{},{}:{}]'.format(*slices)
    return section 


def check_kwargs_defaults(kwargs, defaults):
    """ 
    Build keyword argument by changing a default set of parameters.
    Parameters
    ----------
    kwargs : dict
        Keyword arguments that are different for default values.
    defaults : dict
        The default parameter values.
    Returns
    -------
    kw : dict
        A new keyword argument.
    """
    kw = defaults.copy()
    for k, v in kwargs.items():
        kw[k] = v 
    return kw


def mkdir_if_needed(directory):
    """"
    Create directory if it does not exist.
    """
    if not os.path.isdir(directory):
        logger.info(f'creating directory {directory}')
        os.mkdir(directory)


def check_run_dir(run_dir):
    """
    Change working directory to run directory if necessary.
    """
    if run_dir != '.':
        logger.info('changing directory to ' + run_dir)
        os.chdir(run_dir)


def pixel_xmatch(cat_1, cat_2, max_sep=3, cols_1=['x_fit', 'y_fit'], 
                 cols_2=['x_fit', 'y_fit']):
    """
    Crossmatch catalogs.
    """
    coords_1 = cat_1[cols_1].to_pandas().values
    coords_2 = cat_2[cols_2].to_pandas().values
    kdt = cKDTree(coords_1)
    dist, idx = kdt.query(coords_2)
    match_2_mask = dist < max_sep
    match_1_idx = idx[match_2_mask]
    return match_1_idx, match_2_mask 


def load_config(config_fn):
    with open(config_fn, 'r') as fn:
        config = yaml.load(fn,  Loader=yaml.FullLoader)
    return config


catalog_path = os.path.join(project_dir, 'data')
def load_hsc_lsbg_cat(data_path=catalog_path,  use_pandas=True):
    fn = os.path.join(data_path, 'hsc-lsbg-final-cat.csv')
    return pd.read_csv(fn) if use_pandas else Table.read(fn)


_params = ['ra', 'dec', 'r_e', 'ell', 'PA', 'n', 'g-i', 'g-r']
def get_hsc_lsbg_params(cat_id, data_path=catalog_path, params=_params):
    cat = load_hsc_lsbg_cat(data_path, use_pandas=False)
    parameters= LsstStruct()
    for p in params:
        p_key = p.replace('-', '_')
        parameters[p_key] = cat[cat_id-1][p]
    return parameters


def func_timer(f):
    """
    A function decorator to time how long it takes to execute. The time is
    printed as INFO using the logger.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        dt = end - start
        if dt > 120:
            dt /= 60
            unit = 'min'
        else:
            unit = 'sec'
        logger.info('{} completed in {:.2f} {}'.format(f.__name__, dt, unit))
        return result
    return wrapper
