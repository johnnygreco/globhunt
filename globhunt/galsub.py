###############################################
# Code for removing host galaxy light.
###############################################
from __future__ import division, print_function

import numpy as np
from scipy.ndimage import median_filter
from astropy.convolution import convolve
from .sersic import Sersic
from .utils import embed_slices


__all__ = ['median_unsharp_mask', 
           'subtract_sersic']


def median_unsharp_mask(img, size, **kwargs):
    """
    Remove large-scale features (including the host galaxy)
    using a median filter unsharp mask

    Parameters
    ----------
    img : ndarray
        Galaxy image
    size : int
        Size of the median filter

    Returns
    -------
    residual : ndarray
        Residual image
    """
    img_medfil = median_filter(img, int(size), **kwargs)
    residual = img - img_medfil
    return residual


def subtract_sersic(img, I_e, r_e, n, ell, PA, X0, Y0, psf=None, 
		    subimage_shape=None, **kwargs):
    """
    Subtract Sersic model from image. Assumes imfit parameter names.

    Parameters
    ----------
    img : ndarray
        Galaxy image
    I_e, r_e, n, ell, PA, X0, Y0 : floats
        imfit Sersic params. (lengths should be in pixels)
    psf : ndimage (optional)
        If not None, psf to convolve with model
    subimage_shape : tuple
        If not None, generate model in small subimage. This is useful
        if the main image is large.

    Returns
    -------
    residual : ndarray
        Residual image
    """
    
    _x0, _y0 = np.array(subimage_shape)/2 if subimage_shape else (X0, Y0)
    params = dict(I_e=I_e, r_e=r_e, n=n, ell=ell, PA=PA, X0=_x0, Y0=_y0)
    sersic = Sersic(params)
    
    shape = subimage_shape if subimage_shape else img.shape
    
    if psf is not None:
        model = convolve(sersic.array(shape), psf)
    else:
        model = sersic.array(shape)
        
    if subimage_shape is not None:
        residual = img.copy()
        img_slice, arr_slice = embed_slices((Y0, X0), 
                                            subimage_shape, 
                                            residual.shape)
        residual[img_slice] = residual[img_slice] - model[arr_slice]
    else:
        residual = img - model
        
    return residual
