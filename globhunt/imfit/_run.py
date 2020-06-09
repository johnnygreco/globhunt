import os
import numpy as np
from astropy.io import fits
import pymfit
from ..sersic import Sersic
from ..utils import check_kwargs_defaults
from .. import logger


mask_kws_default = dict(thresh=1.2, kern_sig=1.0, backsize=10, 
                        obj_rmin=4, grow_sig=3.0, use_hsc_mask=False)


def run(img_fn, var_fn=None, psf_fn=None, io_dir='.', run_label='imfit', 
        mask_fn='mask.fits', mask_kws={}, init_params={}, 
        ell_free=True, save_viz=False, make_mask=False, pa_free=True):

    mask_fn = os.path.join(io_dir, mask_fn)

    if make_mask:
        logger.debug('making source mask')
        mask_kws = check_kwargs_defaults(mask_kws, mask_kws_default)
        mask_kws['out_fn'] = mask_fn
        pymfit.make_mask(img_fn, **mask_kws)

    _init = init_params.copy()
    config = pymfit.sersic_config(_init, img_shape=img_fn)

    if ell_free:
        config['ell'] = _init.pop('ell', 0.1)
    if pa_free:
        config['PA'] = _init.pop('PA', 0.0)

    logger.info('running imfit')
    result = pymfit.run(
        img_fn, config=config, mask_fn=mask_fn, 
	config_fn=os.path.join(io_dir, 'config-'+run_label+'.txt'),
	var_fn=var_fn if var_fn else None, psf_fn=psf_fn,
	out_fn=os.path.join(io_dir, 'best-fit-'+run_label+'.dat')
    )

    if result['ell'] < 0:
        logger.debug('ell < 0: flipping so that ell > 0')
        a = (1.0 - result['ell']) * result['r_e']
        b = result['r_e']
        result['ell'] = 1.0 - b/a
        result['r_e'] = a
        result['PA'] -= 90.0
    if (result['PA'] > 180) or (result['PA'] < 0):
        msg = 'wrapping PA = {:.2f} within 0 < PA < 180'.format(result['PA'])
        logger.debug(msg)
        wrapped = result['PA'] % 360.0
        wrapped = wrapped - 180 * (wrapped > 180)
        result['PA'] = wrapped

    sersic = Sersic(result)

    if save_viz:
        figsize = (16, 7)
        save_fn = os.path.join(io_dir, 'viz-' + run_label + '.png')
        pymfit.viz.img_mod_res(img_fn, result, mask_fn, show=False, 
                               save_fn=save_fn, psf_fn=psf_fn, 
                               figsize=figsize)

    return sersic
