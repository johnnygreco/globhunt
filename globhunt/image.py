from __future__ import division, print_function

try:
    import cPickle as pickle
except:
    import pickle

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.visualization import make_lupton_rgb
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.nddata import Cutout2D
from copy import deepcopy
from .utils import get_logger, measure_psf_fwhm, hsc_mag_zpt, hsc_pixscale
from .lsststruct import LsstStruct
from . import logger


__all__ = ['calc_image_stats', 
           'make_file_dict', 
           'fetch_multiband_image', 
           'MultiBandImage']


def calc_image_stats(img, sigma=3.0, iters=5, **kwargs):
    mean, median, stdev = sigma_clipped_stats(
        img, sigma=sigma, iters=iters, **kwargs) 
    return LsstStruct(mean=mean, median=median, stdev=stdev)


def make_file_dict(src_num, bands, path, name_prefix='G'):
    name = f'{name_prefix}{src_num}'
    path = os.path.join(path, name)
    image = {b: os.path.join(path, f'img-{src_num}-{b}.fits') for b in bands}
    psf = {b: os.path.join(path, f'psf-{src_num}-{b}.fits') for b in bands}
    var = {b: os.path.join(path, f'var-{src_num}-{b}.fits') for b in bands}
    mask = {b: os.path.join(path, f'mask-{src_num}-{b}.fits') for b in bands}
    file_dict = dict(image=image, psf=psf, var=var, hsc_mask=mask)
    return file_dict


def fetch_multiband_image(src_num, path, bands, width=None, name_prefix='G'):
    file_dict = make_file_dict(src_num, path, bands, name_prefix)
    mbi = MultiBandImage.from_file_dict(file_dict, width=width)
    return mbi


class MultiBandImage(object):
    """
    Multi-band image class.
    """

    def __init__(self, bands, zpt=hsc_mag_zpt, pixscale=hsc_pixscale, 
                 **kwargs):

        self.bands = bands
        self.zpt = zpt
        self.pixscale = pixscale
        self.wcs = kwargs.pop('wcs', None)
        self.header= kwargs.pop('header', LsstStruct())
        self.image = kwargs.pop('image', LsstStruct())
        self.var = kwargs.pop('var', LsstStruct())
        self.hsc_mask = kwargs.pop('hsc_mask', LsstStruct())
        self.psf = kwargs.pop('psf', LsstStruct())
        self.psf_fwhm = kwargs.pop('psf_fwhm', LsstStruct())
        self.stats = kwargs.pop('stats', LsstStruct())

    @staticmethod
    def from_pickle(filename):
        pkl_file = open(filename, 'rb')
        mbi = pickle.load(pkl_file)
        pkl_file.close()
        return  mbi

    @staticmethod
    def from_file_dict(file_dict, width=None, calc_stats=False, **kwargs):
        """
        Load a MultiBandImage object from a file dictionary with the
        following structure:

        file_dict = dict(
            image={dict of fits image locations},
            var={dict of fits variance image locations},
            hsc_mask={dict of fits hsc mask image locations},
            psf={dict of fits psf image locations},
        ), 

        where the keys of the dictionaries are the bands ('grizy').
        This is useful when the images needed to be saved as single
        extension fits files.

        Parameters
        ----------
        file_dict : dictionary
            File dictionary with all the necessary files to build a
            MultiBandImage.
        width : astropy Quantity or float (optional)
            Width of cutout. Must be in arcsec if not
            an astropy Quantity. If None, the entire image will be used. 
            The source is assumed to be at the center of the image.
        calc_stats : bool
            If True, calculate the image statistics

        Returns
        -------
        mbi : MultiBandImage
            The multiband image object.
        """

        _file_dict = file_dict.copy()
        bands = list(_file_dict['image'].keys())

        mbi = MultiBandImage(bands=bands,
                             zpt=27.0,
                             pixscale=0.168)


        image_files = _file_dict.pop('image')
        mbi.image_fn = LsstStruct()

        for band in bands:
            hdu = fits.open(image_files[band])[0]
            if band == bands[0]:
                mbi.wcs = WCS(hdu.header)
            mbi.header[band] = hdu.header
            mbi.image[band] = hdu.data
            mbi.image_fn[band] = image_files[band]
            if calc_stats:
                mbi.stats[band] = calc_image_stats(hdu.data, **kwargs)

        for dtype in _file_dict.keys():
            setattr(mbi, dtype+'_fn', LsstStruct())
            for band in bands:
                hdu = fits.open(_file_dict[dtype][band])[0]
                mbi_attr = getattr(mbi, dtype)
                mbi_attr[band] = hdu.data
                mbi_attr = getattr(mbi, dtype+'_fn')
                mbi_attr[band] = _file_dict[dtype][band]

        if width is not None:
            if type(width) != u.Quantity:
                width *= u.arcsec
            width = width.to('arcsec').value / hsc_pixscale
            y_c, x_c = np.array(mbi.wcs.array_shape) / 2
            ra, dec = mbi.wcs.wcs_pix2world([[x_c, y_c]], 0)[0]
            for data_type in ['image', 'hsc_mask', 'var']:
                for b in bands:
                    data = getattr(mbi, data_type)[b]
                    cutout = Cutout2D(data, [x_c, y_c], width, wcs=mbi.wcs)
                    mbi_attr = getattr(mbi, data_type)
                    mbi_attr[b] = cutout.data
            mbi.wcs = cutout.wcs

        return mbi

    @property
    def shape(self):
        img_shape = self.image[self.bands[0]].shape
        return (len(self.bands), img_shape[0], img_shape[1])

    def measure_psf_fwhm(self, band, fwhm_guess=1.0/0.168):
        return measure_psf_fwhm(self.psf[band], fwhm_guess)

    def get_hsc_bright_object_mask(self, band):
        return self.hsc_mask[band].astype(int) & 512 != 0

    def make_rgb(self, rgb_bands='irg', stretch=0.8, Q=8):

        rgb = make_lupton_rgb(self.image[rgb_bands[0]],
                              self.image[rgb_bands[1]],
                              self.image[rgb_bands[2]],
                              stretch=stretch, Q=Q)
        return rgb

    def copy(self):
        return deepcopy(self)

    def write_fits_files(self, prefix='src', out_path='.', overwrite=True, 
                         bands='all', which=['image']):

        bands = self.bands if bands == 'all' else bands
        file_label = dict(image='img', var='var', psf='psf', variance='var')

        for b in bands:
            for kind in which:
                data = getattr(self, kind)[b]
                header = self.header[b] if kind != 'psf' else None
                fn = prefix + '-' + file_label[kind] + '-' + b + '.fits'
                fn = os.path.join(out_path, fn)
                logger.debug('writing ' + fn)
                fits.writeto(fn, data, header=header, overwrite=overwrite)

    def to_pickle(self, filename):
        pkl_file = open(filename, 'wb')
        pickle.dump(self, pkl_file)
        pkl_file.close()

    def __repr__(self):
        retstr = "{0}(bands={1}, zpt={2:.2f}, pixscale={3:.4f}, "\
                 "wcs={4}, image={5}, var={6}"\
                 "hsc_mask={7}, psf={8}, stats={9})"
        return retstr.format(self.__class__.__name__, self.bands, self.zpt, 
                             self.pixscale, self.wcs, self.image, self.var, 
                             self.hsc_mask, self.psf, self.stats)
