from __future__ import division, print_function

import os
import abc
import six

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table

from .stamp_collector import HSCSession
from .utils import get_logger, project_dir
from .image import MultiBandImage, calc_image_stats
from .lsststruct import LsstStruct


__all__ = ['TigerButler', 'HSCButler']


@six.add_metaclass(abc.ABCMeta)
class BaseButler(object):
    """
    Butler abstract base class.
    """

    def __init__(self, log_level='info'):
        self.log_level = log_level
        self.logger = get_logger(log_level)

    @abc.abstractmethod
    def fetch_cutout(self, ra, dec, width, bands):
        raise NotImplementedError()


class HSCButler(BaseButler):
    """
    HSC image class that is independent of the LSST stack.
    """

    _DEFAULT_URL = 'https://hscdata.mtk.nao.ac.jp/'

    def __init__(self, username, password=None, base_url=None, **kwargs):

        super(HSCButler, self).__init__(**kwargs)
        self.base_url = base_url if base_url else self._DEFAULT_URL
        self.session = HSCSession(username, password, self.base_url)

    def fetch_cutout(self, ra, dec, width, bands='gri', dr=2.1, 
                     rerun='s18a_wide', calc_stats=False, **kwargs):
        """
        Fetch cutout from Mitaka database.

        Parameters
        ----------
        ra, dec : float 
            Source coordinates in degrees.
        width : astropy Quantity or float  
            Width of cutout. Must be in arcsec if not 
            an astropy Quantity.
        bands : str, optional
            Photometric bands to fetch.
        dr : float or str
            Data release
        calc_stats : bool 
            If True, calculate the image statistics 

        Returns
        -------
        mbi : MultiBandImage
            Multi-band image object.
        """
        if type(width) != u.Quantity:
            width *= u.arcsec

        mbi = MultiBandImage(bands=bands, 
                             zpt=27.0, 
                             pixscale=0.168, 
                             log_level=self.log_level)

        for band in bands:
            self.logger.debug('Fetching '+band+'-band cutout image')
            hdu = self.session.fetch_hsc_cutout(
                ra, dec, band=band, width=width.to('arcsec').value, 
                imageonly=False, dr=dr)[0]
            if band == bands[0]:
                mbi.wcs = WCS(hdu[1].header)
            mbi.header[band] = hdu[1].header
            mbi.image[band] = hdu[1].data
            mbi.hsc_mask[band] = hdu[2].data
            mbi.var[band] = hdu[3].data
            self.logger.debug('Fetching '+band+'-band psf')
            mbi.psf[band] = self.session.fetch_psf(
                ra, dec, band=band, rerun=rerun)
            if calc_stats:
                mbi.stats[band] = calc_image_stats(hdu[1].data, **kwargs)

        return mbi
