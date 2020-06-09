from __future__ import division, print_function

import os
import requests
from getpass import getpass
from io import BytesIO

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb


__all__ = ['HSCSession']


class HSCSession(object):

    def __init__(self, user, password=None, 
                 base_url='https://hsc-release.mtk.nao.ac.jp/'):
        self.session = requests.Session()
        self.base_url = base_url
        if password is None:
            password = getpass('Enter password: ')
        self.session.auth = (user, password)

    def fetch_hsc_cutout(self, ra, dec, width=2.0, height=None, 
                         band='R', imageonly=True, dr=2.1):
        """
        Fetch HSC cutout image at the given ra, dec

        Parameters
        ----------
        ra, dec : float
            in degrees
        width, height : float
            in arcseconds
        band : string of characters
            HSC band names, GRIZY
        imageonly : bool
            return images only not the entire fits hdus
        dr : float or str
            Data release 
        """

        if height is None:
            height = width

        band = band.upper()
        images = []
        for oneband in band:
            url = (os.path.join(self.base_url, 'das_quarry/')+\
                   "dr%s/cgi-bin/quarryImage?"
                   "ra=%.6f&dec=%.6f&sw=%.6fasec&sh=%.6fasec"
                   "&type=coadd&image=on&mask=on&variance=on&"
                   "filter=HSC-%s&tract=&rerun=" %\
                   (str(dr), ra, dec, width/2.0, height/2.0, oneband))
            resp = self.session.get(url)
            if resp.ok:
                images.append(fits.open(BytesIO(resp.content)))
        if imageonly:
            images = np.dstack([hdu[1].data for hdu in images])
        return images

    def fetch_psf(self, ra, dec, band='i', rerun='s18a_wide'):
        """
        Fetch psf at give ra & dec
        """
        num = {'s16a_wide':'4', 's17a_wide':'5', 's18a_wide':'6'}[rerun]
        url = self.base_url+'psf/'+num+'/cgi/getpsf?ra={:.6f}&'
        url += 'dec={:.6f}&filter={}&rerun={}&tract='
        url += 'auto&patch=auto&type=coadd'
        url = url.format(ra, dec, band, rerun)
        resp = self.session.get(url)
        return fits.getdata(BytesIO(resp.content))
 
    def make_rgb_image(self, ra=None, dec=None, width=2.0, height=None, 
                       band='irg', stretch=5, Q=8, images=None):
        """
        Make RGB image.

        Parameters
        ----------
        ra, dec : float
            in degrees
        width, height : float
            in arcseconds
        band : string of characters
            HSC band names for in RGB order
        stretch : float
            Linear stretch of HSC RGB image
        Q : float
            The asinh softening parameter for HSC RGB image
        images : ndarray
            If not None, will make rgb image using these images

        Returns
        -------
        rgb : ndarry
            The RGB image
        """
        if images is None:
            images = self.fetch_hsc_cutout(ra, dec, width, height, band)
        rgb = make_lupton_rgb(images[:, :, 0], images[:, :, 1],
                              images[:, :, 2], stretch=stretch, Q=Q)
        return rgb
