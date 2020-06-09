import os, glob
import warnings
import numpy as np 
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
from astropy.utils.exceptions import AstropyUserWarning

from photutils.detection import DAOStarFinder
from photutils.psf import DAOGroup
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.psf import FittableImageModel, BasicPSFPhotometry
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.psf import subtract_psf

import pymfit
from . import imfit, logger, LsstStruct
from .galsub import subtract_sersic
from .utils import check_kwargs_defaults, measure_psf_fwhm
from .utils import load_config, pixel_xmatch, ext_coeff, dustmap 


__all__ = ['SersicFitter', 'PSFPhotometry']


class SersicFitter(object):
    
    _default_mask_init = dict(
        thresh=1.2, kern_sig=1.0, backsize=8,
        obj_rmin=50, grow_sig=3.5, use_hsc_mask=False
    )
    
    _default_mask_residual = dict(
        thresh=1., kern_sig=1.0, backsize=8, grow_obj=4.5, 
        obj_rmin=0, grow_sig=4.0, use_hsc_mask=False
    )
                       
    def __init__(self, out_dir, mask_init_kws={}, mask_res_kws={}):
        
        self.sersic_model = None
        self.out_dir = out_dir
        self.mask_init_kws = check_kwargs_defaults(mask_init_kws, 
                                                   self._default_mask_init)
        self.mask_res_kws = check_kwargs_defaults(mask_res_kws, 
                                                  self._default_mask_residual)
        
    def _file_string(self, p, v, b):
        return os.path.join(self.out_dir, '{}-{}-{}.fits'.format(p, v, b))
    
    def make_masks(self, mbi, prefix, init_params={}, use_residuals=True, 
                   combine_masks=True, use_hsc_bright_mask=True):
            
            logger.info('making masks')
            if use_residuals:
                logger.info('will iterate fit using residuals')
                
            mask_list = []
            
            for band in self.bands:
                img_fn = self._file_string(prefix, 'img', band)
                mask_fn = self._file_string(prefix, 'msk', band)
                self.mask_init_kws['out_fn'] = mask_fn
                mask = pymfit.make_mask(mbi.image[band], **self.mask_init_kws)

                if use_hsc_bright_mask:
                    mask = (mask | mbi.get_hsc_bright_object_mask(band))

                if use_residuals:
                    var_fn = self._file_string(prefix, 'var', band)
                    psf_fn = self._file_string(prefix, 'psf', band)
                    sersic = imfit.run(img_fn, var_fn, psf_fn, self.out_dir,
                                       prefix + '-' + band, mask_fn, 
                                       init_params=init_params, save_viz=False)
                    res = subtract_sersic(mbi.image[band], 
                                          psf=mbi.psf[band], 
                                          **sersic.params)
                    mask_res = pymfit.make_mask(res, **self.mask_res_kws)
                    mask = (mask | mask_res).astype(int)

                mask_list.append(mask)
            
            if combine_masks:
                logger.info('taking union of masks')
                final_mask = np.zeros_like(mask_list[0], dtype=int)
                for mask in mask_list:
                    final_mask |= mask
                mask_fn = self._file_string(prefix, 'msk', 'union')
                fits.writeto(mask_fn, mask, overwrite=True)        

    def fit(self, mbi, label=None, init_params={}, save_viz=False, clean=True, 
            use_residuals=True, combine_masks=True, use_hsc_bright_mask=True):
        
        logger.info('***** fitting sersic function to galaxy *****')
        
        prefix = label if label else 'src'
        self.bands = mbi.bands
        mbi.write_fits_files(prefix, self.out_dir, 
                             which=['image', 'var', 'psf'])
        self.make_masks(mbi, prefix, init_params, use_residuals, 
                        combine_masks, use_hsc_bright_mask)
        
        self.sersic_model = LsstStruct()
        
        for band in self.bands:
            img_fn = self._file_string(prefix, 'img', band)
            var_fn = self._file_string(prefix, 'var', band)
            psf_fn = self._file_string(prefix, 'psf', band)
            
            if combine_masks:
                mask_fn = self._file_string(prefix, 'msk', 'union')
            else:
                mask_fn = self._file_string(prefix, 'msk', band)
                
            self.sersic_model[band] = imfit.run(img_fn, var_fn, psf_fn, 
                                                self.out_dir,
                                                prefix + '-' + band, mask_fn, 
                                                init_params=init_params, 
                                                save_viz=save_viz)           

        if clean:
            logger.info('deleting intermediate files in ' + self.out_dir)
            cmd = os.path.join(self.out_dir, '*' + prefix + '*')
            files = [fn for fn in glob.glob(cmd) if fn[-3:]!='png']
            for fn in files:
                os.remove(fn)

    def save_model(self, filename):
        logger.debug('saving sersic model to ' + filename)
        np.save(filename, [self.sersic_model])


class PSFPhotometry(object):
    
    def __init__(self, imfit_io, config_file_or_dict={}):

        if type(config_file_or_dict) == str:
            logger.info(f'Loading config file {config_file_or_dict}')
            config = load_config(config_file_or_dict)
        else:
            config = config_file_or_dict.copy()

        self.input_config = config.copy()
        self.imfit_io = config.pop('imfit_io', imfit_io)
        self.sersic_fitter = SersicFitter(out_dir=self.imfit_io)
        self.use_hsc_bright_mask = config.pop('use_hsc_bright_mask', 
                                              dict(phot=False, imfit=True))
        self.residual_image_forced = None
        self.residual_image = None
        
        # daofinder parameters
        self.threshold = config.pop('threshold', 3.0)
        self.daofinder_opt = dict(
            sigma_radius = config.pop('sigma_radius', 3.0),
            sharphi = config.pop('sharphi', 2.0),
            sharplo = config.pop('sharplo', 0.),
            roundlo = config.pop('roundlo', -1.0),
            roundhi = config.pop('roundhi', 1.0),
        )
                
        # daogroup parameter
        self.crit_separation = config.pop('crit_separation', 1.5)

        # TODO: make these bkgrd methods options
        self.bkg = MMMBackground()
        self.bkgrms = MADStdBackgroundRMS()

        # phot parameters
        self.aperture_radius = config.pop('aperture_radius', 1.0)
        self.phot_opts = dict(
            fitshape = config.pop('fitshape', (15, 15)),
            niters = config.pop('niters', 3),
            bkg_estimator = self.bkg,
        )

        self.master_band = config.pop('master_band', 'i')
        self.max_match_sep = config.pop('max_match_sep', 1.0)
        self.min_match_bands= config.pop('min_match_bands', 4)

    def _setup_psf(self):
        # build psf model from hsc pipeline psf image
        self.psf_model = LsstStruct()
        self.psf_fwhm = LsstStruct()
        for band in self.mbi.bands:
            self.psf_model[band] = FittableImageModel(self.mbi.psf[band])
            ybbox, xbbox = np.array(self.mbi.psf[band].shape) // 2
            self.psf_model[band].bounding_box = ((-ybbox, ybbox), 
                                                 (-xbbox, xbbox))
            if len(self.mbi.psf_fwhm) == 0:
                self.psf_fwhm[band] = measure_psf_fwhm(self.mbi.psf[band], 4)
            else:
                self.psf_fwhm[band] = self.mbi.psf_fwhm[band]
        
    def render_psf_model(self, band):    
        return self.psf_model[band].render()

    def subtract_galaxy(self, label=None, sersic_model=None, 
                        init_params={}, **kwargs):
        
        if sersic_model is None:
            use_hsc_mask = self.use_hsc_bright_mask['imfit']
            self.sersic_fitter.fit(self.mbi, label, init_params, 
                                   use_hsc_bright_mask=use_hsc_mask, **kwargs)
            sersic_model = self.sersic_fitter.sersic_model
        if type(sersic_model) == str:
            sersic_model = np.load(sersic_model, allow_pickle=True)[0]

        logger.info('subtracting sersic model from images')

        subtracted_image = LsstStruct()

        for band in self.mbi.bands:
            subtracted_image[band] = subtract_sersic(
                self.mbi.image[band], psf=self.mbi.psf[band], 
                **sersic_model[band].params)
            
        return subtracted_image
        
    def psf_photometry(self, mbi, label=None, subtract_galaxy=True, 
                       sersic_model=None, save_residual_images=False, 
                       **kwargs):
        
        self.mbi = mbi
        self._setup_psf()
        
        if subtract_galaxy:
            self.phot_image = self.subtract_galaxy(
                label, sersic_model, **kwargs)
        else:
            self.phot_image = mbi.image

        if self.use_hsc_bright_mask['phot']:
            logger.info('applying hsc bright object mask')
            for b in mbi.bands:
                mask = mbi.get_hsc_bright_object_mask(b).astype(bool)
                self.phot_image[b][mask] = 0

        catalog = LsstStruct()
        self.stddev = LsstStruct()
        
        if save_residual_images:
            self.residual_image = LsstStruct()
        
        for band in mbi.bands:
            if len(mbi.stats) > 0:
                self.stddev[band] = mbi.stats[band].stdev
            else:
                self.stddev[band] = self.bkgrms(self.phot_image[band])
            daogroup = DAOGroup(self.crit_separation * self.psf_fwhm[band])
            self.daofinder_opt['fwhm'] = self.psf_fwhm[band]
            self.phot_opts['aperture_radius'] = self.aperture_radius 
            self.phot_opts['aperture_radius'] *= self.psf_fwhm[band]
            
            daofind = DAOStarFinder(self.threshold * self.stddev[band], 
                                    exclude_border=True,
                                    **self.daofinder_opt)

            logger.info('performing ' + band + '-band psf photometry')
            
            photometry = IterativelySubtractedPSFPhotometry(
                finder=daofind, group_maker=daogroup,
                psf_model=self.psf_model[band], 
                **self.phot_opts
            )
            
            with warnings.catch_warnings():
                message = '.*The fit may be unsuccessful;.*'
                warnings.filterwarnings('ignore', message=message,
                                        category=AstropyUserWarning)
                catalog[band] = photometry(image=self.phot_image[band])
            
            if save_residual_images:
                logger.info('generating residual image')
                self.residual_image[band] = subtract_psf(mbi.image[band], 
                                                         self.psf_model[band], 
                                                         catalog[band])
        return catalog
    
    def forced_psf_photometry(self, forced_cat, save_residual_images=False):

        catalog = LsstStruct()
        positions = Table(names=['x_0', 'y_0'], 
                          data=[forced_cat['x_fit'], forced_cat['y_fit']])
        if save_residual_images:
            self.residual_image_forced = LsstStruct()
        
        for band in self.mbi.bands:
            daogroup = DAOGroup(self.crit_separation * self.psf_fwhm[band])
            aperture_radius = self.aperture_radius * self.psf_fwhm[band]
 
            logger.info('performing forced photometry for ' + band + ' band')

            self.psf_model[band].x_0.fixed = True
            self.psf_model[band].y_0.fixed = True

            photometry = BasicPSFPhotometry(
                finder=None, group_maker=daogroup, 
                fitshape=self.phot_opts['fitshape'],
                psf_model=self.psf_model[band], bkg_estimator=self.bkg,
                aperture_radius=aperture_radius
            )
            
            catalog[band] = photometry(image=self.phot_image[band], 
                                       init_guesses=positions)

            # turns of the order might be different for each band!!!
            catalog[band].sort('x_fit')
            
            if save_residual_images:
                logger.info('generating residual image')
                self.residual_image_forced[band] = subtract_psf(
                    self.mbi.image[band], self.psf_model[band], catalog[band])
        
        return catalog
    
    def get_matched_catalog(self, catalog, master_band='g'):
        
        logger.info('matching calalogs across bands with ' + master_band +\
                    '-band as the master band')
        other_bands = [b for b in self.mbi.bands if b != master_band]
        
        masks = []
        
        idx_master = np.arange(len(catalog[master_band]))
        count = np.ones_like(idx_master, dtype=int)
    
        for band in other_bands:
            idx_match, _ = pixel_xmatch(catalog[master_band], catalog[band], 
                                        max_sep=self.max_match_sep)
            count += np.in1d(idx_master, idx_match)
            
        idx_final = idx_master[count >= self.min_match_bands]
        matched_catalog = catalog[master_band][idx_final]
        matched_catalog['num_bands'] = count[count >= self.min_match_bands]
        
        return matched_catalog

    def build_final_catalog(self, cat):

        logger.info('building final catalog')

        bands = self.mbi.bands
        b0 = bands[0]

        # build mask to remove sources  
        # with negative flux in any band
        mask = np.ones_like(cat[b0], dtype=bool)
        for b in bands:
            mask &= cat[b]['flux_fit'] > 0
            mask &= cat[b]['flux_unc'] > 0
        for b in bands:
            cat[b] = cat[b][mask]

        final_cat = cat[b0].copy()
        final_cat['mag_' + b0] = 27 - 2.5 * np.log10(final_cat['flux_fit'])
        err = 2.5 * np.log10(1 + final_cat['flux_unc']/final_cat['flux_fit']) 
        final_cat['mag_err_' + b0] = err 
        final_cat.remove_columns(['flux_fit', 'flux_unc'])
        
        for b in bands[1:]:
            final_cat['mag_' + b] = 27 - 2.5 * np.log10(cat[b]['flux_fit'])
            err = 2.5 * np.log10(1 + cat[b]['flux_unc']/cat[b]['flux_fit']) 
            final_cat['mag_err_' + b] = err
        
        wcs = WCS(self.mbi.header[b0])
        skycoord = SkyCoord.from_pixel(final_cat['x_fit'], 
                                       final_cat['y_fit'], wcs)
        final_cat['ra'] = skycoord.ra.value
        final_cat['dec'] = skycoord.dec.value
        final_cat['ebv'] = dustmap.ebv(final_cat['ra'], final_cat['dec'])
        
        for b in bands:
            final_cat['A_' + b] = final_cat['ebv'] * getattr(ext_coeff, b)
            
        return final_cat

    def make_rgb(self, kind, rgb_bands='irg', stretch=0.8, Q=8):

        if not hasattr(self, kind):
            logger.error(f'image of kind {kind} not found')
            rgb = None
        elif getattr(self, kind) is None:
            logger.error(f'image of kind {kind} not found')
            rgb = None
        else:
            image = getattr(self, kind)
            rgb = make_lupton_rgb(image[rgb_bands[0]],
                                  image[rgb_bands[1]],
                                  image[rgb_bands[2]],
                                  stretch=stretch, Q=Q)

        return rgb

    def print_config(self):
        print('***** Pipe Config *****')
        for k, v in self.__dict__.items():
            print(k, '=', v)

    def run_pipeline(self, mbi, label=None, sersic_model=None, 
                     save_residual_images=False, **kwargs):

        logger.info('***** starting PSF photometry pipeline *****')

        self.residual_image = None
        self.residual_image_forced = None

        self.init_cat = self.psf_photometry(
            mbi, label, sersic_model=sersic_model, **kwargs)
        self.matched_cat = self.get_matched_catalog(
            self.init_cat, master_band=self.master_band)
        self.forced_cat = self.forced_psf_photometry(
            self.matched_cat, save_residual_images=save_residual_images)

        catalog = self.build_final_catalog(self.forced_cat)

        return catalog
