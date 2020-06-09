import os, shutil
from getpass import getpass
import matplotlib.pyplot as plt
from multiprocessing import Pool
from astropy.table import Table
from globhunt import utils, logger, PSFPhotometry, HSCButler
from globhunt import MultiBandImage, get_hsc_lsbg_params


@utils.func_timer
def run(run_args):

    src_id, config_fn, out_path, password, username, force_new = run_args
    gal_params = get_hsc_lsbg_params(
        src_id, params=['ra', 'dec', 'I_e', 'n', 'r_e', 'ell', 'PA'])
    config = utils.load_config(config_fn)

    src_label = 'G' + str(src_id)
    cache_path = os.path.join(out_path, 'cache')
    sersic_fn = os.path.join(cache_path, src_label + '-sersic-model.npy')
    mbi_fn = os.path.join(cache_path, src_label + '-mbi.pkl')

    init_params = {} 
    if not os.path.isfile(sersic_fn) or force_new:
        sersic_model = None
        init_params = dict(
            I_e = [gal_params.I_e * utils.hsc_pixscale**2, 0, 1000],
            r_e = [gal_params.r_e / utils.hsc_pixscale, 0, 500],
            n = [gal_params.n, 0.05, 5.0],
            ell = gal_params.ell,
            PA = gal_params.PA,
        )
    else:
        logger.info(f'using previous sersic fit: {sersic_fn}')
        sersic_model = sersic_fn

    if not os.path.isfile(mbi_fn) or force_new:
        logger.info(f'fetching data for {src_label}')
        butler = HSCButler(username, password)
        mbi = butler.fetch_cutout(
            gal_params.ra, gal_params.dec, config['image_width'], 
            bands=config['bands'])
        logger.info(f'saving mbi object to {mbi_fn}')
        mbi.to_pickle(mbi_fn)
    else:
        logger.info(f'using previous mbi object: {mbi_fn}')
        mbi = MultiBandImage.from_pickle(mbi_fn)

    pipe = PSFPhotometry(config['imfit_io'], config)

    catalog = pipe.run_pipeline(
        mbi, label=src_label, 
        sersic_model=sersic_model, 
        init_params=init_params,
        save_residual_images=config['save_residual_images']
    )

    if not os.path.isfile(sersic_fn):
        logger.info(f'saving sersic model to {sersic_fn}')
        pipe.sersic_fitter.save_model(sersic_fn)

    cat_fn = os.path.join(out_path, src_label + '-cat.csv')
    catalog.write(cat_fn, overwrite=True)

    if config['save_residual_images']:
        logger.info('saving residual image')
        fig, ax = plt.subplots(1, 2, figsize=(14, 14), 
                               subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(wspace=0.05)
        ax[0].imshow(mbi.make_rgb(stretch=0.5), origin='lower')
        ax[1].imshow(pipe.make_rgb('residual_image_forced', stretch=0.5), 
                     origin='lower')
        ax[0].set_title('Original Image', fontsize=25)
        ax[1].set_title('Residual Image', fontsize=25)
        fn = os.path.join(out_path, f'residual-images/{src_label}.png')
        fig.savefig(fn, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser

    # parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('-s', '--source-id', type=int, 
                        nargs='*', required=True)
    parser.add_argument('-c', '--config-fn', required=True)
    parser.add_argument('-u', '--username', required=True)
    parser.add_argument('-p', '--password')
    parser.add_argument('-f', '--force-new', action='store_true')
    parser.add_argument('-r', '--run-label', default='pipe-run')
    parser.add_argument('-n', '--nproc', default=1, type=int)
    args = parser.parse_args()

    ids = args.source_id
    logger.info(f'starting pipeline for {len(ids)} sources in HSC LSB catalog')

    if args.password is None:
        password = getpass('Enter password for HSC data access: ')
    else:
        password = args.password

    # create output directories if needed 
    config = utils.load_config(args.config_fn)
    out_path = os.path.join(config['io'], args.run_label)
    utils.mkdir_if_needed(out_path)
    utils.mkdir_if_needed(os.path.join(out_path, 'cache'))

    if config['save_residual_images']:
        res_dir = os.path.join(out_path, 'residual-images')
        utils.mkdir_if_needed(res_dir)

    if args.force_new:
        logger.warning('will force new sersic fits and mbi objects')

    # copy config file to output directory for reference
    shutil.copyfile(args.config_fn, os.path.join(out_path, 'config.yml'))

    pars = [args.config_fn, out_path, password, args.username, args.force_new]

    if args.nproc > 1:
        run_args = [(src_id, *pars) for src_id in ids]
        with Pool(args.nproc) as pool:
            pool.map(run, run_args)
    else:
        run_args = [(src_id, *pars) for src_id in ids]
        for args in run_args:
            run(args)
