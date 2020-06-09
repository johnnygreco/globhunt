import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt


__all__ = ['show_image', 'plot_completeness', 'plot_measurement_offset']


def show_image(image, percentile=[1, 99], subplots=None,
               rasterized=False, **kwargs):
    if subplots is None:
        figsize = kwargs.pop('figsize', (10, 10))
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw=dict(xticks=[], yticks=[]))
    else:
        fig, ax = subplots
    if percentile is not None:
        vmin, vmax = np.nanpercentile(image, percentile)
    else:
        vmin, vmax = None, None
    ax.imshow(image, origin='lower', cmap='gray_r', rasterized=rasterized,
              vmin=vmin, vmax=vmax)
    return fig, ax


def plot_completeness(input_cat, injected, band, subplots=None, xlabel=True, 
                      label_band=True, xlim=[19.8, 28], fs=25, ylim=[0, 1.05], 
                      tick_fs=21):

    if subplots is None:
        fig, ax = plt.subplots(figsize=(7.5, 6))
    else:
        fig, ax = subplots

    mag_range = [input_cat[band].min(), 28]
    h_inject, bin_edges = np.histogram(input_cat[band], 
                                       bins='auto', 
                                       range=mag_range)
    h_recover, _ = np.histogram(injected[band], bins=bin_edges)
    completeness = h_recover/h_inject
    mag_bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax.plot(mag_bins, completeness, 'tab:red', ls='-', lw=2.5)
    ax.plot(mag_bins, completeness, 'o', c='k')
    ax.tick_params('both', labelsize=tick_fs)
    ax.minorticks_on()

    if label_band:
        ax.text(0.15, 0.75, r'$'+ band + '$-band', 
                transform=ax.transAxes, fontsize=fs)

    ax.set_ylabel('Completeness', fontsize=fs)

    if xlabel:
        ax.set_xlabel('Magnitude', fontsize=fs)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    return fig, ax, mag_bins, completeness


def plot_measurement_offset(injected, recovered, band, subplots=None, 
                            label_band=True, xlim=[19.8, 28], 
                            ylim=[-0.65, 0.65], fs=25, tick_fs=21):

    if subplots is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig, ax = subplots

    diff = injected[band] - recovered['mag_' + band]

    bin_edges = np.histogram_bin_edges(injected[band], bins='auto')
    bins = 0.5*(bin_edges[1:] + bin_edges[:-1])

    result_16 = binned_statistic(injected[band], diff, bins=bins,
                                 statistic=lambda arr: np.percentile(arr, 16))
    result_50 = binned_statistic(injected[band], diff, bins=bins,
                                 statistic=lambda arr: np.percentile(arr, 50))
    result_84 = binned_statistic(injected[band], diff, bins=bins,
                                 statistic=lambda arr: np.percentile(arr, 84))
    mag_bins = 0.5*(result_50.bin_edges[1:] + result_50.bin_edges[:-1])


    ax.plot(injected[band], diff, 'k,', alpha=0.5)

    line_color = 'k'
    kw = dict(color='tab:red', lw=3)
    ax.plot(mag_bins, result_16.statistic, **kw)
    ax.plot(mag_bins, result_50.statistic, **kw)
    ax.plot(mag_bins, result_84.statistic, **kw)
    ax.axhline(y=0, ls='--', c=line_color, lw=3, zorder=10)

    ax.set_ylim(*ylim)
    ax.minorticks_on()

    ax.set_xlabel('Magnitude', fontsize=fs)
    ax.set_ylabel(r'$\delta m$', fontsize=fs)
    ax.tick_params('both', labelsize=tick_fs)

    if label_band:
        ax.text(0.15, 0.75, r'$'+ band + '$-band', 
                transform=ax.transAxes, fontsize=fs)

    ax.set_xlim(*xlim)

    return fig, ax
