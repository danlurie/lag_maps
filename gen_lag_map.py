from __future__ import division
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy import signal
from nilearn import image, plotting, masking
import sys

def get_numerator(signal_a, signal_b, lag):
    """
    Calculates the numerator of the cross-correlation equation.
    
    Parameters
    ----------
    signal_a : array_like (1D)
        Reference signal.
    signal_b : array_like (1D)
        Test signal. Must be the same length as signal_a.
    lag : int
        Lag by which signal_b will be shifted relative to signal_a.
        
    Returns
    -------
    array_like (1D)
        Element-wise product of matching time points in the lagged signals.
    """ 
    if lag == 0:
        numerator = np.multiply(signal_a, signal_b)
    # If lag is positive, shift signal_b forwards relative to signal_a.
    if lag > 0:
        numerator = np.multiply(signal_a[lag:], signal_b[0:-lag])
    # If lag is negative, shift signal_b backward relative to signal_a.
    if lag < 0:
        numerator = np.multiply(signal_b[-lag:], signal_a[0:lag])
    return numerator

def get_denominator(signal_a, signal_b):
    """
    Calculates the denominator of the cross-correlation equation.
    
    Parameters
    ----------
    signal_a : array_like (1D)
        Reference signal.
    signal_b : array_like (1D)
        Test signal. Must be the same length as signal_a.
        
    Returns
    -------
    float
        Product of the standard deviations of the input signals.
    """ 
    return np.std(signal_a) * np.std(signal_b)

def calc_xcorr(signal_a, signal_b, lag):
    """
    Calculate the cross-correlation of two signals at a given lag.
    
    Parameters
    ----------
    signal_a : array_like (1D)
        Reference signal.
    signal_b : array_like (1D)
        Test signal. Must be the same length as signal_a.
    lag : int
        Lag by which signal_b will be shifted relative to signal_a.
        
    Returns
    -------
    float
        Normalized cross-correlation.
    """ 
    xcorr = np.true_divide(1., len(signal_a)-np.absolute(lag)) * np.sum(np.true_divide(get_numerator(signal_a, signal_b, lag),
              get_denominator(signal_a, signal_b)))
    return xcorr

def sliding_xcorr(signal_a, signal_b, lags):
    """
    Calculate the cross-correlation of two signals over a range of lags.
    
    Parameters
    ----------
    signal_a : array_like (1D)
        Reference signal.
    signal_b : array_like (1D)
        Test signal. Must be the same length as signal_a.
    lags : array_like (1D)
        Lags by which signal_b will be shifted relative to signal_a.
        
    Returns
    -------
    array_like (1D)
        Normalized cross-correlation at each lag.
    """ 
    xcorr_vals = []
    for lag in lags:
        xcorr = calc_xcorr(signal_a, signal_b, lag)
        xcorr_vals.append(xcorr)
    return np.array(xcorr_vals)

# Adapted from https://gist.github.com/endolith/255291
def parabolic(sample_array, peak_index):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample local maximum when nearby samples are known.
   
    Parameters
    ----------
    sample_array : array_like (1D)
        Array of samples.
    peak_index : int
        Index for the local maximum in sample_array for which to estimate the inter-sample maximum.
   
    Returns
    -------
    tuple
        The (x,y) coordinates of the vertex of a parabola through peak_index and its two neighbors.
    """
    vertex_x = 1/2. * (sample_array[peak_index-1] - sample_array[peak_index+1]) / (sample_array[peak_index-1] - 2 * sample_array[peak_index] + sample_array[peak_index+1]) + peak_index
    vertex_y = sample_array[peak_index] - 1/4. * (sample_array[peak_index-1] - sample_array[peak_index+1]) * (vertex_x - peak_index)
    return (vertex_x, vertex_y)

def gen_lag_map(epi_img, brain_mask_img, gm_mask_img, lags):
    epi_gm_masked = epi_img[gm_mask_img]
    signal_a = np.mean(epi_gm_masked, axis=0)
    epi_brain_masked = epi_img[brain_mask_img]
    lag_index_correction = np.sum(np.array(lags) > 0)
    xcorr_array = []
    for voxel in epi_brain_masked:
    #for i, voxel in enumerate(epi_brain_masked):
        #print(i)
        signal_b = voxel
        vox_xcorr = sliding_xcorr(signal_a, signal_b, lags)
        xcorr_maxima = signal.argrelmax(np.array(vox_xcorr), order=1)[0]
        if len(xcorr_maxima) == 0:
            interp_max = np.argmax(vox_xcorr)
        elif len(xcorr_maxima) == 1:
            interp_max = parabolic(vox_xcorr, xcorr_maxima[0])[0]
            interp_max = interp_max - lag_index_correction
        elif len(xcorr_maxima) > 1:
            xpeak = xcorr_maxima[np.argmax(vox_xcorr[xcorr_maxima])]
            interp_max = parabolic(vox_xcorr, xpeak)[0]
            interp_max = interp_max - lag_index_correction
        xcorr_array.append(interp_max)
    return(np.array(xcorr_array))


epi = sys.argv[1]
brain_mask = sys.argv[2]
gm_mask = sys.argv[3]
max_lag = int(sys.argv[4])
out_prefix = sys.argv[5]

lags = range(-max_lag, max_lag+1)

brain_mask_img = nib.load(brain_mask)
brain_mask_data = brain_mask_img.get_data().astype(bool)

gm_mask_img = nib.load(gm_mask)
gm_mask_data = gm_mask_img.get_data().astype(bool)

epi_img = nib.load(epi)
epi_img_resampled = image.resample_img(epi_img, target_affine=brain_mask_img.get_affine(), target_shape=brain_mask_img.shape)
epi_data_resampled = epi_img_resampled.get_data()

lag_map_data = gen_lag_map(epi_data_resampled, brain_mask_data, gm_mask_data, lags)

lag_map_image = masking.unmask(lag_map_data, brain_mask_img)

nib.save(lag_map_image, '/home/despo/dlurie/Projects/lag_maps/test_data/megarest/sub101/{}_lag_map.nii.gz'.format(out_prefix))
