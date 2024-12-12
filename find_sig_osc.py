# Python translation of MATLAB script: find_sig_peaks.m
import numpy as np
import sys


def find_sig_peaks(data, thresh, max_n, srch_inds=None):
    """
    Find local maxima in data that exceed a threshold.

    Parameters:
        data (np.ndarray): Vector containing data to search.
        thresh (float): Threshold which data must exceed.
        max_n (int): Identify local maxima over this number of points.
        srch_inds (list or np.ndarray, optional): Indices of range within data over which to search.

    Returns:
        np.ndarray: Indices into data of local maxima that exceed the threshold.
    """
    max_inds = find_local_nmax(data, max_n)
    sig_inds = np.where(data > thresh)[0]
    peak_inds = np.intersect1d(max_inds, sig_inds)

    if srch_inds is not None:
        peak_inds = np.intersect1d(peak_inds, srch_inds)

    return peak_inds


# Python translation of MATLAB script: find_local_nmax.m
def find_local_nmax(X, n):
    """
    Find all peaks in vector X and return their index positions.

    Parameters:
        X (np.ndarray): Vector of arbitrary length.
        n (int): Find max over n samples (must be odd).

    Returns:
        np.ndarray: Indices into X where peaks occur.
    """
    if n % 2 == 0:
        raise ValueError("n must be an odd number when using find_local_nmax")

    X = np.asarray(X)
    if len(X.shape) > 1:
        raise ValueError("find_local_nmax only works on 1D arrays")

    filt_len = (n - 1) // 2
    data_len = len(X)

    if data_len < filt_len:
        return np.array([])

    # Pad ends of data (by mirroring)
    start_ra = X[filt_len - 1 :: -1]
    stop_ra = X[: -(filt_len + 1) : -1]
    pad_X = np.concatenate((start_ra, X, stop_ra))

    filt_ind = np.arange(-filt_len, filt_len + 1)
    start = filt_len
    stop = data_len + start

    ind = []
    for i in range(start, stop):
        if pad_X[i] == max(pad_X[i + filt_ind]):
            ind.append(i - filt_len)

    # Ensure peaks are > n points apart
    ind = np.array(ind)
    good_inds = []
    for i in range(len(ind)):
        near_inds = np.abs(ind - ind[i]) < filt_len
        near_inds[i] = False

        if not any(near_inds):
            good_inds.append(ind[i])
        else:
            if X[ind[i]] > max(X[ind[near_inds]]):
                good_inds.append(ind[i])

    return np.array(good_inds)


# Python translation of MATLAB script: find_sig_osc.m
def find_sig_osc(psd, phshift, srch_inds, cntl_inds, max_n, psd_threshp, phase_threshp):
    """
    Given PSD and phase shift, finds significant oscillations.

    Parameters:
        psd (np.ndarray): Power spectrum (N = # frequencies).
        phshift (np.ndarray): Phase shift plot.
        srch_inds (tuple): Low and high indices of psd/phshift to search for significant peak/trough.
        cntl_inds (tuple): Low and high indices to use for control confidence interval.
        max_n (int): # points local extreme must be more extreme than to count as peak/trough.
        psd_threshp (float): P-value threshold for PSD significance.
        phase_threshp (float): P-value threshold for phase significance.

    Returns:
        tuple: Indices of significant phase shift trough and PSD peak (sigp_inds),
               indices of significant PSD peak ignoring phase shift (sig_inds).
    """
    from scipy.stats import norm

    p = psd_threshp / (srch_inds[1] - srch_inds[0] + 1)

    thresh = norm.ppf(1 - p) * np.std(psd[cntl_inds[0] : cntl_inds[1]]) + 1

    sig_inds = find_sig_peaks(
        psd[: srch_inds[1] + int(np.ceil(max_n / 2))],
        thresh,
        max_n,
        np.arange(srch_inds[0], srch_inds[1] + 1),
    )

    if len(sig_inds) > 0:
        phasep = phase_threshp / len(sig_inds)
        phase_thresh = -norm.ppf(1 - phasep) * np.std(
            phshift[cntl_inds[0] : cntl_inds[1]]
        ) + np.mean(phshift[cntl_inds[0] : cntl_inds[1]])
        sigp_inds = sig_inds[phshift[sig_inds] < phase_thresh]
    else:
        sigp_inds = []

    return sigp_inds, sig_inds
