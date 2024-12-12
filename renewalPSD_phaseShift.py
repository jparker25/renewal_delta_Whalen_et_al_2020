"""
renewalPSD_phaseShift.py

Implements Whalen et al., 2020 method of renewal power detection. 
Originally in MATLAB, rewritten in Python for ease of use.

MATLAB author: Timothy C. Whalen
Python author: John E. Parker
"""

# import python modules
import numpy as np


def renewalPSD_phaseShift(ts, wind=2**12, step=2**9, FS=1000):
    """
    Calculates renewal-corrected PSD and phase shift from spike train using Welch's method (average of multiple overlapping windows).

    Parameters:
    ts (numpy.ndarray): Nx1 array, spike times in seconds
    wind (int): Window size for Welch's method (ideally power of 2, also determines Rayleigh frequency)
    step (int): Step size for overlapping windows
    FS (float): Frequency to downsample spike train to (also determines Nyquist frequency)

    Returns:
    tuple: psd_corr (renewal-corrected PSD), phshift (mean phase shift), psd_unc (uncorrected PSD)
    """
    freqs = np.arange(0, FS / 2 + FS / wind, FS / wind)

    # Convert to binary train
    spkt = np.round(FS * ts).astype(int)
    len_delt = spkt[-1] - spkt[0] + 1
    spkdelt = np.zeros(len_delt)
    spkdelt[spkt - spkt[0]] = 1

    Ncalc = int(
        np.floor(len_delt / step - wind / step + 1)
    )  # Number of FFTs to compute
    psds = np.zeros((wind // 2, Ncalc))
    psds_unc = np.zeros((wind // 2, Ncalc))
    nspikes = np.zeros(Ncalc)
    cphf = np.zeros((wind // 2, Ncalc))

    for s in range(Ncalc):
        segdelt = spkdelt[step * (s) : step * (s) + wind]
        sps = np.where(segdelt)[0]
        nspikes[s] = sps.size
        if nspikes[s] <= 3:  # Edge case that gives anomalous results
            cphf[:, s] = np.nan
            psds[:, s] = np.nan
            psds_unc[:, s] = np.nan
            continue

        # Compute PSD of renewal process
        isi = np.diff(sps)
        edges = np.arange(0, wind + 1) - 0.001
        isin = np.histogram(isi, bins=edges)[0]
        isidist = isin / np.sum(isin)
        phat = np.fft.fft(isidist)
        phat[0] = 0
        chat = np.real((1 + phat) / (1 - phat))

        # Compute PSD and normalize to renewal PSD
        ff = np.fft.fft(segdelt - np.mean(segdelt))
        powf = np.abs(ff) ** 2  # Power spectrum
        psd_corr = (powf / nspikes[s]) / chat
        psds_unc[:, s] = powf[: wind // 2] * 2 / (FS * wind)
        psds[:, s] = psd_corr[: wind // 2]

        # Compute and align phases
        phf = np.angle(ff)
        phf = phf[: wind // 2]  # Remove repeated half
        cphf[:, s] = (
            np.mod(
                np.pi + (phf - 2 * np.pi * (s) * step / 1000 * freqs[: wind // 2]),
                2 * np.pi,
            )
            - np.pi
        )

    phdiff = np.diff(cphf, axis=1)  # Phase differences over time
    cphdiff = np.pi - np.abs(np.abs(phdiff) - np.pi)  # Make pi farthest, 2pi like zero
    cphdiff = cphdiff[:, ~np.isnan(cphdiff[0, :])]
    phshift = np.mean(cphdiff, axis=1)

    nspikes_nonan = nspikes[~np.isnan(psds[1, :])]
    nspikes_norm = nspikes_nonan / np.sum(nspikes_nonan)
    psd_nonan = psds[:, ~np.isnan(psds[1, :])]
    psd_corr = np.sum(
        psd_nonan * nspikes_norm, axis=1
    )  # Rescale by number of spikes in each window
    psds_unc_nonan = psds_unc[:, ~np.isnan(psds_unc[1, :])]
    psd_unc = np.mean(psds_unc_nonan, axis=1)

    return psd_corr, phshift, psd_unc
