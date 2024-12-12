"""
oscillation_detection.py

Analyzes list of spike trains for oscillation detection via Whalen et al., 2020 method.

MATLAB author: Timothy C. Whalen
Python author: John E. Parker
"""

# import python modules
import numpy as np

# import user modules
import renewalPSD_phaseShift as rpp
import find_sig_osc as fso


def band_power_and_osc_detection(
    spikes,
    rates,
    lengths,
    srch_lo=0.5,
    srch_hi=4,
    force_freq=2,
    min_rate=5,
    min_length=30,
    cntl_lo=250,
    cntl_hi=500,
    FS=1000,
    wind=2**12,
    step=2**9,
    psd_threshp=0.05,
    phase_threshp=0.05,
    max_n=7,
):
    """
    Calculates renewal-corrected PSD and phase shift from list of spike trains using Welch's method (average of multiple overlapping windows).

    Parameters:
    spikes (list): List of lists, sublists containing spike times for analysis.
    rates (np array): Float of firing rates for spike trains.
    lengths (list): Float of times of recording lengths for spike trains.
    srch_lo=0.5 (float): Low end of frequency band to detect oscillations in.
    srch_hi=4 (float): High end of frequency band to detect oscillations in.
    force_freq=2 (float): Expected value for high power.
    min_rate=5 (float): Minimum firing rate needed for spike train to be evaluated.
    min_length=30 (float): Minimum recording length needed for spike to be evaluated.
    cntl_lo=250 (float): Low indices to use for control confidence interval.
    cntl_hi=500 (float): High indices to use for control confidence interval.
    FS=1000 (float): Frequency to downsample spike train to (also determines Nyquist frequency).
    wind=2**12 (int): Window size for Welch's method (ideally power of 2, also determines Rayleigh frequency)
    step=2**9 (int): Step size for overlapping windows
    psd_threshp=0.05: P-value threshold for PSD significance.
    phase_threshp=0.05 (float): P-value threshold for phase significance.
    max_n=7 (int): # points local extreme must be more extreme than to count as peak/trough.

    Returns:
    renewal_band_power (np array): Array of max renewal power in frequency band.
    band_power (np array): Array of max power in frequency band.
    band_oscs (np array, int): Boolean array (0 and 1) if oscillation detected in frequency band.
    """

    # Set up frequencyes
    freqs = np.arange(0, 50, FS / wind)
    freqs_long = np.arange(
        0,
        FS / 2,
        FS / wind,
    )

    # Find index closes to force_freq
    index = np.argmin(np.abs(freqs - force_freq))

    # Determine indicies in freq band and control intervals
    srch_inds = [
        np.argmax(freqs_long >= srch_lo),
        len(freqs_long) - np.argmax(freqs_long[::-1] <= srch_hi) - 1,
    ]
    cntl_inds = [
        np.argmax(freqs_long > cntl_lo),
        len(freqs_long) - np.argmax(freqs_long[::-1] <= cntl_hi) - 1,
    ]

    # Set up empty lists to be returned
    renewal_band_power = np.zeros(len(spikes))
    band_power = np.zeros(len(spikes))
    band_oscs = np.zeros(len(spikes))

    # Iterate through all spike trains and find power and oscillation
    for i in range(len(spikes)):
        if rates[i] >= min_rate and lengths[i] >= min_length:
            # Call renewal power and phase code
            psd_corr, phshift, psd_unc = rpp.renewalPSD_phaseShift(
                spikes[i], wind=wind, step=step, FS=FS
            )

            # Determine if significant oscillation
            sigp_inds, sig_inds = fso.find_sig_osc(
                psd_corr,
                phshift,
                srch_inds,
                cntl_inds,
                max_n,
                psd_threshp,
                phase_threshp,
            )

            # Set power values to return
            renewal_band_power[i] = psd_corr[index] / rates[i]
            band_power[i] = psd_unc[index] / rates[i]

            # Determine if oscillation occurred
            if len(sigp_inds) > 0:
                band_oscs[i] = 1

    return renewal_band_power, band_power, band_oscs
