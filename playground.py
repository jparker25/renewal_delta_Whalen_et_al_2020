import numpy as np
import renewalPSD_phaseShift as rpp
import sys
from matplotlib import pyplot as plt

data_direc = (
    "/Users/johnparker/paper_repos/Aristieta_Parker_Rubin_Gittis_2024_motor_rescue/data"
)

FS = 1000
wind = 2**12
min_rate = 5
rates = np.loadtxt(f"{data_direc}/training/dd_neurons/cell_baseline_frs.txt")
truth = np.loadtxt(f"{data_direc}/training/dd_neurons/osc_data.txt", delimiter=",")
psd_checks = np.zeros(truth.shape[0])

for i in range(len(psd_checks)):
    if rates[i] >= min_rate:
        spikes = np.loadtxt(
            f"{data_direc}/training/dd_neurons/spikes/cell_{i:04d}_baseline_spikes.txt"
        )
        freqs = np.arange(0, 50, FS / wind)
        delta_force_freq = 2
        index = np.argmin(np.abs(freqs - delta_force_freq))
        psd_corr, phshift, psd_unc = rpp.renewalPSD_phaseShift(spikes)
        psd_checks[i] = psd_corr[index] / rates[i]

    else:
        psd_checks[i] = 0

percent_change = np.abs(truth[:, 1] - psd_checks) * 100 / truth[:, 1]
percent_change[~np.isfinite(percent_change)] = 0
print(np.max(percent_change))
