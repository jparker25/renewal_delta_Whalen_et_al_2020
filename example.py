"""
example.py

Script that compares oscillation detection
and power values from MATLAB versus Python
algorithm implementations via percent error.

Author: John E. Parker
"""

import numpy as np
import oscillation_detection as od
from matplotlib import pyplot as plt
import seaborn as sns


data_direc = "sample_data"

# read in rates, spike trains, and recording lengths
rates = np.loadtxt(f"{data_direc}/cell_baseline_frs.txt")
recording_lengths = np.loadtxt(f"{data_direc}/cell_baseline_lengths.txt")

# read in all spike trains
all_spikes = [
    np.loadtxt(f"{data_direc}/spikes/cell_{i:04d}_baseline_spikes.txt")
    for i in range(len(rates))
]

# run oscillation detection algorithm with default parameters
rpow, pow, osc_detec = od.band_power_and_osc_detection(
    all_spikes, rates, recording_lengths
)

# read in oscillation metrics from matlab
truth = np.loadtxt(f"{data_direc}/osc_data.txt", delimiter=",")

# calculate percent change from python vs matlab algorithms
percent_change = np.abs(truth[:, 1] - rpow) * 100 / truth[:, 1]

# filter NaNs
percent_change[~np.isfinite(percent_change)] = 0

# find number of disagreements in oscillation detected
misses = 0
for i in range(truth.shape[0]):
    if truth[i, 0] != osc_detec[i]:
        misses += 1

# generate figure of percent error distributions
fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
sns.histplot(
    x=percent_change, stat="probability", kde=True, edgecolor="w", color="k", ax=ax
)
ax.set_xlabel("Percent Error")
fig.savefig("sample_data/percent_error.jpg", bbox_inches="tight")
plt.close()
