## This repository detects oscillations in given frequency bands existing in neural spike train data as described in Whalen et al., 2020. Please cite according, reference provided below.

This is the Python version of the oscillation detection algorithm described in Whalen et al., 2020:

<b>Whalen TC, Willard AM, Rubin JE, Gittis AH. Delta oscillations are a robust biomarker of dopamine depletion severity and motor dysfunction in awake mice. J Neurophysiol. 2020 Aug 1;124(2):312-329. doi: 10.1152/jn.00158.2020. Epub 2020 Jun 24. PMID: 32579421; PMCID: PMC7500379.</b>

This repository was created by John E. Parker and streamlined using ChatGPT. If using, please cite this repository and the above paper.

## Usage:
The ideal usage would be to create a list of spike trains, an array of firing rates, and an array of recording lengths. Snippets of `example.py` are reproduced below for reference that show how this can be done.

```
data_direc = "sample_data"

# read in rates, spike trains, and recording lengths
rates = np.loadtxt(f"{data_direc}/cell_baseline_frs.txt")
recording_lengths = np.loadtxt(f"{data_direc}/cell_baseline_lengths.txt")

# read in all spike trains
all_spikes = [
    np.loadtxt(f"{data_direc}/spikes/cell_{i:04d}_baseline_spikes.txt")
    for i in range(len(rates))
]
```
Then call the oscillation detection algorithm with the respective data.
```
# run oscillation detection algorithm with default parameters
rpow, pow, osc_detec = od.band_power_and_osc_detection(
    all_spikes, rates, recording_lengths
)
```
Below is a figure comparing the percent error between the MATLAB version and this version of renewal power for the dataset provided in `sample_data`. The maximum of the horizontal axis is at approximately 0.85%, meaning that the MATLAB and Python implementations agree quite well. 

<img src="/sample_data/percent_error.jpg" alt="Percent Error" width="300" height="200">

The data used in `example.py` is a subset of the data from the following bioRxiv paper:

Asier Aristieta, John E. Parker, Mary D. Cundiff, Thomas Fuchs, Byung Kook Lim, Jonathan E. Rubin, Aryn H. Gittis. "Stimulation of the Medial SNr Promotes Sustained Motor Recovery and Counteracts Parkinsonian Pathophysiology in Dopamine Depleted Mice". bioRxiv 2024.12.09.627637; doi: 10.1101/2024.12.09.627637


## Tips
- Ensure spike trains have at least 30s of data.
- Ensure spike train rates are at least 5 Hz or more. 


## Getting Started
Python 3.12.11 was used in a virtual environment for all analysis and figures in this paper. Please refer to Creation of Virtual Environments on how to use a virtual environment.

Once the virtual environment has been created, install the required packages in `requirements.txt`:

```
$ pip install -r requirements.txt
```

