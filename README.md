# Automated Estimation of Frequency and Spatial Extent of Periodic and Rhythmic Epileptiform Activity

This repository contains the code and data to reproduce the results from:

**Tăuțan AM, Jing J, Basovic L, Hadar PN, Sartipi S, Fernandes MP, Kim J, Struck AF, Westover MB, Zafar SF.** "Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data"

## Overview

This repository provides automated algorithms for detecting and characterizing epileptiform activity in continuous EEG data, specifically:
- **LPD**: Lateralized Periodic Discharges
- **GPD**: Generalized Periodic Discharges
- **LRDA**: Lateralized Rhythmic Delta Activity
- **GRDA**: Generalized Rhythmic Delta Activity

The algorithms extract two key features:
1. **Frequency** of the epileptiform activity
2. **Spatial extent** (brain regions affected)

## Quick Links

- **New to this project?** Start with [QUICKSTART.md](QUICKSTART.md)
- **Need data?** See [DATASET_INFO.md](DATASET_INFO.md)
- **Testing?** See [TESTING.md](TESTING.md)

## Table of Contents

- [Installation](#installation)
- [Data Access](#data-access)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Citation](#citation)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

## Installation

### Prerequisites
- Python 3.8.19
- Conda or Miniconda

### Setup Environment

1. Clone this repository:
```bash
git clone https://github.com/bdsp-core/IIIC-Frequency-Analysis-2.git
cd IIIC-Frequency-Analysis-2
```

2. Create the conda environment from the provided environment file:
```bash
conda env create -f code/environment.yml
conda activate foe
```

**Alternative installation**: If you prefer to install packages separately, we recommend installing with `fooof` to simplify dependency requirements.

### Key Dependencies
- Python 3.8.19
- MNE (1.0.3) - EEG data processing
- NumPy (1.24.4)
- Pandas (1.5.3)
- SciPy (1.10.1)
- Matplotlib (3.7.3)
- fooof (1.0.0) - Frequency modeling
- pyhht (0.1.0) - Hilbert-Huang Transform
- irrcac (0.4.0) - Inter-rater reliability analysis

See [code/environment.yml](code/environment.yml) for the complete list of dependencies.

## Data Access

### EEG Dataset
The EEG dataset and expert annotations used in this study are available through the **Brain Data Science Platform (BDSP)**.

**To access the data:**
1. Visit [bdsp.io](https://bdsp.io)
2. Submit an access request through their portal
3. Once approved, download the dataset ZIP file from AWS S3
4. Extract the ZIP file to create the `data/` directory in your repository

**Dataset Contents:**
- **1,060 EEG files** (.mat format, ~283 MB total)
- **12 annotation files** (CSV format with expert ratings)
- See [data_manifest.csv](data_manifest.csv) for complete file listing

**For detailed data access instructions, see [DATASET_INFO.md](DATASET_INFO.md)**

**Dataset structure** (after extraction):
```
data/
├── dataset_eeg/          # EEG recordings (50-second segments)
│   ├── gpd/              # Generalized Periodic Discharges
│   ├── lpd/              # Lateralized Periodic Discharges
│   ├── grda/             # Generalized Rhythmic Delta Activity
│   └── lrda/             # Lateralized Rhythmic Delta Activity
└── annotations/          # Expert annotations (CSV files)
    ├── GPDS_LB_2_2025.csv
    ├── GPDS_PH_3_2025.csv
    ├── GPDS_SZ_3_2025.csv
    ├── GRDA_LB_2_2025.csv
    ├── GRDA_PH_3_2025.csv
    ├── GRDA_SZ_3_2025.csv
    ├── LPDS_LB_2_2025.csv
    ├── LPDS_PH_3_3025.csv
    ├── LPDS_SZ_3_2025.csv
    ├── LRDA_LB_2_2025.csv
    ├── LRDA_PH_3_2025.csv
    └── LRDA_SZ_3_2025.csv
```

**Data format:**
- EEG files: MATLAB `.mat` files containing 50-second segments sampled at 200 Hz
- Annotations: CSV files with expert ratings for frequency and spatial extent
- Three expert annotators: LB, PH, SZ

## Repository Structure

```
.
├── code/
│   ├── pd_detector/              # Periodic discharge detectors
│   │   ├── pd_detect.py          # Original McGraw et al. detector
│   │   ├── calculate.py          # Core calculation functions
│   │   ├── find_events.py        # Event detection utilities
│   │   ├── eegfilt.py            # EEG filtering
│   │   └── utils.py              # Utility functions
│   ├── pd_detector_alternate/    # Enhanced PD detector
│   │   └── pd_detect_alternate.py # Supports 'apd' and 'zscore' peak detection
│   ├── rda_detector/             # Rhythmic delta activity detectors
│   │   ├── rda1a_fft.py          # FFT-based with frequency modeling
│   │   ├── rda1b_fft.py          # FFT-based with enhanced peak selection
│   │   └── rda2_hht.py           # Hilbert-Huang Transform based
│   ├── imageGeneration/          # Visualization functions
│   │   └── plot_events.py        # Plotting functions for algorithm output
│   ├── extract_frequency_spatial_extent.py  # Main analysis script
│   ├── visualize_output.py       # Example visualization script
│   ├── irr_analysis_onagreement.ipynb       # IRR analysis (agreement subset)
│   ├── irr_analysis_fulldataset.ipynb       # IRR analysis (full dataset)
│   ├── environment.yml           # Conda environment specification
│   └── readme.txt                # Original code documentation
├── data/                         # Data directory (download separately)
│   ├── dataset_eeg/              # EEG recordings
│   └── annotations/              # Expert annotations
├── results/                      # Algorithm outputs
│   ├── gpd_results.csv           # GPD detection results
│   ├── lpd_results.csv           # LPD detection results
│   ├── grda_results.csv          # GRDA detection results
│   ├── lrda_results.csv          # LRDA detection results
│   └── results_figures/          # Generated figures
└── README.md                     # This file
```

## Usage

### Quick Start

After installing dependencies and downloading data, you can test the algorithms on individual files:

```python
import hdf5storage
import h5py
import rda_detector as rda
import pd_detector_alternate as pd_detect
import imageGeneration as im
import matplotlib.pyplot as plt

# Load EEG data
def load_mat_file(filepath):
    try:
        return hdf5storage.loadmat(filepath)
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            return {key: f[key][()] for key in f.keys()}

# Example 1: Detect LRDA
mat = load_mat_file('data/dataset_eeg/lrda/abn514_20130114_210759_3843_seg_5.mat')
segment = mat['data']
fs = 200  # Sampling frequency

# Run detector
data_obj = rda.rda2_hht(segment, fs, 1)

# View results
print(f"Event type: {data_obj['type_event']}")
print(f"Frequency: {data_obj['event_frequency']} Hz")
print(f"Spatial extent: {data_obj['spatial_extent']}")
print(f"Brain regions: {data_obj['spatial_areas']}")

# Visualize
fig = im.plot_rda_events(segment, data_obj, 0, int(segment.shape[1]/fs), fs)
plt.show()

# Example 2: Detect LPD
mat = load_mat_file('data/dataset_eeg/lpd/abn1762_20180428_125520_2948_seg_3.mat')
segment = mat['data']
data_obj = pd_detect.pd_detect_alternate(segment, fs, pk_detect='apd')

# View results
fig = im.plot_pd_events(segment, data_obj, 0, int(segment.shape[1]/fs), fs)
plt.show()
```

### Processing Full Dataset

To run all algorithms on the complete dataset:

```bash
cd code
python extract_frequency_spatial_extent.py
```

This will:
1. Process all EEG files in `data/dataset_eeg/` for each event type (GPD, LPD, GRDA, LRDA)
2. Run multiple detector variants on each file
3. Save results to `results/` directory as CSV files
4. Generate optional visualization figures

**Expected runtime:** Approximately 2-4 hours depending on hardware (processing ~1,000+ EEG segments)

## Algorithm Details

### Rhythmic Delta Activity (RDA) Detectors

Three variants are implemented for LRDA and GRDA detection:

1. **rda1a_fft**: FFT-based detector with frequency modeling using FOOOF
2. **rda1b_fft**: FFT-based with additional peak selection logic (best performance for RDA)
3. **rda2_hht**: Hilbert-Huang Transform based detector

### Periodic Discharge (PD) Detectors

Three variants are implemented for LPD and GPD detection:

1. **pd_detect**: Original McGraw et al. detector
2. **pd_detect_alternate (pk_detect='apd')**: Enhanced detector with adaptive peak detection (best performance for PD)
3. **pd_detect_alternate (pk_detect='zscore')**: Z-score based peak detection

### Output Format

Each detector returns a dictionary with:
- `type_event`: Detected event type (e.g., 'LPD', 'GPD', 'LRDA', 'GRDA')
- `event_frequency`: Dominant frequency in Hz
- `spatial_extent`: Proportion of brain regions affected (0-1)
- `spatial_areas`: List of affected brain regions (e.g., ['LF', 'RF', 'LT', 'RT'])
- `channels`: Channel-level detection information
- `peaks`: Detected peak information

**Brain region abbreviations:**
- LF: Left Frontal
- RF: Right Frontal
- LT: Left Temporal
- RT: Right Temporal
- LO: Left Occipital
- RO: Right Occipital
- LCP: Left Central-Parietal
- RCP: Right Central-Parietal

## Reproducing Paper Results

### 1. Run Detectors on Dataset

```bash
cd code
python extract_frequency_spatial_extent.py
```

This generates result CSV files in `results/`:
- `gpd_results.csv`
- `lpd_results.csv`
- `grda_results.csv`
- `lrda_results.csv`

### 2. Inter-Rater Reliability Analysis

Open and run the Jupyter notebooks:

```bash
cd code
jupyter notebook
```

**For agreement subset analysis** (segments where all annotators agreed on classification):
- Open `irr_analysis_onagreement.ipynb`
- Run all cells to compute inter-rater reliability metrics

**For full dataset analysis**:
- Open `irr_analysis_fulldataset.ipynb`
- Run all cells for comprehensive analysis

These notebooks compute:
- Intraclass Correlation Coefficient (ICC)
- Concordance Correlation Coefficient (CCC)
- Agreement statistics between algorithms and expert annotators

### 3. Visualization

To visualize algorithm outputs:

```bash
cd code
python visualize_output.py
```

This script demonstrates:
- Loading EEG data
- Running detectors
- Visualizing detected events with frequency and spatial extent overlays

## Best Performing Algorithms

Based on validation results in the paper:

- **For RDA (LRDA/GRDA)**: `rda1b_fft`
- **For PD (LPD/GPD)**: `pd_detect_alternate` with `pk_detect='apd'`

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{tautan2025automated,
  title={Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data},
  author={T{\u{a}}u{\c{t}}an, Alexandra-Maria and Jing, Jin and Basovic, Lara and Hadar, Parimala Nallappan and Sartipi, Sahar and Fernandes, Marcos Paulo and Kim, Jonathan and Struck, Aaron F and Westover, M Brandon and Zafar, Sahar F},
  journal={Journal of Neural Engineering},
  volume={22},
  number={6},
  year={2025},
  month={dec},
  doi={10.1088/1741-2552/ae2716},
  pmid={41330044},
  url={https://pubmed.ncbi.nlm.nih.gov/41330044/}
}
```

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError` when importing detectors**
```bash
# Solution: Ensure you're running Python from the code/ directory
cd code
python extract_frequency_spatial_extent.py
```

**Issue: Cannot load .mat files**
```python
# Solution: Use the load_mat_file helper function
import hdf5storage
import h5py

def load_mat_file(filepath):
    try:
        return hdf5storage.loadmat(filepath)
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            return {key: f[key][()] for key in f.keys()}
```

**Issue: Conda environment creation fails**
```bash
# Solution: Try creating environment with specific channels
conda env create -f code/environment.yml -c conda-forge
# Or install fooof first
conda create -n foe python=3.8
conda activate foe
conda install -c conda-forge fooof
```

**Issue: Results differ from paper**
- Verify you're using the correct detector variant (see "Best Performing Algorithms" section)
- Check that data files match expected format (200 Hz, 19 channels)
- Ensure you're analyzing the same segments as in the paper

## Contact

For questions about:
- **Data access**: Brain Data Science Platform at [bdsp.io](https://bdsp.io) or support@bdsp.io
- **Code issues**: Open an issue on [GitHub Issues](https://github.com/bdsp-core/IIIC-Frequency-Analysis-2/issues)
- **Research collaboration**: Contact the corresponding authors via the publication

## License

This code is licensed under **CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)** - see the [LICENSE.txt](LICENSE.txt) file for details.

**Key Terms:**
- ✅ **Attribution Required**: Cite the paper when using this code
- ✅ **Academic/Research Use**: Free for non-commercial research
- ❌ **Commercial Use Prohibited**: Contact authors for commercial licensing

**Data License**: The EEG dataset is provided by BDSP under their Data Use Agreement. Users must comply with BDSP terms of use.

## Acknowledgments

This work was conducted at:
- Massachusetts General Hospital, Department of Neurology
- Brain Data Science Platform (BDSP)
- Harvard Medical School

We thank all the expert EEG annotators (LB, PH, SZ) for their careful review of the dataset.

## Related Publications

If you're interested in related work on automated EEG analysis:
- McGraw CM, et al. "Detection of rhythmic delta activity in the EEG" (original PD detector)
- Additional references can be found in the main publication
