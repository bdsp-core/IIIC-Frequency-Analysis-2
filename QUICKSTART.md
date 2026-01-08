# Quick Start Guide

Get started with IIIC-Frequency-Analysis-2 in minutes!

## 1. Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/bdsp-core/IIIC-Frequency-Analysis-2.git
cd IIIC-Frequency-Analysis-2

# Create conda environment
conda env create -f code/environment.yml
conda activate foe
```

## 2. Get Data Access (1-2 business days)

1. Visit [https://bdsp.io](https://bdsp.io)
2. Sign up and request data access
3. Once approved, download dataset via AWS S3 (see [DATASET_INFO.md](DATASET_INFO.md))

## 3. Fix Annotation Paths (30 seconds)

After downloading data, run this once to prepare annotation files:

```bash
cd code
python fix_annotation_paths.py
```

## 4. Test Installation (1 minute)

```bash
cd code
python test_detectors.py
```

## 5. Run Your First Detection (2 minutes)

### Option A: Run visualization example
```bash
cd code
python visualize_output.py
```
This will process example files and display results.

### Option B: Jupyter notebook
```bash
cd code
jupyter notebook
# Open either irr_analysis_onagreement.ipynb or irr_analysis_fulldataset.ipynb
```

## 6. Process Full Dataset (2-4 hours)

```bash
cd code
python extract_frequency_spatial_extent.py
```

Results will be saved to `results/` directory:
- `gpd_results.csv` - Generalized Periodic Discharges
- `lpd_results.csv` - Lateralized Periodic Discharges
- `grda_results.csv` - Generalized Rhythmic Delta Activity
- `lrda_results.csv` - Lateralized Rhythmic Delta Activity

## Common Commands

### Detect LRDA
```python
import rda_detector as rda
data_obj = rda.rda1b_fft(segment, fs=200, plot=0)
print(f"Frequency: {data_obj['event_frequency']} Hz")
print(f"Spatial extent: {data_obj['spatial_extent']}")
```

### Detect LPD
```python
import pd_detector_alternate as pd
data_obj = pd.pd_detect_alternate(segment, fs=200, pk_detect='apd')
print(f"Frequency: {data_obj['event_frequency']} Hz")
print(f"Spatial extent: {data_obj['spatial_extent']}")
```

### Load EEG file
```python
import hdf5storage
import h5py

def load_mat_file(filepath):
    try:
        return hdf5storage.loadmat(filepath)
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            return {key: f[key][()] for key in f.keys()}

mat = load_mat_file('data/dataset_eeg/lpd/file.mat')
segment = mat['data']  # or mat['data_50sec']
fs = 200
```

## Best Performing Algorithms

| Event Type | Algorithm | Parameter |
|------------|-----------|-----------|
| LRDA/GRDA | `rda.rda1b_fft()` | - |
| LPD/GPD | `pd.pd_detect_alternate()` | `pk_detect='apd'` |

## Output Format

Each detector returns:
```python
{
    'type_event': 'LPD',              # Detected event type
    'event_frequency': 2.1,           # Frequency in Hz
    'spatial_extent': 0.67,           # Proportion (0-1)
    'spatial_areas': ['LF', 'RF'],    # Brain regions
    'channels': [...],                # Channel info
    'peaks': [...]                    # Peak info
}
```

## Brain Regions

- **LF**: Left Frontal
- **RF**: Right Frontal
- **LT**: Left Temporal
- **RT**: Right Temporal
- **LO**: Left Occipital
- **RO**: Right Occipital
- **LCP**: Left Central-Parietal
- **RCP**: Right Central-Parietal

## Need Help?

- **Full documentation**: See [README.md](README.md)
- **Testing guide**: See [TESTING.md](TESTING.md)
- **Data access**: See [DATASET_INFO.md](DATASET_INFO.md)
- **Issues**: Open an issue on [GitHub](https://github.com/bdsp-core/IIIC-Frequency-Analysis-2/issues)

## Directory Structure

```
IIIC-Frequency-Analysis-2/
├── code/                    # Source code
│   ├── pd_detector/         # Periodic discharge detectors
│   ├── rda_detector/        # Rhythmic delta activity detectors
│   ├── imageGeneration/     # Visualization
│   └── *.py, *.ipynb        # Analysis scripts
├── data/                    # Download separately via BDSP
│   ├── dataset_eeg/         # EEG files (.mat)
│   └── annotations/         # Expert annotations (.csv)
├── results/                 # Generated results
└── README.md               # Full documentation
```

## Citation

```bibtex
@article{tautan2025automated,
  title={Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data},
  author={T{\u{a}}u{\c{t}}an, Alexandra-Maria and Jing, Jin and Basovic, Lara and Hadar, Parimala Nallappan and Sartipi, Sahar and Fernandes, Marcos Paulo and Kim, Jonathan and Struck, Aaron F and Westover, M Brandon and Zafar, Sahar F},
  year={2025}
}
```
