import pandas as pd
import pdb
import os
import warnings
import hdf5storage
import h5py
from pathlib import Path

import rda_detector as rda
import pd_detector_alternate as pd_detect
import imageGeneration as im

import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

def load_mat_file(filepath):
    """Load MATLAB file, handling both v7.3 and earlier versions."""
    try:
        return hdf5storage.loadmat(filepath)
    except NotImplementedError as e:
        if 'HDF reader for matlab v7.3 files' in str(e):
            with h5py.File(filepath,'r') as f:
                return {key: f[key][()] for key in f.keys()}
        else:
            raise

# Robust path handling - works whether running from code/ or repo root
script_dir = Path(__file__).parent
repo_root = script_dir.parent if script_dir.name == 'code' else script_dir

# Define paths
data_dir = repo_root / 'data' / 'dataset_eeg'

# Verify data directory exists
if not data_dir.exists():
    raise FileNotFoundError(
        f"Data directory not found: {data_dir}\n"
        f"Please download the dataset from BDSP and place it in {repo_root / 'data'}\n"
        f"See DATASET_INFO.md for instructions."
    )

#############################################################
#RDA example                                                #
#############################################################
print("="*70)
print("RDA Detection Example")
print("="*70)

# Load data - use first available LRDA file
lrda_dir = data_dir / 'lrda'
if not lrda_dir.exists():
    raise FileNotFoundError(f"LRDA directory not found: {lrda_dir}")

# Find first .mat file
lrda_files = sorted(list(lrda_dir.glob('*.mat')))
if not lrda_files:
    raise FileNotFoundError(f"No .mat files found in {lrda_dir}")

file_rda = lrda_files[0]
print(f"Loading file: {file_rda.name}")


try:
    mat = load_mat_file(str(file_rda))
except Exception as e:
    print(f"Failed to load {file_rda}: {e}")
    raise

try:
    segment = mat['data']
except KeyError:
    segment = mat['data_50sec']

fs = 200

# Calculate frequency and spatial extent using best-performing RDA detector
print("Running rda2_hht detector...")
data_obj = rda.rda2_hht(segment, fs, 1)

print(f"\nResults:")
print(f"  Event type: {data_obj['type_event']}")
print(f"  Frequency: {data_obj['event_frequency']:.2f} Hz")
print(f"  Spatial extent: {data_obj['spatial_extent']:.2f}")
print(f"  Spatial areas: {data_obj['spatial_areas']}")

# View signals and output
print("\nGenerating visualization...")
fig = im.plot_rda_events(segment, data_obj, 0, int(segment.shape[1]/fs), fs)
plt.show()

#############################################################
#PD example                                                 #
#############################################################
print("\n" + "="*70)
print("PD Detection Example")
print("="*70)

# Load data - use first available LPD file
lpd_dir = data_dir / 'lpd'
if not lpd_dir.exists():
    raise FileNotFoundError(f"LPD directory not found: {lpd_dir}")

# Find first .mat file
lpd_files = sorted(list(lpd_dir.glob('*.mat')))
if not lpd_files:
    raise FileNotFoundError(f"No .mat files found in {lpd_dir}")

file_pd = lpd_files[0]
print(f"Loading file: {file_pd.name}")

try:
    mat = load_mat_file(str(file_pd))
except Exception as e:
    print(f"Failed to load {file_pd}: {e}")
    raise

try:
    segment = mat['data']
except KeyError:
    segment = mat['data_50sec']

fs = 200

# Run best-performing PD detector
print("Running pd_detect_alternate with APD...")
try:
    data_obj = pd_detect.pd_detect_alternate(segment, fs, pk_detect='apd')

    print(f"\nResults:")
    print(f"  Event type: {data_obj['type_event']}")
    print(f"  Frequency: {data_obj['event_frequency']:.2f} Hz")
    print(f"  Spatial extent: {data_obj['spatial_extent']:.2f}")
    print(f"  Spatial areas: {data_obj['spatial_areas']}")

    print("\nGenerating visualization...")
    fig = im.plot_pd_events(segment, data_obj, 0, int(segment.shape[1]/fs), fs)
    plt.show()

except Exception as e:
    print(f'Error at pd detector: {file_pd.name}')
    print(f'Error details: {e}')
    raise

print("\n" + "="*70)
print("Visualization complete!")
print("="*70)


