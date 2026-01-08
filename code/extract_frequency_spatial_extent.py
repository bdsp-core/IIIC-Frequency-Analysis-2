import pandas as pd
import pdb
import numpy as np
import warnings
import hdf5storage
import h5py
import rda_detector as rda
import pd_detector as pddet
import pd_detector_alternate as pddeta
import imageGeneration as im
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm

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


# Configuration
events_rda = ['lrda','grda']
events_pd = ['gpd','lpd']

# Robust path handling - works whether running from code/ or repo root
script_dir = Path(__file__).parent
repo_root = script_dir.parent if script_dir.name == 'code' else script_dir

# Define paths
data_dir = repo_root / 'data' / 'dataset_eeg'
results_dir = repo_root / 'results'

# Create results directory if it doesn't exist
results_dir.mkdir(parents=True, exist_ok=True)

# Verify data directory exists
if not data_dir.exists():
    raise FileNotFoundError(
        f"Data directory not found: {data_dir}\n"
        f"Please download the dataset from BDSP and place it in {repo_root / 'data'}\n"
        f"See DATASET_INFO.md for instructions."
    )


for event in events_rda:
    print('*'*70)
    print(event.upper())
    print('*'*70)

    event_dir = data_dir / event
    if not event_dir.exists():
        print(f"Warning: Directory {event_dir} not found. Skipping {event}.")
        continue

    files = []

    event_type_rda2_hhtt = []
    freq_rda2_hhtt = []
    spatial_rda2_hhtt = []
    spatial_areas_rda2_hhtt = []

    event_type_rda1b_fft = []
    freq_rda1b_fft = []
    spatial_rda1b_fft = []
    spatial_areas_rda1b_fft = []

    #event_type_rda1b_fft_chan = []
    #freq_rda1b_fft_chan = []
    #spatial_rda1b_fft_chan = []
    #spatial_areas_rda1b_fft_chan = []

    event_type_rda1a_fft = []
    freq_rda1a_fft = []
    spatial_rda1a_fft = []
    spatial_areas_rda1a_fft = []

    # Get all .mat files in directory
    files_4analysis = sorted([f for f in event_dir.iterdir() if f.suffix == '.mat'])

    for file_path in tqdm(files_4analysis, total=len(files_4analysis), desc=f"Processing {event.upper()}"):
        filename = file_path.name
        fs = 200

        try:
            mat = load_mat_file(str(file_path))
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        try:
            segment = mat['data_50sec']
        except (KeyError, Exception):
            segment = mat['data']

        data_obj_rda2_hhtt = rda.rda2_hht(segment,fs,1)
        data_obj_rda1b_fft,spectra, freqs = rda.rda1b_fft(segment,fs,0)
        #data_obj_rda1b_fft_chan,spectra, freqs = rda.rda_detector_enhanced(segment,fs,1)
        data_obj_rda1a_fft = rda.rda1a_fft(segment,fs)

        
        #im.plot_rda_events(segment, data_obj_rda2_hhtt,0,int(segment.shape[1]/fs),fs)
        #plt.savefig(path_out+'results_figures/'+event+'/'+filename[:-4]+'_rda2_hhtt.png', bbox_inches='tight', pad_inches=0.1)

        #im.plot_rda_events(segment, data_obj_rda1b_fft,0,int(segment.shape[1]/fs),fs)
        #plt.savefig(path_out+'results_figures/'+event+'/'+filename[:-4]+'_rda1b_fft.png', bbox_inches='tight', pad_inches=0.1)

        #im.plot_rda_events(segment, data_obj_rda1b_fft_chan,0,int(segment.shape[1]/fs),fs)
        #plt.savefig(path_out+'results_figures/'+event+'/'+filename[:-4]+'_rda1b_fft_chan.png', bbox_inches='tight', pad_inches=0.1)

        #im.plot_rda_events(segment, data_obj_rda1a_fft,0,int(segment.shape[1]/fs),fs)
        #plt.savefig(path_out+'results_figures/'+event+'/'+filename[:-4]+'_rda1a_fft.png', bbox_inches='tight', pad_inches=0.1)

        #plt.close('all')
        files.append(file_path.stem)  # filename without extension

        event_type_rda2_hhtt.append(data_obj_rda2_hhtt['type_event'])
        freq_rda2_hhtt.append(data_obj_rda2_hhtt['event_frequency'])
        spatial_rda2_hhtt.append(data_obj_rda2_hhtt['spatial_extent'])
        spatial_areas_rda2_hhtt.append(data_obj_rda2_hhtt['spatial_areas'])

        event_type_rda1b_fft.append(data_obj_rda1b_fft['type_event'])
        freq_rda1b_fft.append(data_obj_rda1b_fft['event_frequency'])
        spatial_rda1b_fft.append(data_obj_rda1b_fft['spatial_extent'])
        spatial_areas_rda1b_fft.append(data_obj_rda1b_fft['spatial_areas'])

        #event_type_rda1b_fft_chan.append(data_obj_rda1b_fft_chan['type_event'])
        #freq_rda1b_fft_chan.append(data_obj_rda1b_fft_chan['event_frequency'])
        #spatial_rda1b_fft_chan.append(data_obj_rda1b_fft_chan['spatial_extent'])
        #spatial_areas_rda1b_fft_chan.append(data_obj_rda1b_fft_chan['spatial_areas'])

        event_type_rda1a_fft.append(data_obj_rda1a_fft['type_event'])
        freq_rda1a_fft.append(data_obj_rda1a_fft['event_frequency'])
        spatial_rda1a_fft.append(data_obj_rda1a_fft['spatial_extent'])
        spatial_areas_rda1a_fft.append(data_obj_rda1a_fft['spatial_areas'])

        collected ={
            'files':files,
            'event_type_rda2_hhtt':event_type_rda2_hhtt,
            'freq_rda2_hhtt':freq_rda2_hhtt,
            'spatial_rda2_hhtt':spatial_rda2_hhtt,
            'spatial_areas_rda2_hhtt':spatial_areas_rda2_hhtt,

            'event_type_rda1b_fft':event_type_rda1b_fft,
            'freq_rda1b_fft':freq_rda1b_fft,
            'spatial_rda1b_fft':spatial_rda1b_fft,
            'spatial_areas_rda1b_fft':spatial_areas_rda1b_fft,

            #'event_type_rda1b_fft_chan':event_type_rda1b_fft_chan,
            #'freq_rda1b_fft_chan':freq_rda1b_fft_chan,
            #'spatial_rda1b_fft_chan':spatial_rda1b_fft_chan,
            #'spatial_areas_rda1b_fft_chan':spatial_areas_rda1b_fft_chan,

            'event_type_rda1a_fft':event_type_rda1a_fft,
            'freq_rda1a_fft':freq_rda1a_fft,
            'spatial_rda1a_fft':spatial_rda1a_fft,
            'spatial_areas_rda1a_fft':spatial_areas_rda1a_fft
        }

        df_results = pd.DataFrame(collected)
        output_file = results_dir / f'{event}_results.csv'
        df_results.to_csv(output_file, index=False)

    print(f"Results saved to: {results_dir / f'{event}_results.csv'}")



for event in events_pd:
    print('*'*70)
    print(event.upper())
    print('*'*70)

    event_dir = data_dir / event
    if not event_dir.exists():
        print(f"Warning: Directory {event_dir} not found. Skipping {event}.")
        continue

    files = []

    event_type = []
    freq = []
    spatial = []
    spatial_areas = []


    event_type_apd = []
    freq_apd = []
    spatial_apd = []
    spatial_areas_apd = []


    event_type_zscore = []
    freq_zscore = []
    spatial_zscore = []
    spatial_areas_zscore = []

    # Get all .mat files in directory
    files_4analysis = sorted([f for f in event_dir.iterdir() if f.suffix == '.mat'])

    for file_path in tqdm(files_4analysis, total=len(files_4analysis), desc=f"Processing {event.upper()}"):
        filename = file_path.name
        fs = 200

        try:
            mat = load_mat_file(str(file_path))
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        try:
            segment = mat['data_50sec']
        except (KeyError, Exception):
            segment = mat['data']


        #data_obj_alternate = pddeta.pd_detect_alternate(segment,fs)
        data_obj_apd = pddeta.pd_detect_alternate(segment,fs,pk_detect='apd')
        data_obj_zscore = pddeta.pd_detect_alternate(segment,fs,pk_detect='zscore')
        
        try:
            data_obj = pddet.pd_detect(segment)
            
        except Exception as e:
            print(f'Error at pd detector: {filename}')
            data_obj = {
            "type_event":np.nan,
            "event_frequency": np.nan,
            "spatial_extent": np.nan,
            "spatial_areas":np.nan,
            "channels": np.nan,
            "peaks":np.nan
            }   

        #fig = im.plot_pd_events(segment, data_obj,0,int(segment.shape[1]/fs),fs)
        #plt.savefig(path_out+'results_figures/'+event+'/'+filename[:-4]+'.png', bbox_inches='tight', pad_inches=0.1)

        #plt.close('all')

        files.append(file_path.stem)  # filename without extension

        event_type.append(data_obj['type_event'])
        freq.append(data_obj['event_frequency'])
        spatial.append(data_obj['spatial_extent'])
        spatial_areas.append(data_obj['spatial_areas'])

        #event_type_alternate.append(data_obj_alternate['type_event'])
        #freq_alternate.append(data_obj_alternate['event_frequency'])
        #spatial_alternate.append(data_obj_alternate['spatial_extent'])
        #spatial_areas_alternate.append(data_obj_alternate['spatial_areas'])
        event_type_apd.append(data_obj_apd['type_event'])
        freq_apd.append(data_obj_apd['event_frequency'])
        spatial_apd.append(data_obj_apd['spatial_extent'])
        spatial_areas_apd.append(data_obj_apd['spatial_areas'])

        
        event_type_zscore.append(data_obj_zscore['type_event'])
        freq_zscore.append(data_obj_zscore['event_frequency'])
        spatial_zscore.append(data_obj_zscore['spatial_extent'])
        spatial_areas_zscore.append(data_obj_zscore['spatial_areas'])


        collected ={
            'files':files,
            'event_type':event_type,
            'freq':freq,
            'spatial':spatial,
            'spatial_areas':spatial_areas,

            'event_type_apd':event_type_apd,
            'freq_apd':freq_apd,
            'spatial_apd':spatial_apd,
            'spatial_areas_apd':spatial_areas_apd,

            'event_type_zscore':event_type_zscore,
            'freq_zscore':freq_zscore,
            'spatial_zscore':spatial_zscore,
            'spatial_areas_zscore':spatial_areas_zscore,
        }

        df_results = pd.DataFrame(collected)
        output_file = results_dir / f'{event}_results.csv'
        df_results.to_csv(output_file, index=False)

    print(f"Results saved to: {results_dir / f'{event}_results.csv'}")

print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)
print(f"\nAll results saved to: {results_dir}")
print("\nGenerated files:")
for event in events_rda + events_pd:
    output_file = results_dir / f'{event}_results.csv'
    if output_file.exists():
        print(f"  - {output_file.name}")

