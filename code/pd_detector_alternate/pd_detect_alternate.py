from mne.filter import notch_filter,filter_data
import hdf5storage 
import scipy.io as sio
import numpy as np
import pandas as pd
import warnings
import numpy
import math
import os
import pdb
import scipy.io as io
from scipy.signal import find_peaks, detrend
from scipy.stats import shapiro
from scipy import signal
from scipy.signal import savgol_filter


# global var
bipolar_channels=['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2','Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4',
                 'F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz']
mono_channels=['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2','EKG']

freq_range=[.5,4]

def fcn_getBanana(X):
    """
    Apply a differential longitudinal montage to EEG data
    Attributes:
        X (numpy array): 19 raw EEG channels
    Output:
        bipolar_data (numpy array): re-referenced EEG data
    """  
    bipolar_ids = np.array([[mono_channels.index(bc.split('-')[0]),mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = X[bipolar_ids[:,0]]-X[bipolar_ids[:,1]]
    return bipolar_data


def zscore_peak_detection(signal, window_size=200, z_threshold=2):
    """
    Detection of peaks from signal using cwt
    Attributes:
        signal (numpy array): signal data
        window_size (numpy) : window size for detecting peaks - default 200 samples
        z_threshold (numpy) : threshold for detecting peaks using zscore
    Output:
        peaks (numpy array): location of detected peaks
    """  
    # Calculate moving z-score
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(signal, (window_size//2, window_size//2), mode='reflect'), 
        window_size
    )
    
    local_mean = np.mean(windows, axis=1)[:len(signal)]
    local_std = np.std(windows, axis=1)[:len(signal)]
    
    # Calculate z-score
    z_scores = (signal - local_mean) / local_std
    
    # Find peaks based on z-score threshold
    peaks, _ = find_peaks(z_scores, height=z_threshold)
    
    return peaks

def cwt_peak_detection(data,fs, widths=np.arange(1, 64)):
    """
    Detection of peaks from signal using cwt
    Attributes:
        data (numpy array): signal data
        widths (numpy) : array for cwt size
    Output:
        peaks (numpy area): location of detected peaks
    """   
    cwt_matrix = signal.cwt(data, signal.ricker, widths)
    
    # Sum across scales to get ridge lines
    ridge_sum = np.sum(cwt_matrix, axis=0)
    
    # Find peaks in the ridge sum
    peaks, _ = find_peaks(ridge_sum, distance=(1/5)*fs,prominence = 1)
    
    return peaks

def adaptive_peak_detection(signal, window_size=200, n_std=3):
    """
    Detection of peaks from signal
    Attributes:
        signal (numpy array): signal data
        window_size (numpy) : window size for detecting peaks - default 200 samples
        n_std (numpy): threshold for standard deviation
    Output:
        peak indices (numpy area): location of detected peaks
    """   
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(signal, (window_size//2, window_size//2), mode='reflect'), 
        window_size
    )
    
    local_mean = np.mean(windows, axis=1)[:len(signal)]
    local_std = np.std(windows, axis=1)[:len(signal)]
    
    # Adaptive threshold based on local statistics
    threshold = local_mean + n_std * local_std
    
    # Find peaks that exceed local threshold
    peak_indices = []
    for i in range(len(signal)):
        if i > 0 and i < len(signal)-1:
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold[i]:
                peak_indices.append(i)
    
    return np.array(peak_indices)


def pd_detect_alternate(segment,fs,pk_detect='apd'):
    """
    Provides the frequency of events, spatial extent and spatial area of a PD marked EEG segment.
    Attributes:
        segment (numpy array): 10s raw EEG segment containing 19 channels
        fs (numpy): sampling frequency
        pk_detect (str): select the type of peak detector function applied on the second derivative
                            - apd - adaptive peak detector
                            - cwt - continuous wavelet transform peak detector
                            - zscore - z-score based peak detector
    Output:
        data_obj (data struct): 
                type_event: LPD or GPD based on the spatial extent
                event_frequency: frequency of event as a median over all channel frequencies
                spatial_extent: 0-1, 0 - no channel contains an event, 1 - all channels contain an event
                spatial_areas: brain areas that show events based on the channels identified: LF, RF, LCP, RCP, LT, RT, LO, RO
                channels: differential channels detected to have events
                peaks: location of the peaks 
    """

    # filters to denoise
    segment=notch_filter(segment,fs,60,n_jobs=-1,verbose="ERROR")
    segment=filter_data(segment,fs,0.5,40,n_jobs=-1,verbose="ERROR")

    # L-bipolar
    segment=fcn_getBanana(segment)
    segment=np.array(segment)

    seg=segment
    df_peaks = pd.DataFrame(index=range(len(seg)),columns=['peaks','intervals','frequency'])

    for i in range(0,seg.shape[0]):
        x = seg[i,:]
        #skip if no eeg signal detected on channel
        sig_range = np.max(abs(x))-np.min(abs(x))
        if np.var(x)>50*sig_range:
            continue

        x = detrend(x - np.mean(x))
        x = savgol_filter(x, window_length=10, polyorder=2)  # Adjust parameters
        d_x = np.diff(x)

        if pk_detect=='apd':
            d_peaks = adaptive_peak_detection(d_x)
        elif pk_detect=='cwt':
            d_peaks = cwt_peak_detection(d_x,fs)
        elif pk_detect=='zscore':
            d_peaks = zscore_peak_detection(d_x)

        intervals = np.diff(d_peaks/fs)
        if (d_peaks.shape[0]>1) & (intervals.shape[0]>1) & (np.std(intervals, ddof=1) < 1):
            df_peaks.loc[i,'peaks'] = d_peaks
            df_peaks.loc[i,'intervals'] = intervals
            df_peaks.loc[i,'frequency'] = 1/np.mean(intervals)


    df_peaks['chan'] = bipolar_channels
    df_peaks = df_peaks[~df_peaks['frequency'].isna()]
    spatial_areas = []
    if len(df_peaks)==0:
        data_obj = {
            "type_event":np.nan,
            "event_frequency": np.nan,
            "spatial_extent": np.nan,
            "spatial_areas":np.nan,
            "channels": np.nan,
            "peaks":np.nan

        }               
    else:
        for channel in df_peaks['chan']:
            if channel in ['Fp1-F7','Fp1-F3','F3-C3','F7-T3']:
                spatial_areas.append('LF')
            if channel in ['Fp2-F8','F8-T4','Fp2-F4','F4-C4']:
                spatial_areas.append('RF')
            if channel in ['C3-P3']:
                spatial_areas.append('LCP')
            if channel in ['C4-P4']:
                spatial_areas.append('RCP')
            if channel in ['T3-T5','T5-O1']:
                spatial_areas.append('LT')
            if channel in ['T4-T6','T6-O2']:
                spatial_areas.append('RT')
            if channel in ['P3-O1']:
                spatial_areas.append('LO')
            if channel in ['P4-O2']:
                spatial_areas.append('RO')

            spatial_areas = list(set(spatial_areas))

            if len(df_peaks['chan'])/seg.shape[0] > 0.8:
                type_event = "GPD"
            else:
                type_event = "LPD"

            data_obj = {
                "type_event":type_event,
                "event_frequency": df_peaks['frequency'].median(),
                "spatial_extent": df_peaks.shape[0]/18,
                "spatial_areas":spatial_areas,
                "channels": pd.Series(df_peaks['chan']),
                "peaks":df_peaks['peaks']
                }


    return data_obj