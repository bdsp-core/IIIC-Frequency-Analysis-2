from fooof import FOOOF
from joblib import Parallel, delayed
from mne.filter import notch_filter,filter_data
import hdf5storage as hs
import scipy.io as sio
import numpy as np
import pandas as pd
import numpy.matlib
import math
import os
import pdb
from pyhht.emd import EMD
from scipy.signal import hilbert, find_peaks
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from collections import Counter
import zlib

# global var
bipolar_channels=['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2','Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4',
                 'F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz']
mono_channels=['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2','EKG']
fooofgap=np.empty((1,5))
fooofgap[:]=np.nan
fooofgap=fooofgap[0]
thr_bw=0.5
freq_range=[.5,3]


# callbacks 
def fcn_computeSpectra(x,Fs):
    """
    Calculate signal spectrum using Fourier Transform
    Attributes:
        x (numpy array) : input signals
        Fs (numpy) : sampling frequency
    Output:
        psdx (numpy array): power spectrum (non-logged)
    """ 
    N=len(x)
    xdft=np.fft.fft(x)
    xdft=xdft[0:int(N/2)]
    psdx=(1/(Fs*N))*np.square(abs(xdft))
    psdx[1:-2]=2*psdx[1:-2]
    return psdx


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

####################################################################################################
#Hilbert Transform based algorithm
####################################################################################################

def fcn_instantaneous_frequency(imf, t):
        """
        Calculate the instantenous frequency of the intrinsic mode functions using the Hilbert Transform
        Attributes:
            imf (numpy array): intrinsic mode function
            t (numpy array): time vectors for each IMF
        Output:
            instantenous freq array for each IMF
        """ 
        analytic_signal = hilbert(imf)  # Compute the analytic signal using Hilbert transform
        phase = np.angle(analytic_signal)  # Extract the phase of the analytic signal
        freq = np.diff(phase) / (2 * np.pi * np.diff(t))  # Derivative of phase to get frequency
        return np.concatenate(([freq[0]], freq))

def fcn_rdahilbert(seg,fs,channel_filter,debug):
        """
        Apply Hilbert-Huang Transform on the EEG data to identify RDA events
        Attributes:
            seg (numpy array): 18 EEG channels
            fs (numpy) : sampling frequency
            channel filter (boolean) : 0/1 - apply a channel selector based on valid EEG data or noise
            debug (boolean) : 0/1 - whether to print and plot debugging information
        Output:
            rda_freq (numpy array): list of frequency values for the channels with a detected rda event
            channels (numpy array): list of channels where an rda is detected
        """
        rda_f_mean = np.full(len(seg),np.nan)
        t = np.arange(0,seg.shape[1]/fs,1/fs)
        for idx_signal in range(0,len(seg)):
            if debug:
                print(f'Channel {idx_signal}. {bipolar_channels[idx_signal]}')
            x = seg[idx_signal,:]

            if channel_filter:
                sig_range = np.max(abs(x))-np.min(abs(x))
                if np.var(x)>50*sig_range:
                    if debug:
                        print('No signal recorded.')
                    continue
                

            # Initialize EMD and perform the decomposition
            emd = EMD(x)
            IMFs = emd.decompose()

            max_amp_orig = np.max(abs(x))    
            imf_amp = np.empty([len(IMFs)])
            imf_amp_std = np.empty([len(IMFs)])
            imf_amp_perc = np.empty([len(IMFs)]) #percentage from original signal
            imf_freq_m = np.empty([len(IMFs)])
            imf_freq_std = np.empty([len(IMFs)])
            rda_f = np.full([len(IMFs)],np.nan)
            
            
            for i,imf in enumerate(IMFs):
                #print(f'IMF {i+1}')
                freq = fcn_instantaneous_frequency(imf, t)
                peaks, _ = find_peaks(abs(freq),np.mean(freq)+2*np.std(freq))
                
                imf_amp[i] = np.max(imf)
                imf_amp_std[i] = np.std(imf)
                imf_freq_m[i] = np.mean(fs/np.diff(peaks))
                imf_freq_std[i] = np.std(fs/np.diff(peaks))
                imf_amp_perc[i] = (imf_amp[i]*100)/max_amp_orig

                if (imf_freq_m[i] > 0.5) & (imf_freq_m[i] < 4):
                    if imf_freq_m[i] >= 2*imf_freq_std[i]:
                        if imf_amp[i] < 2*imf_amp_std[i]:
                            continue
                        elif imf_amp_perc[i] < 25:    
                            continue
                        else:
                            if debug:
                                print(f'IMF {i+1}')
                                #print('*'*10)
                                #print('RDA!')
                                #print('*'*10)
                                print(f'Amplitude {imf_amp[i]} with std {imf_amp_std[i]}')
                                print(f'Frequency {imf_freq_m[i]} with std {imf_freq_std[i]}')
                            rda_f[i] = imf_freq_m[i]
            
            rda_f_mean[idx_signal] = np.mean(rda_f[~np.isnan(rda_f)])
            
            if debug:
                if ~np.isnan(rda_f_mean[idx_signal]):
                    print(f'Channel has rda of freq {rda_f_mean[idx_signal]}')
                print("#"*10)
            
                plt.figure(figsize=(12, 8))

                # Plot the original signal
                plt.subplot(len(IMFs)+1, 3, 1)
                plt.plot(t, x, label="Original Signal")
                plt.title("Original Signal")

                # Plot the original signal fft
                plt.subplot(len(IMFs)+1, 3, 3)
                psdx = fcn_computeSpectra(x,fs)
                freqs = np.fft.fftfreq(len(x),1/fs)
                freqs = freqs[:len(freqs) //2]
                plt.plot(freqs, psdx, label="FFT of Original Signal")
                plt.title("FFT of Original Signal")
                plt.xlim([0,10])

                s_idx = 4
                for i, imf in enumerate(IMFs):
                    
                    # Plot each IMF
                    plt.subplot(len(IMFs)+1, 3, s_idx)
                    
                    plt.plot(t, imf, label=f'IMF {i+1}')
                    plt.title(f'IMF {i+1}')
                    
                    # Calculate and plot the instantaneous frequency for each IMF
                    freq = fcn_instantaneous_frequency(imf, t)
                    plt.subplot(len(IMFs) + 1, 3,  s_idx+1)
                    plt.plot(t, freq, label=f'IF of IMF {i+1}', color='orange')
                    plt.title(f'Instantaneous Frequency of IMF {i+1}')
            

                    #plot fft of the IMFs
                    psdx = fcn_computeSpectra(imf,fs)
                    freqs = np.fft.fftfreq(len(imf),1/fs)
                    freqs = freqs[:len(freqs) //2]
                    plt.subplot(len(IMFs) + 1, 3,  s_idx+2)
                    plt.plot(freqs, psdx, label=f'FFT of IMF {i+1}', color='orange')
                    plt.title(f'FFT of IMF {i+1}')
                    plt.xlim([0,10])


                    s_idx+=3
                plt.suptitle(f'Channel {idx_signal}. {bipolar_channels[idx_signal]}', fontsize=16)
        plt.show()

                
        rda_freq = rda_f_mean[~np.isnan(rda_f_mean)]
        channels =np.array(bipolar_channels)[~np.isnan(rda_f_mean)]

        return rda_freq, channels

def rda2_hht(segment,fs,channel_filter):
    """
    Calculate frequency of event, spatial extent and spatial areas for RDA using the algorithm rda2_hht
    Attributes:
        segment (numpy array): 18 EEG channel segments
        fs (numpy) : sampling frequency
        channel filter (boolean) : 0/1 - apply a channel selector based on valid EEG data or noise
    Output:
        data_obj (panda series): 
            type_event: LRDA or GRDA based on the spatial extent value
            event_frequency: median frequency of events across all detected channels
            spatial_extent: 0 - no channels show rda, 1 - all channels show rda
            spatial_areas: spatial areas identified based on the channels selected - LF, RF, LCP, RCP, RT, LT, LO, RO
            channels: number of channels detected to have RDA events
    """ 
    # filters to denoise
    segment=notch_filter(segment,fs,60,n_jobs=-1,verbose="ERROR")
    segment=filter_data(segment,fs,0.5,40,n_jobs=-1,verbose="ERROR")

    # L-bipolar
    segment=fcn_getBanana(segment)
    segment=np.array(segment)

    seg=segment#[:16,:]
    
    rda_freq, channels = fcn_rdahilbert(seg,fs,channel_filter,0)

    spatial_areas = []
                  
    if len(channels)==0:
        data_obj = {
            "type_event":np.nan,
            "event_frequency": np.nan,
            "spatial_extent": np.nan,
            "spatial_areas":np.nan,
            "channels": np.nan,
        }               
    else:
        for channel in channels:
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

        if len(channels)/seg.shape[0] > 0.8:
            type_event = "GRDA"
        else:
            type_event = "LRDA"

        data_obj = {
            "type_event":type_event,
            "event_frequency": np.median(rda_freq),
            "spatial_extent": len(channels)/seg.shape[0],
            "spatial_areas":spatial_areas,
            "channels": pd.Series(channels),
        }

    return data_obj


