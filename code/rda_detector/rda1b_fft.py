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
from scipy.signal import find_peaks
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
freq_range=[.5,4]

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
    x = x-np.mean(x)
    N=len(x)
    #xdft=np.fft.fft(x)
    #xdft=xdft[0:int(N/2)]
    #psdx=(1/(Fs*N))*np.square(abs(xdft))
    #psdx[1:-2]=2*psdx[1:-2]

    xdft = np.fft.fft(x)
    xdft = xdft[:len(xdft) //2]
    psdx = (1/(Fs*N))*np.square(np.abs(xdft))

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

def fcn_rdafooof_enhanced(seg,freqs, Fs,channel_filter):
    """
    Apply fooof model on the channel spectra
    Attributes:
        seg (numpy array): 18 EEG channels
        freqs (numpy array): 19 spectra of EEG channels
        Fs (numpy) : sampling frequency
        channel filter (boolean) : 0/1 - apply a channel selector based on valid EEG data or noise
    Output:
        fooof_ (numpy array): peaks, location and bw of frequency peaks that meet rda condition
        channels (numpy array): list of channels where an rda is detected
        spectra (numpy array): list of spectra for channels where an rda is detected
    """
    fooof=[]
    count = 0
    spectra = np.zeros([len(seg),int(seg.shape[1]/2)])
    for k in range(0,len(seg)):

        if channel_filter:
            x = seg[k]
            sig_range = np.max(abs(x))-np.min(abs(x))
            if np.var(x)>50*sig_range:
                continue


        spectrum=fcn_computeSpectra(seg[k],Fs)
        spectra[count,:] = spectrum
        #spectrum=s[0]
        #spectrum=s
        fm = FOOOF()
        freqs = np.array(freqs)
        spectrum = np.array(spectrum)
        
        try:
            fm.fit(freqs, spectrum, freq_range)
        except Exception as e:
            print(f"Error fitting FOOOF model for segment {k+1}: {e}")
            return 69420, 69420, 69420

        
        tmp=fm.peak_params_
        
        if len(tmp)==0:
            x=fooofgap
            fooof.append(x)
        else:
            x=np.append(np.append(tmp[0],fm.error_),fm.r_squared_)
            fooof.append(x)
        count+=1
    fooof=np.array(fooof)
    bw=np.round(100*fooof[:,2])/100 
    
    
    idx=np.where(bw<=thr_bw)
    idx2 = []
    pks = fooof[:,0]
    bw = fooof[:,2]
    
    for pk_idx,pk in enumerate(pks):
        #print('Channel :'+str(pk_idx))
        #print(pk)
        closest_value = min(freqs, key=lambda x: abs(x-pk))
        #print(closest_value)
        
        #select the indexes corresponding to the detected peak as well as the bandwidth identified around it - this all would represent the freq content of interest
        select_pk_bw = (freqs > closest_value-(bw[pk_idx]/2)) & (freqs < closest_value+(bw[pk_idx]/2))

        #select the rest of the signal in the delta band [0.5, 3] Hz and everything outside the peak/content of interest - calculate the mean
        condition_proeminence = np.mean(spectra[pk_idx,(freqs<=3) & (freqs>=0.5) & ~select_pk_bw])#+np.std(spectra[pk_idx,(freqs<3)  & (freqs>0.5)])
        #print(condition_proeminence)
        #print(np.mean(spectra[pk_idx,(freqs > closest_value-(bw[pk_idx]/2)) & (freqs < closest_value+(bw[pk_idx]/2))]))
        #print('#'*25)

        if np.mean(spectra[pk_idx,select_pk_bw]) > condition_proeminence:
            idx2.append(pk_idx)

    if len(idx2)==0:
        return 69420, 69420, 69420
    
    idx2 = (np.array(idx2),)
    #pdb.set_trace()
    
    #print(idx2)
    fooof_=fooof[idx2]


    channels_=np.array(bipolar_channels)[idx2[0]]

    return fooof_,channels_,spectra


def rda1b_fft(segment,fs,channel_filter):
    """
    Calculate frequency of event, spatial extent and spatial areas for RDA using the algorithm rda1_fft
    Attributes:
        segment (numpy array): 18 EEG channel segments
        fs (numpy) : sampling frequency
        channel filter (boolean) : 0/1 - apply a channel selector based on valid EEG data or noise
    Output:
        data_obj (panda series): 
            type_event: LRDA or GRDA based on the spatial extent value
            event_frequency: median frequency of events across all detected channels
            power: median event power
            bandwidth: median peak bandwidth of identified peaks
            fit_error: median fooof fit error across channels
            r_squared: median r-squared for fooof model
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
    
    #N=len(seg[0])     
    #freqs1=np.arange(0,fs/2,fs/N) 
    freqs = np.fft.fftfreq(seg.shape[1],1/fs)
    freqs = freqs[:len(freqs) //2]
    #pdb.set_trace()
    fooof,channels,spectra=fcn_rdafooof_enhanced(seg,freqs, fs,channel_filter)

    spatial_areas = []
    if np.any(fooof == 69420) or np.any(channels == 69420):
        data_obj = {
            "type_event":np.nan,
            "event_frequency": np.nan,
            "power": np.nan,
            "bandwidth": np.nan,
            "fit_error": np.nan,
            "r_squared": np.nan,
            "spatial_extent": np.nan,
            "spatial_areas":np.nan,
            "channels": np.nan
        }    
    else:
        
        #if sparcnet_score==4: # LRDA
            
        #    if len(channels)!=0:
        #        scr=np.square(fooof[:,1])*fooof[:,4]/(fooof[:,2]*fooof[:,3]) 
        #        idx=np.argsort(scr) 
        #        if len(idx)>1:
        #            fooof=fooof[idx[[-1,-2]]]
        #            channels=channels[idx[[-1,-2]]]
                                
        if len(channels)==0:
            data_obj = {
                "type_event":np.nan,
                "event_frequency": np.nan,
                "power": np.nan,
                "bandwidth": np.nan,
                "fit_error": np.nan,
                "r_squared": np.nan,
                "spatial_extent": np.nan,
                "spatial_areas":np.nan,
                "channels": np.nan
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
                "event_frequency": np.median(fooof[:,0]),
                "power": np.median(fooof[:,1]),
                "bandwidth": np.median(fooof[:,2]),
                "fit_error": np.median(fooof[:,3]),
                "r_squared": np.median(fooof[:,4]),
                "spatial_extent": len(channels)/seg.shape[0],
                "spatial_areas":spatial_areas,
                "channels": pd.Series(channels),
                }

    return data_obj,spectra, freqs

