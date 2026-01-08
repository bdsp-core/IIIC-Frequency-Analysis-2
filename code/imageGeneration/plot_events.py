
import pandas as pd
import numpy as np
import time
import pdb

import pd_detector as pd_detect

import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from mne.filter import notch_filter,filter_data
from mne.viz import plot_topomap
import mne




def plot_rda_events(segment,data_obj,start_sec,end_sec,fs): 
    segment=notch_filter(segment,fs,60,n_jobs=-1,verbose="ERROR")
    segment=filter_data(segment,fs,0.5,40,n_jobs=-1,verbose="ERROR")

    gs = GridSpec(18,2, width_ratios=[100,1],hspace=0.1)

    #plot for first 10s
    #fig, axes = plt.subplots(18, 1, figsize=(14, 8.5), sharex=True)

    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)

    time1 = np.linspace(start_sec,end_sec,(end_sec-start_sec)*fs)
    chan_list = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2','Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4',
                 'F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz']

    seg_bi = pd_detect.utils.l_bipolar(segment)  
    

    ax_list = []

    for i in range(0,seg_bi.shape[0]): #range(0,data_obj['channels'].shape[0]):
        ax = fig.add_subplot(gs[i,0], frame_on=False)

        if np.isin(data_obj['channels'], chan_list[i]).any(): #data_obj['channels'].index(chan_list[i]):
            cl = 'b'
        else:
            cl = 'k'

        ax.plot(time1,seg_bi[i,:],cl)
        ax.set_ylabel(chan_list[i],fontsize=8, rotation=360, labelpad=20)
        ax.tick_params(axis='y', labelsize=5)
        
        if i != seg_bi.shape[0]-1:
            ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for line in ax.get_xgridlines():
            line.set_color('red')
            line.set_linestyle('--')
        for line in ax.get_ygridlines():
            line.set_color('grey')

        current_pos = ax.get_position()
        ax.set_position([0.06,current_pos.y0,current_pos.width+0.09,current_pos.height])
        ax_list.append(ax)
    
    
    #gs_nested = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1], hspace=0.3)

    #topo_ax = fig.add_subplot(gs_nested[1,0])
    #plot_top_freq(data_obj,fig,topo_ax,"blue")
    #topo_ax.set_aspect('auto')
    #topo_ax.set_position([0.79, 0.15, 0.18, 0.3])

    #txt = 'RDA Algorithm Label: \n' + '$\mathbf {' +str(data_obj['type_event']) +'}$' + \
    #    '\n\n\nFrequency: $\mathbf{' + str(round(data_obj['event_frequency'],3)) + '}$ \nSpatial Extent: $\mathbf{' + str(round(data_obj['spatial_extent'],3)) + \
    #     '}$ \nAreas: \n' + str(data_obj['spatial_areas']) + '\n\n\nFilters: notch, 0.5-40Hz'
    #txt = '\n\n\nFrequency: $\mathbf{' + str(round(data_obj['event_frequency'],3)) + '}$ \nSpatial Extent: $\mathbf{' + str(round(data_obj['spatial_extent'],3)) + \
    #     '}$ \nAreas: \n' + str(data_obj['spatial_areas']) + '\n\n\nFilters: notch, 0.5-40Hz'
    txt = '\n\n\nFrequency: $\mathbf{' + str(round(data_obj['event_frequency'],3)) + '}$ \nSpatial Extent: $\mathbf{' + str(round(data_obj['spatial_extent'],3)) + '}$\n\n\nFilters: notch, 0.5-40Hz'

    text_ax = fig.add_subplot(gs[0,1])
    text_ax.text(-0.5, 0.5,txt,fontsize=11, ha='left', va='center')
    text_ax.axis('off')
    text_ax.set_position([0.97, 0.5, 0.2, 0.4])

    #plt.subplots_adjust(left=0.1, right=0.9, top=0.1, bottom=0.1)
    
    return fig


def plot_pd_events(segment,data_obj,start_sec,end_sec,fs): 

    
    segment=notch_filter(segment,fs,60,n_jobs=-1,verbose="ERROR")
    segment=filter_data(segment,fs,0.5,40,n_jobs=-1,verbose="ERROR")

    gs = GridSpec(18,2, width_ratios=[100,1],hspace=0.01)

    #plot for first 10s
    #fig, axes = plt.subplots(18, 1, figsize=(14, 8.5), sharex=True)

    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)

    time = np.linspace(start_sec,end_sec,segment.shape[1])
    time1 = np.linspace(start_sec,end_sec,np.diff(segment).shape[1])
    chan_list = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2','Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4',
                 'F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz']
    seg_bi = pd_detect.utils.l_bipolar(segment)  
    df_peaks = data_obj['peaks']

    ax_list = []

    

    for i in range(0,seg_bi.shape[0]): #range(0,data_obj['channels'].shape[0]):
        ax = fig.add_subplot(gs[i,0], frame_on=False)

        if np.isin(data_obj['channels'], chan_list[i]).any(): #data_obj['channels'].index(chan_list[i]):
            cl = '#A52A4F'
        else:
            cl = 'k'

        ax.plot(time,seg_bi[i,:],cl)
       
        if isinstance(df_peaks, pd.DataFrame):
            if not df_peaks.empty:
                if i in df_peaks.index:
                    ax.scatter(df_peaks.loc[i]/fs,seg_bi[i,df_peaks.loc[i]],color='red')
        ax.plot(time1,np.diff(seg_bi[i,:]),color='gray',alpha=0.1)
        ax.set_ylabel(chan_list[i],fontsize=8, rotation=360, labelpad=20)
        ax.tick_params(axis='y', labelsize=5)
        if i != seg_bi.shape[0]-1:
            ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for line in ax.get_xgridlines():
            line.set_color('red')
            line.set_linestyle('--')
        for line in ax.get_ygridlines():
            line.set_color('grey')
        ax_list.append(ax)
    
    #gs_nested = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1], hspace=0.1)

    #topo_ax = fig.add_subplot(gs_nested[1,0])
    #plot_top_freq(data_obj,fig,topo_ax,"red")
    #topo_ax.set_aspect('auto')
    #topo_ax.set_position([0.79, 0.15, 0.18, 0.3])

    #txt = 'PD Algorithm Label: \n' + '$\mathbf {' +str(data_obj['type_event']) +'}$' + \
    #    '\n\n\nFrequency: $\mathbf{' + str(round(data_obj['event_frequency'],3)) + '}$ \nSpatial Extent: $\mathbf{' + str(round(data_obj['spatial_extent'],3)) + \
    #     '}$ \nAreas: \n' + str(data_obj['spatial_areas']) + '\n\n\nFilters: notch, 0.5-40Hz'
    
    #txt = 'Frequency: $\mathbf{' + str(round(data_obj['event_frequency'],3)) + '}$ \nSpatial Extent: $\mathbf{' + str(round(data_obj['spatial_extent'],3)) + \
    #    '}$ \nAreas: \n' + str(data_obj['spatial_areas']) + '\n\n\nFilters: notch, 0.5-40Hz'
    
    txt = 'Frequency: $\mathbf{' + str(round(data_obj['event_frequency'],3)) + '}$ \nSpatial Extent: $\mathbf{' + str(round(data_obj['spatial_extent'],3)) + '}$\n\n\nFilters: notch, 0.5-40Hz'

    text_ax = fig.add_subplot(gs[0,1])
    text_ax.text(-0.5, 0.5,txt,fontsize=11, ha='left', va='center')
    text_ax.axis('off')
    text_ax.set_position([0.97, 0.5, 0.2, 0.4])
    
    
    return fig

