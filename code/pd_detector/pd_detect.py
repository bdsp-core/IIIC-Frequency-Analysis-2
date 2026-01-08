import typing

import pandas as pd
import numpy as np

import pd_detector.utils
import pd_detector


DEBUG_FLAG = False


def pd_detect(
    seg: np.ndarray,
) -> typing.Tuple[float, pd.Series, typing.List, typing.Dict]:
    """Function to detect PD events

    Args:
        seg (np.ndarray): Array of EEG data

    Returns:
        freq (float): Sampling frequency
        channels (pd.Series): Pandas Series of channels from bipolar montage
        interpretation (list): List of interpretations of EEG data
        data_obj (dict): Dictionary containing various statistics used in
            diagnosis
    """

    seg_bi = pd_detector.utils.l_bipolar(seg)

    gap = np.empty((1, seg.shape[1]))
    gap[:] = np.nan

    Fs = 200
    win_size = 5
    step_size = 20
    smooth_win_size = 120
    thr_dur = 40

    channels_L = [x for x in range(4)] + [x for x in range(8, 12)]
    channels_R = [x for x in range(4, 8)] + [x for x in range(12, 16)]

    c_bi = pd_detector.utils.get_channel_labels_bipolar()

    thresh_hi = 80
    thresh_lo = 75

    smooth_win = Fs / 10
    boost_win = Fs / 10
    mean_win = 80

    # For PD
    freq_L = 2
    freq_H = 14

    total_samples = seg.shape[1]

    # TODO: One extra row in bipolar_events
    loc, bipolar_events, eeg_bp_smooth = pd_detector.hilo_dynamic_auto6(
        seg_bi,
        Fs,
        thresh_hi,
        thresh_lo,
        smooth_win,
        freq_L,
        freq_H,
        c_bi,
    )

    criteria = [
        '(e["intensity_max"])>18',
        '(e["intensity_max_raw"])-(e["intensity_min_raw"])>20',
    ]

    pd_events_bi = pd_detector.utils.select_events(bipolar_events, criteria)

    if DEBUG_FLAG:
        # TODO: Implement fct_vizEEG_bipolar(eeg_bp_smooth)
        pass

    
    group_events_bi = pd_detector.group_events_bi(
        pd_events_bi,
        seg_bi,
        eeg_bp_smooth,
        c_bi,
    )
    these_assays_group = [
        ["Left_t2", "Right_t2", "Global_t4"],
        ["Left_t0", "Right_t0", "Global_t0"],
    ]

    events_t4 = group_events_bi["events"][
        group_events_bi["channel"] == "Global_t4"
    ].item()
    events_t0 = group_events_bi["events"][
        group_events_bi["channel"] == "Global_t0"
    ].item()

    if events_t0 == 0:
        # No event detected across all channels
        freq = 0
        channels = [0]
        interpretation = ["no events"]
        data_obj = np.nan
        return freq, channels, interpretation, data_obj
    elif (events_t0 == 1) and (events_t4 == 0):
        # A single event across all channels
        freq = 1 / (total_samples / Fs)
        group_events_bi_byChan = group_events_bi.iloc[:18, :]
        group_events_bi_gt0 = group_events_bi[group_events_bi_byChan["events"] > 0]
        channels = group_events_bi_gt0["channel"]
        interpretation = ["one event"]
        data_obj = np.nan
        return freq, channels, interpretation, data_obj
    elif events_t4 == 0:
        # more than 1 event at theshold=0, but no events at threshold =4 --> use lowest threshold assays
        these_assays = these_assays_group[1]
    else:
        # events even at theshold =4 --> use intermediate threshold assays
        these_assays = these_assays_group[0]

    these_ge_b = group_events_bi[group_events_bi["channel"].isin(these_assays)]
    these_ge_b_pdr_max = np.nanmax(these_ge_b["power_discharges_rel"])
    indx = np.nanargmax(these_ge_b["power_discharges_rel"])

    pow_G = these_ge_b["power_discharges_rel"].iloc[2]
    pow_R = these_ge_b["power_discharges_rel"].iloc[1]
    pow_L = these_ge_b["power_discharges_rel"].iloc[0]

    rel_PLvPG = np.abs((pow_L - pow_G) / pow_G)
    rel_PRvPG = np.abs((pow_R - pow_G) / pow_G)
    rel_PLvPR = np.abs(np.log(pow_L / pow_R))

    stats_obj = pd.DataFrame(
        {
            "pow_L": [pow_L],
            "pow_R": [pow_R],
            "pow_G": [pow_G],
            "rel_PLvPG": [rel_PLvPG],
            "rel_PRvPG": [rel_PRvPG],
            "rel_PLvPR": [rel_PLvPR],
        }
    )

    # Finding which set of discharges (left, right, global) has the highest
    # "relative power" (ie. greatest percentage of total BP power that is
    # explained by the dischanges)
    if indx == 0:
        # TODO: test
        # Report as Left LPD, all left channels with events > 1, unless the
        # relative difference in relative power b/w L vs G and R vs G is too
        # similar

        if rel_PLvPG < 0.1 and rel_PLvPR < 0.15:
            # Report as GPD
            group_events_bi_byChan = group_events_bi.iloc[:18]
            group_events_bi_gt0 = group_events_bi_byChan[
                group_events_bi_byChan["events"] > 0
            ]
            channels = group_events_bi_gt0["channel"]
            interpretation = ["GPD"]

            # Redefine as GPD
            indx = 2
            this_loc = these_ge_b["loc"].iloc[2]
            this_evolution_slope = these_ge_b["evolution_slope"].iloc[2]
            this_evolution_rmse = these_ge_b["evolution_RMSE"].iloc[2]

        else:
            group_events_bi_byChan = group_events_bi.iloc[channels_L + [16, 17]]
            group_events_bi_gt0 = group_events_bi_byChan[
                group_events_bi_byChan["events"] > 0
            ]
            channels = group_events_bi_gt0["channel"]
            interpretation = ["L LPD"]

            this_loc = these_ge_b["loc"].iloc[0]
            this_evolution_slope = these_ge_b["evolution_slope"].iloc[0]
            this_evolution_rmse = these_ge_b["evolution_RMSE"].iloc[0]

    elif indx == 1:
        # TODO: test
        # Report as right LPD, all right channels with events > 1, unless the
        # relative difference in relative power b/w L vs G and R vs G is too
        # similar
        if rel_PLvPG < 0.1 and rel_PLvPR < 0.15:
            # Report as GPD
            group_events_bi_byChan = group_events_bi.iloc[:18]
            group_events_bi_gt0 = group_events_bi_byChan[
                group_events_bi_byChan["events"] > 0
            ]
            channels = group_events_bi_gt0["channel"]
            interpretation = ["GPD"]

            # Redefine as GPD
            indx = 2
            this_loc = these_ge_b["loc"].iloc[2]
            this_evolution_slope = these_ge_b["evolution_slope"].iloc[2]
            this_evolution_rmse = these_ge_b["evolution_RMSE"].iloc[2]
        else:
            group_events_bi_byChan = group_events_bi.iloc[channels_R + [16, 17]]
            group_events_bi_gt0 = group_events_bi_byChan[
                group_events_bi_byChan["events"] > 0
            ]
            channels = group_events_bi_gt0["channel"]
            interpretation = ["R LPD"]

            this_loc = these_ge_b["loc"].iloc[1]
            this_evolution_slope = these_ge_b["evolution_slope"].iloc[1]
            this_evolution_rmse = these_ge_b["evolution_RMSE"].iloc[1]
    elif indx == 2:
        # Report as GPD, all channels with events > 1
        group_events_bi_byChan = group_events_bi.iloc[:18]
        group_events_bi_gt0 = group_events_bi_byChan[
            group_events_bi_byChan["events"] > 0
        ]
        channels = group_events_bi_gt0["channel"]

        if rel_PLvPR > 0.08:
            if pow_L > pow_R:
                interpretation = ["GPD-bilateral asym, L>R"]
            else:
                interpretation = ["GPD-bilateral asym R>L"]
        else:
            interpretation = ["GPD"]

        this_loc = these_ge_b["loc"].iloc[2]
        this_evolution_slope = these_ge_b["evolution_slope"].iloc[2]
        this_evolution_rmse = these_ge_b["evolution_RMSE"].iloc[2]

    channels = channels.reset_index(drop=True)

    # Calculate frequency (Hz) based on the median of the IDI for detected
    # events, unless the % near the median is <75%, in which case use the
    # frequency calculated from cross-correlation based method.

    freq = [
        these_ge_b.iloc[indx]["this_idi_xcorr_Hz"],
        these_ge_b.iloc[indx]["this_idi_hz_med"],
    ]

    if np.isnan(freq[0]) and np.isnan(freq[1]):
        freq = these_ge_b.iloc[indx]["event_rate"]
    elif (not np.isnan(freq[1])) and (
        (these_ge_b.iloc[indx]["this_idi_pctNearMed"] >= 0.74) or np.isnan(freq[0])
    ):
        freq = freq[1]
    else:
        freq = freq[0]

    if np.isnan(freq):
        freq = 0

    # TODO: Implement
    spatial_extent = pd_detector.calc_spatial_extent(group_events_bi_gt0, c_bi)

    spatial_areas = []
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


    data_obj = {
        "seg_bi": seg_bi,
        "this_loc": this_loc,
        "pd_events_bi": pd_events_bi,
        "pd_events_bi_select": pd_detector.utils.cull_events_by_channel(
            pd_events_bi, np.unique(group_events_bi_gt0["channel"])
        ),
        "group_events_bi": group_events_bi,
        "group_events_bi_gt0": group_events_bi_gt0,
        "this_evolution_slope": this_evolution_slope,
        "this_evolution_rmse": this_evolution_rmse,
        "freq": freq,
        "channels": channels,
        "interpretation": [interpretation],
        "type_event":interpretation[0],
        "event_frequency":freq,
        "spatial_extent": spatial_extent,
        "spatial_areas":spatial_areas,
        "stats_obj": stats_obj,
    }
    return data_obj
