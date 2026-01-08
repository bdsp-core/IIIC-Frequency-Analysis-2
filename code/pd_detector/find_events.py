import typing

import skimage.measure
import pandas as pd
import numpy as np

import pd_detector.utils
import pd_detector


def hilo_dynamic_auto6(
    seg: np.ndarray,
    Fs: float,
    thresh_hi: float,
    thresh_lo: float,
    smooth_win: float,
    freq_lo: float,
    freq_hi: float,
    channel_labels: typing.List[str],
) -> typing.Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Use a high-low threshold approach to find events. Naive to specific BP or
    montage. Accepts thresh_hi, thresh_lo based on percentile -- but uses local
    slope to determine low threshold.

    Args:
       seg (np.ndarray): n x m double array of EEG in bipolar montage
       Fs (float):  sampling freq in sec-1

    Returns:
       loc (np.ndarray): n x m array corresponding to positions of candidate discharges.
       theseEvents (list): list containing stat objects from region props
       eeg_bp_smooth (np.ndarray): np.ndarray containing smoothed EEG signal
    """
    channels = channel_labels
    loc = np.zeros(seg.shape)
    total_samples = seg.shape[1]

    eeg_bp, filtwts = pd_detector.eegfilt(
        seg,
        Fs,
        freq_lo,
        freq_hi,
        epochframes=0,
        filtorder=0,
        revfilt=0,
        firtype="fir1",
        causal=0,
    )
    eeg_bp_smooth = np.empty(eeg_bp.shape)
    eeg_bp_smooth[:] = np.nan

    these_events = []

    for i in range(eeg_bp.shape[0]):
        this_eeg_bp_smooth = pd_detector.utils.smooth(
            eeg_bp[i, :] ** 2, int(smooth_win)
        )
        eeg_bp_smooth[i, :] = this_eeg_bp_smooth

    for i in range(eeg_bp_smooth.shape[0]):
        this_eeg_bp_smooth = eeg_bp_smooth[i, :]

        # "midpoint" interpolation follows MATLAB standard
        eeg_prctl_hi = np.percentile(
            this_eeg_bp_smooth,
            thresh_hi,
            #method="midpoint", python version >3.6, numpy >1.22
            interpolation = 'midpoint'
        )

        eeg_prctl_lo = np.percentile(
            this_eeg_bp_smooth,
            thresh_lo,
            #method="midpoint", python version >3.6, numpy >1.22
            interpolation = 'midpoint'
        )

        bw_hi = this_eeg_bp_smooth > eeg_prctl_hi
        bw_lo1 = np.diff(eeg_bp[i, :]) > 0
        bw_lo2 = np.diff(eeg_bp[i, :]) < 0

        # TODO: This returns a value not exactly the same as MATLAB, but fairly close
        lpds_hi = skimage.measure.regionprops(bw_hi[None, :].astype(int))

        if len(lpds_hi) > 1:
            raise RuntimeError("Too many regionprops objects found!")

        indx_hi = lpds_hi[0].coords[:, 1]
        indx_hi = np.stack([indx_hi, np.zeros(indx_hi.shape[0])], axis=-1)

      
        isect1 = pd_detector.utils.bwselect(
            np.stack([bw_lo1, bw_lo1]).astype(int).T,
            indx_hi,
            n=1,
        )
        isect1 = isect1[:, 0].T
        isect2 = pd_detector.utils.bwselect(
            np.stack([bw_lo2, bw_lo2]).astype(int).T,
            indx_hi,
            n=1,
        )
        isect2 = isect2[:, 0].T
        isect = isect1 | isect2
        isect = np.append(isect, 0)
        isect_label = skimage.measure.label(isect[None, :])

        lpds_ = skimage.measure.regionprops_table(
            label_image=isect_label,
            intensity_image=this_eeg_bp_smooth[None, :],
            properties=[
                "area",
                "bbox",
                "coords",
                "intensity_max",
                "intensity_mean",
                "image_intensity",
                "centroid_weighted",
            ],
        )
        lpds_ = pd.DataFrame(lpds_)
        lpds_raw = skimage.measure.regionprops_table(
            label_image=isect_label,
            intensity_image=eeg_bp[i, :][None, :],
            properties=[
                "intensity_max",
                "intensity_mean",
                "intensity_min",
                "image_intensity",
            ],
        )
        
        lpds_raw = pd.DataFrame(lpds_raw)

        these_var_names = list(lpds_raw.keys())
        rename_dict = {x: x + "_raw" for x in these_var_names}
        lpds_raw = lpds_raw.rename(columns=rename_dict)

        this_result = pd.concat([lpds_, lpds_raw], axis=1)
        this_result["channel"] = np.tile(channels[i], lpds_.shape[0])
        this_result["channel_num"] = np.tile(i, lpds_.shape[0])
        this_result["channel_event_num"] = np.arange(lpds_.shape[0]).astype(int)

        these_events.append(this_result)

    these_events = pd.concat(these_events, axis=0)
    these_events["event_num"] = np.arange(these_events.shape[0]).astype(int)
    loc = pd_detector.utils.locate_events_from_table(
        channels,
        these_events,
        total_samples,
    )

    return loc, these_events, eeg_bp_smooth


def group_events_bi(
    bi_events: pd.DataFrame,
    seg_bi: np.ndarray,
    eeg_bp_smooth: np.ndarray,
    channels: typing.List[str],
) -> pd.DataFrame:
    """Function to group bipolar events

    Args:
        bi_event(pd.DataFrame): Pandas DataFrame of bipolar events
        seg_bi (np.ndarray): Differential EEG signals
        eeg_bp_smooth (np.ndarray): Numpy array containing smoothed EEG signal
        channels (list): List of strings containing channel labels

    Returns:
        these_events (pd.DataFrame): Pandas DataFrame containing relevant data
            from specified events
    """

    Fs = 200
    total_samples = seg_bi.shape[1]
    these_results = pd.DataFrame()

    for channel_idx in range(len(channels)):
        this_channel = bi_events[bi_events["channel_num"] == channel_idx]

        if this_channel.shape[0] == 0:
            this_channel = pd_detector.utils.create_nan_df(this_channel)
            this_channel["channel"] = [channels[channel_idx]]
            this_channel["channel_num"] = [channel_idx]

        this_eeg_bp_smooth = eeg_bp_smooth[channel_idx, :]
        this_result = pd_detector.calc_periodicity(
            this_channel,
            this_eeg_bp_smooth,
            Fs,
        )
        these_loc = pd_detector.utils.locate_events_from_table(
            [this_channel["channel"].iloc[0]], this_channel, total_samples
        )
        this_loc = np.sum(these_loc, axis=0)
        this_result["loc"] = [this_loc]
        this_result["threshold"] = np.nan
        these_results = pd.concat((these_results, this_result), axis=0)

    these_result = these_results.reset_index(drop=True)
    these_groups_labels = [
        "Left_t0",
        "Right_t0",
        "Central_t0",
        "Global_t0",
        "Left_t2",
        "Right_t2",
        "Central_t1",
        "Global_t4",
    ]
    channels_L = [np.arange(0, 4), np.arange(8, 12)]
    channels_R = [np.arange(4, 8), np.arange(12, 16)]
    channels_C = [np.arange(16, 18)]
    channels_G = [np.arange(0, 18)]

    these_groups_channelNums = [
        channels_L,
        channels_R,
        channels_C,
        channels_G,
        channels_L,
        channels_R,
        channels_C,
        channels_G,
    ]
    these_groups_t = [0, 0, 0, 0, 2, 2, 1, 4]

    for i in range(len(these_groups_labels)):
        this_group_label = these_groups_labels[i]

        this_groups_channel_nums = np.concatenate(these_groups_channelNums[i]).tolist()
        these_chan = bi_events[
            np.isin(
                bi_events["channel_num"],
                this_groups_channel_nums,
            )
        ]

        these_loc = pd_detector.utils.locate_events_from_table(
            this_groups_channel_nums,
            these_chan,
            total_samples,
        )

        # TODO: Fairly close, but not exact
        this_loc = these_loc.sum(0)
        this_eeg_bp_smooth = eeg_bp_smooth[this_groups_channel_nums, :].sum(0)
        _, this_chan = find_events_simple(
            (this_loc > these_groups_t[i])[None, :],
            this_eeg_bp_smooth,
            [this_group_label],
        )
        this_result = pd_detector.calc_periodicity(
            this_chan,
            this_eeg_bp_smooth,
            Fs,
        )

        this_result["loc"] = [this_loc]
        this_result["threshold"] = these_groups_t[i]
        these_results = pd.concat((these_results, this_result), axis=0)

    these_groups_labels = [
        "Left_kmeans",
        "Right_kmeans",
        "Central_kmeans",
        "Global_kmeans",
    ]
    thse_groups_channel_nums = [
        channels_L,
        channels_R,
        channels_C,
        channels_G,
    ]

    for i in range(len(these_groups_labels)):
        this_group_label = these_groups_labels[i]
        this_groups_channel_nums = np.concatenate(these_groups_channelNums[i]).tolist()
        these_chan = bi_events[
            np.isin(
                bi_events["channel_num"],
                this_groups_channel_nums,
            )
        ]

        these_loc = pd_detector.utils.locate_events_from_table(
            this_groups_channel_nums,
            these_chan,
            total_samples,
        )

        this_loc = these_loc.sum(0)
        this_eeg_bp_smooth = eeg_bp_smooth[this_groups_channel_nums, :].sum(0)
        this_eeg_kmeans = pd_detector.utils.kmeans(
            this_eeg_bp_smooth[:, None], n_clusters=3
        )
        _, this_chan = find_events_simple(
            (this_eeg_kmeans > 0)[None, :],
            this_eeg_bp_smooth,
            [this_group_label],
        )
        this_result = pd_detector.calc_periodicity(
            this_chan,
            this_eeg_bp_smooth,
            Fs,
        )
        this_result["loc"] = [this_loc]
        this_result["threshold"] = "kmeans"
        these_results = pd.concat((these_results, this_result), axis=0)

    these_results = these_results.reset_index(drop=True)

    these_groups_labels = ["_global_kmeans"]
    these_subgroup_labels = ["Left", "Right"]
    these_groups_channelNums = [channels_G]
    these_subgroup_channelNums = [
        np.concatenate(channels_L),
        np.concatenate(channels_R),
    ]

    for i in range(len(these_groups_labels)):
        this_group_label = these_groups_labels[i]
        this_groups_channel_nums = np.concatenate(these_groups_channelNums[i]).tolist()

        these_subgroup_channelNums = np.stack(these_subgroup_channelNums)

        eeg_bp_smooth_L = np.sum(eeg_bp_smooth[these_subgroup_channelNums[0, :], :], 0)
        eeg_bp_smooth_R = np.sum(eeg_bp_smooth[these_subgroup_channelNums[1, :], :], 0)

        this_eeg_kmeans = pd_detector.utils.kmeans(
            np.stack([eeg_bp_smooth_L, eeg_bp_smooth_R], axis=0),
            n_clusters=3,
        )

        this_loc, this_chan = find_events_simple(
            (this_eeg_kmeans[:, 0] > 0)[None, :],
            eeg_bp_smooth_L,
            [these_subgroup_labels[0] + this_group_label],
        )

        this_result = pd_detector.calc_periodicity(
            this_chan,
            eeg_bp_smooth_L,
            Fs,
        )
        this_result["loc"] = [this_loc]
        this_result["threshold"] = "kmeans"
        these_results = pd.concat((these_results, this_result), axis=0)

        this_loc, this_chan = find_events_simple(
            (this_eeg_kmeans[:, 1] > 0)[None, :],
            eeg_bp_smooth_R,
            [these_subgroup_labels[1] + this_group_label],
        )

        this_result = pd_detector.calc_periodicity(
            this_chan,
            eeg_bp_smooth_R,
            Fs,
        )
        this_result["loc"] = [this_loc]
        this_result["threshold"] = "kmeans"
        these_results = pd.concat((these_results, this_result), axis=0)

    these_events = these_results.reset_index(drop=True)
    return these_events


def find_events_simple(
    data_log: np.ndarray,
    this_eeg_bp_smooth: np.ndarray,
    channel_labels: typing.List[str],
) -> typing.Tuple[np.ndarray, pd.DataFrame]:
    """Simple version of find_events function when the input data is already
    logical and we just want to find the events

    Args:
        data_log (np.ndarray): Logical numpy array containing location of events
        this_eeg_bp_smooth (np.ndarray): Numpy array with signal data
        channel_labels (list): List of channel labels

    Returns:
        loc (np.ndarray): Logical numpy array containing location of events
        these_events (pd.DataFrame): Pandas DataFrame containing relevant data
            from specified events
    """
    channels = channel_labels
    loc = np.zeros(data_log.shape)
    total_samples = data_log.shape[1]

    these_events = pd.DataFrame()

    for i in range(data_log.shape[0]):
        isect = data_log
        # TODO: This returns a value not exactly the same as MATLAB, but fairly
        # close. Missing a 24-element area for test case
        isect_label = skimage.measure.label(isect)
        lpds_ = skimage.measure.regionprops_table(
            label_image=isect_label,
            intensity_image=this_eeg_bp_smooth[None, :],
            properties=[
                "area",
                "bbox",
                "coords",
                "intensity_max",
                "intensity_mean",
                "image_intensity",
                "centroid_weighted",
            ],
        )

        this_result = pd.DataFrame(lpds_)

        if not this_result.empty:
            this_result["channel"] = channels[i]
            this_result["channel_num"] = i
            this_result["channel_event_num"] = [x for x in range(this_result.shape[0])]

        else:
            this_result = pd_detector.utils.create_nan_df(this_result)
            this_result["channel"] = [channels[i]]
            this_result["channel_num"] = [i]
            this_result["channel_event_num"] = [np.nan]

        these_events = pd.concat((these_events, this_result), axis=0)

    these_events["event_num"] = [x for x in range(these_events.shape[0])]

    loc = pd_detector.utils.locate_events_from_table(
        channels,
        these_events,
        total_samples,
    )
    return loc, these_events
