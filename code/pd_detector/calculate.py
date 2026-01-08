import warnings
import typing

import pandas as pd
import numpy as np

import sklearn.linear_model
import scipy.signal
import scipy.stats

import pd_detector.utils


def calc_periodicity(
    this_chan: pd.DataFrame, this_eeg_bp_smooth: np.ndarray, Fs: float
) -> pd.DataFrame:
    """Function to calculate periodicity of an input signal

    Args:
        this_chan (pd.DataFrame): Table of events for which to calculate
            periodicity
        this_eeg_bp_smooth (np.ndarray): Full time series of bipolar power from
            channel where events were called
        Fs (float): Sampling frequency

    Returns:
        result (pd.DataFrame): A table of periodicity calculations based on the
            events provided in this_chan
    """
    t_lo = 0.33
    t_hi = 3

    if (
        (not this_chan.empty)
        and (not this_chan.area.isnull().any())
        and (this_chan.shape[0] > 1)
    ):
        these_idi = np.diff(this_chan["centroid_weighted-1"]) / Fs

        # Fit distribution to log2 transformed data (given distribution around
        # mean is not symmetric d/t ~0.5-2x)
        this_idi_log2 = np.log2(these_idi)

        # NOTE: The original matlab code uses the `mle` function; we use manual
        # fitting of a normal distribution
        this_idi_log2 = np.asarray(scipy.stats.norm.fit(this_idi_log2))
        this_idi = 2**this_idi_log2
        this_idi_hz = 1 / this_idi

        hc_mode = these_idi > (t_hi * this_idi[0])
        lc_mode = these_idi < (t_lo * this_idi[0])
        this_idi_pctNearMode = (
            these_idi.size - np.sum(np.logical_or(hc_mode, lc_mode))
        ) / these_idi.size

        # NOTE: Look for insufficient data error. Thrown in MATLAB, not sure
        # about scipy

        this_idi_hz_med = 1 / np.median(these_idi)

        hc_med = these_idi > (t_hi * np.median(these_idi))
        lc_med = these_idi < (t_lo * np.median(these_idi))

        this_idi_pctNearMed = (
            these_idi.size - np.sum(np.logical_or(hc_med, lc_med))
        ) / these_idi.size

        this_loc = np.zeros((1, this_eeg_bp_smooth.shape[0]))

        for coord_idx in this_chan.coords:
            true_coord_idx = [x[1] for x in coord_idx]
            this_loc[:, true_coord_idx] = 1

        r = pd_detector.utils.xcorr(this_loc.squeeze(), this_loc.squeeze(), 5 * Fs)

        min_dist_t = 0.5
        min_dist_idx = Fs * min_dist_t

        # TODO: This returns very close to the MATLAB peaks, but not exactly
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
            )
            locs_idx, peak_props = scipy.signal.find_peaks(
                r[5 * Fs - 1 :],
                distance=min_dist_idx,
                height=(None, None),
                width=(None, None),
                prominence=(None, None),
            )

        # Converting locs to time indexing
        locs = locs_idx / Fs
        pks = peak_props["peak_heights"]
        w = peak_props["widths"] / Fs
        p = peak_props["prominences"]

        indx_prom_gt_0 = np.logical_and(np.round(p, 3) > 0.1, np.round(pks, 3) > 0.2)
        indx_prom_gt_0_sub = np.where(indx_prom_gt_0)[0]
        this_indx_minlag = np.argsort(locs[indx_prom_gt_0])

        if indx_prom_gt_0.any():
            this_indx = indx_prom_gt_0_sub[this_indx_minlag[0]]
            this_idi_xcorr_loc = locs[this_indx]
            this_idi_xcorr_prm = p[this_indx]
            this_idi_xcorr_Hz = 1 / locs[this_indx]
        else:
            this_idi_xcorr_loc = np.nan
            this_idi_xcorr_prm = np.nan
            this_idi_xcorr_Hz = np.nan

        r = pd_detector.utils.xcorr(this_eeg_bp_smooth, this_eeg_bp_smooth, 5 * Fs)
        locs_idx, peak_props = scipy.signal.find_peaks(
            r[5 * Fs - 1 :],
            distance=min_dist_idx,
            height=(None, None),
            width=(None, None),
            prominence=(None, None),
        )

        locs = locs_idx / Fs
        pks = peak_props["peak_heights"]
        w = peak_props["widths"] / Fs
        p = peak_props["prominences"]

        indx_prom_gt_0 = np.logical_and(np.round(p, 3) > 0.1, np.round(pks, 3) > 0.2)
        indx_prom_gt_0_sub = np.where(indx_prom_gt_0)[0]
        this_indx_minlag = np.argsort(locs[indx_prom_gt_0])

        if indx_prom_gt_0.any():
            this_indx = indx_prom_gt_0_sub[this_indx_minlag[0]]
            this_idi_xcorr_BP_loc = locs[this_indx]
            this_idi_xcorr_BP_prm = p[this_indx]
            this_idi_xcorr_BP_Hz = 1 / locs[this_indx]
        else:
            this_idi_xcorr_BP_loc = np.nan
            this_idi_xcorr_BP_prm = np.nan
            this_idi_xcorr_BP_Hz = np.nan

        # Calculate power discharges relative to total power for channel
        power_discharges = np.hstack(this_chan["image_intensity"]).sum()
        power_total = np.sum(this_eeg_bp_smooth)
        power_discharges_rel = power_discharges / power_total

        N = these_idi.shape[0]
        indx_idi = np.arange(N)[:, None]
        Y = (1 / these_idi)[:, None]
        mdl = sklearn.linear_model.LinearRegression().fit(indx_idi, Y)
        evolution_slope = mdl.coef_.item()

        # Calculating squared error
        preds = mdl.predict(indx_idi)
        res = (Y - preds) ** 2

        # NOTE: this is a little sus but it returns the correct value. How SE is
        # calculated is not documented in MATLAB
        SE = np.std(preds - Y, ddof=0) / N

        # NOTE: Also slightly off from MATLAB
        RMSE = np.sqrt(np.sum((preds - Y) ** 2) / N)

        # Results dictionary
        thisResult = {
            "channel": this_chan.channel.iloc[0],
            "channel_num": np.unique(this_chan.channel_num).item(),
            "events": this_chan.shape[0],
            "event_rate": this_chan.shape[0] / (this_eeg_bp_smooth.shape[0] / Fs),
            "this_idi_hz": this_idi_hz,
            "this_idi_hz_med": this_idi_hz_med,
            "this_idi_pctNearMed": this_idi_pctNearMed,
            "this_idi_pctNearMode": this_idi_pctNearMode,
            "this_idi_xcorr_Hz": this_idi_xcorr_Hz,
            "this_idi_xcorr_lag": this_idi_xcorr_loc,
            "this_idi_xcorr_prm": this_idi_xcorr_prm,
            "this_idi_xcorr_BP_Hz": this_idi_xcorr_BP_Hz,
            "this_idi_xcorr_BP_lag": this_idi_xcorr_BP_loc,
            "this_idi_xcorr_BP_prm": this_idi_xcorr_BP_prm,
            "power_discharges": power_discharges,
            "power_total": power_total,
            "power_discharges_rel": power_discharges_rel,
            "evolution_slope": evolution_slope,
            "evolution_SE": SE,
            "evolution_RMSE": RMSE,
        }

    else:
        thisResult = {}
        thisResult["channel"] = this_chan.channel.iloc[0]
        thisResult["channel_num"] = np.unique(this_chan.channel_num).item()

        if not this_chan.area.isnull().any() and this_chan.shape[0] >= 1:
            thisResult["events"] = this_chan.shape[0]
            thisResult["eventRate"] = (
                this_chan.shape[0] / (this_eeg_bp_smooth.shape[0] / Fs),
            )
        else:
            thisResult["events"] = 0
            thisResult["eventRate"] = 0

        thisResult.update(
            {
                "this_idi_hz": np.nan,
                "this_idi_hz_med": np.nan,
                "this_idi_pctNearMed": np.nan,
                "this_idi_pctNearMode": np.nan,
                "this_idi_xcorr_Hz": np.nan,
                "this_idi_xcorr_lag": np.nan,
                "this_idi_xcorr_prm": np.nan,
                "this_idi_xcorr_BP_Hz": np.nan,
                "this_idi_xcorr_BP_lag": np.nan,
                "this_idi_xcorr_BP_prm": np.nan,
                "power_discharges": np.nan,
                "power_total": np.nan,
                "power_discharges_rel": np.nan,
                "evolution_slope": np.nan,
                "evolution_SE": np.nan,
                "evolution_RMSE": np.nan,
            }
        )

    thisResult = pd.DataFrame.from_dict(thisResult, orient="index").transpose()
    return thisResult


def calc_spatial_extent(group_events_bi_gt0: pd.DataFrame, c_bi: typing.List) -> float:
    """Calculate the spatial extent as a percentage of bipolar channels

     N.b. In theory, bipolar events over-represents frontal/occipital channels
     -- events from average montage may be less biased

    Args:
        group_events_bi_gt0 (pd.DataFrame): A pandas DataFrame of events
            (rows=channels) with channel statistics
        c_bi (list): A list of bipolar channel pairs

    Returns:
        spatial_extent (float): A float in the range [0,1] representing the
        percentage of channels involved
    """
    channels = group_events_bi_gt0["channel"].tolist()
    channel_members = [x for x in c_bi if x in channels]
    spatial_extent = len(channel_members) / len(c_bi)
    return spatial_extent
