import matplotlib.pyplot as plt
import sklearn.linear_model
import scipy.signal
import numpy as np

import pd_detector.utils
import pd_detector.eegfilt


def show_eeg_events_and_stats(data_obj):
    Fs = 200
    num_plots = 10
    seg_bi = data_obj["seg_bi"]
    bipolar_events = data_obj["pd_events_bi_select"]
    this_loc = data_obj["this_loc"]
    stats_obj = data_obj["stats_obj"]

    fig, axs_events = plt.subplots(num_plots, 1)

    locs, peaks = scipy.signal.find_peaks(
        pd_detector.utils.smooth(this_loc, 10),
        distance=20,
        height=(None, None),
        width=(None, None),
        prominence=2,
    )

    axs_events[0].plot(pd_detector.utils.smooth(this_loc, 10))
    axs_events[0].set_ylabel("Discharge Groups")

    # Evolution in frequency
    if len(locs) > 2:
        idi_pks = np.diff(locs) / Fs
        indx_pks = np.asarray([x for x in range(1, idi_pks.size + 1)])[:, None]
        N = idi_pks.shape[0]

        mdl = sklearn.linear_model.LinearRegression().fit(indx_pks, 1 / idi_pks)
        Y = (1 / indx_pks)[:, None]
        preds = mdl.predict(indx_pks)

        evolution_intercept = mdl.intercept_
        evolution_slope = mdl.coef_.item()
        evolution_se = np.std(preds - Y, ddof=0) / N
        evolution_rmse = np.sqrt(np.sum((preds - Y) ** 2) / N)

        # TODO: Currently not calculating this, but leaving it in for
        # consistency with MATLAB
        evolution_pVal = np.nan
        evolution_range = [
            evolution_intercept,
            evolution_intercept + evolution_slope * indx_pks[-1],
        ]
    else:
        evolution_slope = np.nan
        evolution_se = np.nan
        evolution_rmse = np.nan
        evolution_pVal = np.nan
        evolution_range = np.nan

    fig, ax_eeg = plt.subplots(num_plots, 1)
    seg_bi = pd_detector.eegfilt(seg_bi, 200, 1, 40, 0, 0, 0, "fir1", 0)

    gap = np.nan(1, seg_bi[0].shape[1])

    zScale = 1 / 100
    channel_withspace = [
        "Fp1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "",
        "Fp2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
        "",
        "Fp1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "",
        "Fp2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "",
        "Fz-Cz",
        "Cz-Pz",
    ]
    data = np.asarray(
        [
            seg_bi[0:3, :],
            gap,
            seg_bi[4:7, :],
            gap,
            seg_bi[8:11, :],
            gap,
            seg_bi[12:15, :],
            gap,
            seg_bi[16:17, :],
        ]
    )

    nCh = data.shape[0]
