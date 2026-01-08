import dataclasses
import typing
import math
import os

import skimage.morphology
import sklearn.cluster
import scipy.fftpack
import pandas as pd
import numpy as np

import matlab.engine


def make_dir_safe(dirname: str) -> None:
    """Helper function which checks for directory existence before creation"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def create_table_df(f: matlab.object, eng: matlab.engine) -> pd.DataFrame:
    data_dict = {
        "Other": np.asarray(eng.getfield(f, "Other")).squeeze(),
        "Seizure": np.asarray(eng.getfield(f, "Seizure")).squeeze(),
        "LPD": np.asarray(eng.getfield(f, "LPD")).squeeze(),
        "GPD": np.asarray(eng.getfield(f, "GPD")).squeeze(),
        "LRDA": np.asarray(eng.getfield(f, "LRDA")).squeeze(),
        "GRDA": np.asarray(eng.getfield(f, "GRDA")).squeeze(),
    }
    if data_dict["Other"].size < 2:
        for k, v in data_dict.items():
            data_dict[k] = [v.item()]

    data_df = pd.DataFrame.from_dict(data_dict)
    return data_df


class MATFile:
    def __init__(self, fname: str, eng: matlab.engine):
        self.raw_data = eng.load(fname)
        self.table_vars = ["P_model", "Y_human"]
        for k, v in self.raw_data.items():
            if k in self.table_vars:
                v = create_table_df(v, eng)
            setattr(self, k, v)


@dataclasses.dataclass
class FileData:
    n: int
    file: str
    y_human_types: list
    y_human_types_pct: pd.DataFrame
    stats_obj: None


def get_channel_labels_bipolar() -> typing.List[str]:
    channelLabels_bipolar = [
        "Fp1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "Fp2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
        "Fp1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "Fp2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "Fz-Cz",
        "Cz-Pz",
    ]
    return channelLabels_bipolar


def smooth(arr: np.ndarray, win_size: int) -> np.ndarray:
    """
    Python implementation of MATLAB's smooth function

    Implemented in: https://stackoverflow.com/a/40443565

    TODO: This is slightly (<1%) off from the MATLAB implementation
    """

    #  if win_size % 2 == 0:
    #      raise ValueError("win_size must be an odd number")

    #  smooth_arr = np.convolve(arr, np.ones(win_size), "valid") / win_size
    wsz_2 = win_size // 2
    smooth_arr = (np.convolve(arr, np.ones(win_size), "same") / win_size)[wsz_2:-wsz_2]

    r = np.arange(1, win_size, 2)

    start = np.cumsum(arr[: win_size - 1])[::2] / r
    stop = (np.cumsum(arr[:-win_size:-1])[::2] / r)[::-1]

    out = np.concatenate((start, smooth_arr, stop))

    return out


def bwselect(arr: np.ndarray, idxs: np.ndarray, n: int = 2) -> np.ndarray:
    """Reimplementation of MATLAB bwselect"""
    label_arr = skimage.morphology.label(arr, connectivity=n)
    label_unselected = np.arange(1, np.max(label_arr).astype(int) + 1).tolist()
    selected_img = np.zeros(arr.shape).astype(int)

    for idx in idxs:
        idx = idx.astype(int)
        label_val = label_arr[idx[0], idx[1]]

        if (label_val > 0) and (label_val in label_unselected):
            selected_img[label_arr == label_val] = 1
            label_unselected.remove(label_val)

    # TODO: This is pretty close, but not exactly the same as the MATLAB
    # implementation
    return selected_img


def locate_events_from_table(
    channels: typing.List[str],
    df: pd.DataFrame,
    total_samples: int,
) -> np.ndarray:
    """Takes a dataframe of events and returns an n x m logical array
    indicating the location of all events in an epoch"""

    channel_nums = np.unique(df["channel_num"])
    loc_arr = np.zeros((len(channels), total_samples))

    for idx, channel_num in enumerate(channel_nums):
        lpds_tbl = df[df["channel_num"] == channel_num].reset_index(drop=True)
        these_pixel_idx_list = lpds_tbl["coords"]
        tPIL_s = these_pixel_idx_list.shape[0]

        if not (
            lpds_tbl.shape[0] == 1 and tPIL_s == 1 and np.isnan(lpds_tbl["area"].item())
        ):
            if (not lpds_tbl.empty) and (not np.isnan(lpds_tbl["coords"][0]).all()):
                lpds_idx = np.concatenate([x for x in lpds_tbl["coords"]], axis=0)
                loc_arr[idx, lpds_idx] = 1
    return loc_arr


def select_events(
    bipolar_events: pd.DataFrame,
    criteria: typing.List[str],
) -> pd.DataFrame:
    """Function to take an event dataframe and return a new table containing
    events based on passed criteria

    Args:
        bipolar_events (pd.DataFrame): Pandas dataframe containing events
            and associated data
        criteria (list): List of strings containing comparative expressions to
            be evaluated using eval command. Must reference headers in
            bipolar_events, e.g. "(e.Area>30)" or "!(e.Area>30)". Use e.***
            syntax

    Returns:
        these_events_new (pd.DataFrame): Pandas dataframe containing subset of
            the original n rows of bipolar_events according to specified
            criteria
    """

    e = bipolar_events.reset_index(drop=True)
    Fs = 200

    these_events_new = pd.DataFrame
    these_indx = np.ones(bipolar_events.shape[0]).astype(bool)

    for c in criteria:
        this_indx = eval(c).to_numpy()
        these_indx = these_indx & this_indx
        pass

    these_events_new = e.loc[these_indx].reset_index(drop=True)
    return these_events_new


def l_bipolar(data: np.ndarray) -> np.ndarray:
    # labels = {'Fp1';'F3';'C3';'P3';'F7';'T3';'T5';'O1';'Fz';'Cz';'Pz';'Fp2';'F4';'C4';'P4';'F8';'T4';'T6';'O2'};

    data_bipolar = np.zeros((18, data.shape[1]))

    data_bipolar[8, :] = data[0, :] - data[1, :]  # Fp1-F3
    data_bipolar[9, :] = data[1, :] - data[2, :]  # F3-C3
    data_bipolar[10, :] = data[2, :] - data[3, :]  # C3-P3
    data_bipolar[11, :] = data[3, :] - data[7, :]  # P3-O1

    data_bipolar[12, :] = data[11, :] - data[12, :]  # Fp2-F4
    data_bipolar[13, :] = data[12, :] - data[13, :]  # F4-C4
    data_bipolar[14, :] = data[13, :] - data[14, :]  # C4-P4
    data_bipolar[15, :] = data[14, :] - data[18, :]  # P4-O2

    data_bipolar[0, :] = data[0, :] - data[4, :]  # Fp1-F7
    data_bipolar[1, :] = data[4, :] - data[5, :]  # F7-T3
    data_bipolar[2, :] = data[5, :] - data[6, :]  # T3-T5
    data_bipolar[3, :] = data[6, :] - data[7, :]  # T5-O1

    data_bipolar[4, :] = data[11, :] - data[15, :]  # Fp2-F8
    data_bipolar[5, :] = data[15, :] - data[16, :]  # F8-T4
    data_bipolar[6, :] = data[16, :] - data[17, :]  # T4-T6
    data_bipolar[7, :] = data[17, :] - data[18, :]  # T6-O2

    data_bipolar[16, :] = data[8, :] - data[9, :]  # Fz-Cz
    data_bipolar[17, :] = data[9, :] - data[10, :]  # Cz-Pz
    return data_bipolar


def create_nan_df(this_table: pd.DataFrame) -> pd.DataFrame:
    """Helper function to create a pandas DataFrame of the same format as the
    input, but filled with NaN values

    Args:
        this_table (pd.DataFrame): DataFrame to replicate formatting from

    Returns:
        nan_table (pd.DataFrame): DataFrame with NaN values
    """
    index = this_table.index.tolist()
    cols = this_table.columns.tolist()
    nan_table = pd.DataFrame(np.nan, index=index, columns=cols)
    return nan_table


def nextpow2(x):
    """Helper function to return next largest value that is a power of 2"""
    if x == 0:
        y = 0
    else:
        y = math.ceil(math.log2(x))
    return y


def xcorr(x, y, maxlag):
    """Implementation of MATLAB xcorr

    Taken from: https://stackoverflow.com/a/60245667

    Args:
        x (np.ndarray): First array for correlation
        y (np.ndarray): Second array for correlation
        maxlag (int): Lag range
    Returns:
        corr_arr (np.ndarray): Cross correlation values normalized to
            cross-correlation at lag index 0
    """
    m = max(len(x), len(y))
    mx1 = min(maxlag, m - 1)
    ceilLog2 = nextpow2(2 * m - 1)
    m2 = 2**ceilLog2

    X = scipy.fftpack.fft(x, m2)
    Y = scipy.fftpack.fft(y, m2)
    c1 = np.real(scipy.fftpack.ifft(X * np.conj(Y)))
    index1 = np.arange(1, mx1 + 1, 1) + (m2 - mx1 - 1)
    index2 = np.arange(1, mx1 + 2, 1) - 1
    corr_arr = np.hstack((c1[index1], c1[index2]))

    # Normalizing array to lag at index 0
    corr_arr = corr_arr / c1[index2[0]]
    return corr_arr


def kmeans(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Wrapper for kmeans function that makes sure that kmeans output labels the
    data with higest number, and background as zero, and returns in same
    dimensions as provided"""

    if data.shape[0] < data.shape[1]:
        data = data.T

    model = sklearn.cluster.KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=42,
    )
    data_km = model.fit_predict(data.reshape(-1, 1)).reshape(data.shape)
    data_km = np.squeeze(data_km)

    these_k_vals = np.unique(data_km)

    # Using heuristic, determine background signal vs noise based simply on
    # np.sum(np.abs(x)) per pixel -- whichever is greater should have the higher
    # value

    # Small modification @smanjunath: we arrange labels from smallest value in
    # the data array to largest value in the data array rather than only
    # checking the first and last elements as is performed in the original
    # function as implemented in MATLAB
    kval_data = []
    for kval in these_k_vals:
        indx = np.where(data_km == kval)[0]
        data_km_indx = np.abs(np.sum(data[indx])) / indx.size
        kval_data.append(data_km_indx)

    label_sort = np.argsort(kval_data)
    data_km_fix = np.zeros_like(data_km)

    for idx, label in enumerate(label_sort):
        data_km_fix[data_km == idx] = label

    return data_km_fix


def cull_events_by_channel(
    pd_events_bi: pd.DataFrame, c_bi: typing.List
) -> pd.DataFrame:
    """Select only events from event pandas DataFrame that occur in channels
    found in the list of channels

    Args:
        pd_events_bi (pd.DataFrame): A pandas DataFrame of channel events (rows=events)
        c_bi (list): A list of bipolar channel pairs

    Returns:
        pd_events_bi_s (pd.DataFrame): A pandas DataFrame of channel events that
        includes only those involving the specified channel names
    """

    channels = pd_events_bi["channel"].tolist()
    channel_members = []
    for channel in channels:
        if channel in c_bi:
            channel_members.append(True)
        else:
            channel_members.append(False)
    pd_events_bi_s = pd_events_bi.iloc[channel_members]
    return pd_events_bi_s
