# eegfilt -  (high|low|band)-pass filter data using two-way least-squares
#            FIR filtering. Optionally uses the window method instead of
#            least-squares. Multiple data channels and epochs supported.
# Usage:
#  >> [smoothdata] = eegfilt(data,srate,locutoff,hicutoff);
#  >> [smoothdata,filtwts] = eegfilt(data,srate,locutoff,hicutoff, ...
#                                    epochframes,filtorder,revfilt,firtype,causal);
# Inputs:
#   data        = (channels,frames*epochs) data to filter
#   srate       = data sampling rate (Hz)
#   locutoff    = low-edge frequency in pass band (Hz)  {0 -> lowpass}
#   hicutoff    = high-edge frequency in pass band (Hz) {0 -> highpass}
#   epochframes = frames per epoch (filter each epoch separately {def/0: data is 1 epoch}
#   filtorder   = length of the filter in points {default 3*fix(srate/locutoff)}
#   revfilt     = [0|1] reverse filter (i.e. bandpass filter to notch filter). {default 0}
#   firtype     = 'firls'|'fir1' {'firls'}
#   causal      = [0|1] use causal filter if set to 1 (default 0)
#
# Outputs:
#    smoothdata = smoothed data
#    filtwts    = filter coefficients [smoothdata <- filtfilt(filtwts,1,data)]

# Original Author: Scott Makeig, Arnaud Delorme, Clemens Brunner SCCN/INC/UCSD, La Jolla, 1997
# Ported to Python by: Shashank Manjunath (initials sxm), Northeastern University, Boston, MA, 2023

# Copyright (C) 4-22-97 from bandpass.m Scott Makeig, SCCN/INC/UCSD, scott@sccn.ucsd.edu
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

# 05-08-97 fixed frequency bound computation -sm
# 10-22-97 added MINFREQ tests -sm
# 12-05-00 added error() calls -sm
# 01-25-02 reformated help & license, added links -ad
# 03-20-12 added firtype option -cb
# 05-17-23 ported to Python, minor bug fixes - sxm

import logging
import typing

import scipy.signal
import numpy as np


__all__ = ["eegfilt"]


def eegfilt(
    data: np.ndarray,
    srate: float,
    locutoff: float,
    hicutoff: float,
    epochframes: int = 0,
    filtorder: int = 0,
    revfilt: int = 0,
    firtype: str = "firls",
    causal: int = 0,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    channels = data.shape[0]
    frames = data.shape[1]

    if channels > 1 and frames == 1:
        raise ValueError("Input data should be a row vector")

    nyq = srate * 0.5
    min_freq = 0

    min_fac = 3  # This many (lo)cutoff-freq cycles in filter
    min_filtorder = 15  # Minimum filter length
    trans = 0.15  # Franctional width of transition zones

    if locutoff > 0 and hicutoff > 0 and locutoff > hicutoff:
        raise ValueError("locutoff must be less than hicutoff")

    if locutoff < 0 or hicutoff < 0:
        raise ValueError("locutoff and hicutoff must be greater than zero")

    if locutoff > nyq:
        raise ValueError("locutoff must be greater than srate / 2")

    if hicutoff > nyq:
        raise ValueError("hicutoff must be greater than srate / 2")

    if firtype == "firls":
        logging.warning(
            "Using firls to estimate filter coefficients. We recommend that \
            you use fir1 instead, which yields larger attenuation. In future, \
            fir1 will be used by default!"
        )

    if filtorder == 0:
        if locutoff > 0:
            filtorder = int(min_fac * np.fix(srate / locutoff))
        elif hicutoff > 0:
            filtorder = int(min_fac * np.fix(srate / hicutoff))

        if filtorder < min_filtorder:
            filtorder = min_filtorder

    if epochframes == 0:
        epochframes = frames

    epochs = int(np.fix(frames / epochframes))

    if epochs * epochframes != frames:
        raise RuntimeError("epochframes does not divide frames")

    if filtorder * 3 > epochframes:
        raise ValueError("epochframes must be at most 3 times the filtorder")

    if (1 + trans) * hicutoff / nyq > 1:
        raise ValueError("hicutoff frequency too close to Nyquist frequency")

    if locutoff > 0 and hicutoff > 0:
        if revfilt:
            logging.info(f"eegfilt - performing {filtorder:d}-point notch filtering.")
        else:
            logging.info(
                f"eegfilt - performing {filtorder:d}-point bandpass filtering."
            )

        logging.info(
            """If a message, ''Matrix is close to singular or badly scaled,''
            appears, then Matlab has failed to design a good filter. As a
            workaround, for band-pass filtering, first highpass the data,
            then lowpass it."""
        )

        if firtype == "firls":
            f = [
                min_freq,
                (1 - trans) * locutoff / nyq,
                locutoff / nyq,
                hicutoff / nyq,
                (1 + trans) * hicutoff / nyq,
                1,
            ]

            lbw = (f[2] - f[1]) * srate / 2
            hbw = (f[4] - f[3]) * srate / 2
            m = [0, 0, 1, 1, 0, 0]
            logging.info(
                f"eegfilt - low transition band width is {lbw:.1f} Hz; high \
                trans. band width, {hbw:.1f} Hz."
            )
        elif firtype == "fir1":
            Wn = np.asarray([locutoff, hicutoff]) / (srate / 2)
            filtwts = scipy.signal.firwin(
                filtorder + 1, Wn, window="hamming", pass_zero="bandpass"
            )

    elif locutoff > 0:
        # TODO: Not tested!
        if locutoff / nyq < min_freq:
            raise ValueError(
                f"eegfilt - highpass cutoff freq must be > {min_freq*nyq:.3f} Hz"
            )
        logging.info(f"eegfilt - performing {filtorder:d}-point highpass filtering.")

        if firtype == "firls":
            f = [min_freq, (1 - trans) * locutoff / nyq, locutoff / nyq, 1]
            hbw = (f[2] - f[1]) * srate / 2
            logging.info(f"eegfilt - highpass transition band width is {hbw:.1f} Hz.")
            m = [0, 0, 1, 1]
        elif firtype == "fir1":
            Wn = locutoff / (srate / 2)
            filtwts = scipy.signal.butter(filtorder, Wn, "high")

    elif hicutoff > 0:
        # TODO: Not tested!
        if hicutoff / nyq < min_freq:
            raise ValueError(
                f"eegfilt - lowpass cutoff freq must be > {min_freq*nyq:.3f} Hz"
            )
        logging.info(f"eegfilt - performing {filtorder:d}-point lowpass filtering.")

        if firtype == "firls":
            f = [min_freq, hicutoff / nyq, hicutoff * (1 + trans) / nyq, 1]
            lbw = (f[2] - f[1]) * srate / 2
            logging.info(f"eegfilt() - lowpass transition band width is {lbw} Hz.")
            m = [1, 1, 0, 0]
        elif firtype == "fir1":
            Wn = hicutoff / (srate / 2)
            filtwts = scipy.signal.butter(filtorder, Wn, "low")

    else:
        raise ValueError("You must provide a non-0 low or high cut-off frequency")

    if revfilt:
        # TODO: Not tested!
        if firtype == "fir1":
            raise ValueError("Cannot reverse filter using ''fir1'' option")
        else:
            # TODO: Double check what revfilt does exactly
            m = [~i + 2 for i in m]

    if firtype == "firls":
        # TODO: Not tested!
        filtwts = scipy.signal.firls(filtorder, f, m)

    smoothdata = np.zeros((channels, frames))

    for e in range(epochs):
        for c in range(channels):
            epoch_start = e * epochframes
            epoch_end = (e + 1) * epochframes + 1
            epoch_data = data[c, epoch_start:epoch_end]

            if causal:
                fdata = scipy.signal.lfilter(filtwts, 1, epoch_data)
            else:
                fdata = scipy.signal.filtfilt(filtwts, 1, epoch_data)

            smoothdata[c, epoch_start:epoch_end] = fdata

    return smoothdata, filtwts
