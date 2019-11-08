"""
wave and acaustic-feature functions.
# 音響特徴量ユーティリティー
"""
import pyworld as pw
import numpy as np
def wave2world(data):
    """
    Parameters
    ----------
    data : float64
        SamplingRate: 44100
        ValueRange  : [-1.0,1.0]
        Shape: (input_size)
    Returns
    -------
    _f0 : float64
        Shape: (N)
    _cepstrum : float64
        Shape: (N, 64)
    _aperiodicity : float64
        Shape: (N,513)
    NOTE: input_size is defined in config file.
          N is determined by input_size.
    """
    sampling_rate = 44100
    _f0, _t = pw.dio(data, sampling_rate, frame_period=10)
    _f0 = pw.stonemask(data, _f0, _t, sampling_rate)
    _cepstrum = pw.cheaptrick(data, _f0, _t, sampling_rate)
    _cepstrum = (np.log(_cepstrum) + 10) / 10
    _cepstrum = np.clip(_cepstrum, -1.0, 1.0)
    _aperiodicity = pw.d4c(data, _f0, _t, sampling_rate)
    return _f0, _cepstrum.astype(np.float32), _aperiodicity

def world2wave(_f0, _cepstrum, _aperiodicity):
    """
    Parameters
    ----------
    _f0 : float64
        Shape: (N)
    _cepstrum : float64
        Shape: (N, 64)
    _aperiodicity : float64
        Shape: (N, 513)
    Returns
    -------
    wave: float64
        SamplingRate: 44100
        ValueRange  : [-1.0,1.0]
    """

    _cepstrum = np.exp(_cepstrum * 10 - 10)
    _cepstrum = _cepstrum.astype(np.float).copy("C")
    _aperiodicity = _aperiodicity.astype(np.float)
    return pw.synthesize(_f0, _cepstrum, _aperiodicity, 44100, frame_period=10)
