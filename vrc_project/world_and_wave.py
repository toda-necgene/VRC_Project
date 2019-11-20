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
    _cepstrum = (np.log(_cepstrum) + 7) / 9
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

    _cepstrum = np.exp(_cepstrum * 9 - 7)
    _cepstrum = _cepstrum.astype(np.float).copy("C")
    _aperiodicity = _aperiodicity.astype(np.float)
    return pw.synthesize(_f0, _cepstrum, _aperiodicity, 44100, frame_period=10)
def fft(_data, nfft=2048):
    """
    stftを計算

     Parameters
    ----------
    _data: np.ndarray
        音声データ
        range  : [-1.0,1.0]
        dtype  : float64
    Returns
    -------
    spec_po: np.ndarray
        パワースペクトラム
        power-spectram
        Shape : (n,512)
    """
    shift = nfft //2
    time_ruler = _data.shape[0] // shift
    if _data.shape[0] % shift == 0:
        time_ruler -= 1
    window = np.hamming(nfft)
    pos = 0
    wined = np.zeros([time_ruler, nfft])
    for fft_index in range(time_ruler):
        frame = _data[pos:pos + nfft]
        padding_size = nfft-frame.shape[0]
        if padding_size > 0:
            frame = np.pad(frame, (0, padding_size), "constant")
        wined[fft_index] = frame * window
        pos += shift
    fft_r = np.fft.fft(wined, n=nfft, axis=1)
    spec_re = fft_r.real.reshape(time_ruler, -1)
    spec_im = fft_r.imag.reshape(time_ruler, -1)
    spec_po = np.log(np.power(spec_re, 2) + np.power(spec_im, 2) + 1e-8).reshape(time_ruler, -1)[:, -512:]
    spec_po = np.clip((spec_po + 5) / 10, -1.0, 1.0)
    return spec_po
