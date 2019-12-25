"""
wave and acaustic-feature functions.
# 音響特徴量ユーティリティー
"""
import pyworld as pw
import numpy as np
import wave
SAMPLEING_RATE = 44100
OUT_DIMENTION = 64
FRAME_PERIOD = 10
ENVELOPE_FFT_SIZE = 2048

def load_wave_file(_path_to_file):
    """
    Parameters
    ----------
    _path_to_file: str
    Returns
    -------
    _data: int16
    """
    wave_data = wave.open(_path_to_file, "rb")
    _data = np.zeros([1], dtype=np.int16)
    dds = wave_data.readframes(1024)
    while dds != b'':
        _data = np.append(_data, np.frombuffer(dds, "int16"))
        dds = wave_data.readframes(1024)
    wave_data.close()
    _data = _data[1:]
    return _data

def wave2world_lofi(data):
    """
    f0-estimation : dio + stonemask
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
    _f0, _t = pw.dio(data, SAMPLEING_RATE, frame_period=FRAME_PERIOD)
    _f0 = pw.stonemask(data, _f0, _t, SAMPLEING_RATE)
    _cepstrum = pw.cheaptrick(data, _f0, _t, SAMPLEING_RATE)
    _cepstrum = (np.log(_cepstrum) + 7) / 9
    _cepstrum = np.clip(_cepstrum, -1.0, 1.0)
    _aperiodicity = pw.d4c(data, _f0, _t, SAMPLEING_RATE)
    return _f0, _cepstrum.astype(np.float32), _aperiodicity
def wave2world_hifi(data):
    """
    f0-estimation : harvest
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
    _f0, _t = pw.harvest(data, SAMPLEING_RATE, frame_period=FRAME_PERIOD)
    _cepstrum = pw.cheaptrick(data, _f0, _t, SAMPLEING_RATE)
    _cepstrum = (np.log(_cepstrum) + 7) / 9
    _cepstrum = np.clip(_cepstrum, -1.0, 1.0)
    _aperiodicity = pw.d4c(data, _f0, _t, SAMPLEING_RATE)
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
    _cepstrum = _cepstrum = np.exp(_cepstrum * 9 - 7).astype(np.float).copy("C")
    _aperiodicity = _aperiodicity.astype(np.float)
    return pw.synthesize(_f0, _cepstrum, _aperiodicity, SAMPLEING_RATE, frame_period=FRAME_PERIOD)
def fft(_data, nfft=2048, out_dimention=512):
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
    spec_po = np.log(np.power(spec_re, 2) + np.power(spec_im, 2) + 1e-8).reshape(time_ruler, -1)[:, -out_dimention:]
    spec_po = np.clip((spec_po + 5) / 10, -1.0, 1.0)
    return spec_po
