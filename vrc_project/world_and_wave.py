"""
製作者:TODA

音響特徴量ユーティリティー
"""
import pyworld as pw
import numpy as np
def wave2world(data):
    """
    #音声をWorldに変換します

    Parameters
    ----------
    data : ndarray
        入力データ
        SamplingRate: 16000
        ValueRange  : [-1.0,1.0]
        dtype       : float64
    Returns
    -------
    World: list(3items)
        出力
        _f0 : f0 estimation
        Shape(N)
        dtype       : float64
        _sp : spectram envelobe
        Shape(N,513)
        dtype       : float64
        _ap : aperiodicity
        Shape(N,513)
        dtype       : float64
    """
    sampleing_rate = 16000
    _f0, _t = pw.dio(data, sampleing_rate, frame_period=10)
    _f0 = pw.stonemask(data, _f0, _t, sampleing_rate)
    _sp = pw.cheaptrick(data, _f0, _t, sampleing_rate)
    _sp = pw.code_spectral_envelope(_sp, sampleing_rate, 64)
    _ap = pw.d4c(data, _f0, _t, sampleing_rate)
    # _sp = np.log(_sp)
    # _sp = (np.log(_sp) + 15) / 20
    #_sp = np.clip(_sp, -1.0, 1.0)
    return _f0, _sp.astype(np.float32), _ap

def world2wave(_f0, _sp, _ap):
    """
    #Worldを音声に変換します
    Parameters
    ----------
    _f0 : np.ndarray
        _f0 estimation
        Shape(N)
        dtype       : float64
    _sp : np.ndarray
        spectram envelobe
        Shape(N,513)
        dtype       : float64
    _ap : np.ndarray
        aperiodicity
        Shape(N,513)
        dtype       : float64
    Returns
    -------
    World: list(3items)
        #出力
        SamplingRate: 16000
        ValueRange  : [-1.0,1.0]
        dtype       : float64
    """
    #_sp = np.exp(_sp * 20 - 15).astype(np.float)
    # _sp = np.exp(_sp)
    _sp = pw.decode_spectral_envelope(_sp.astype(np.float), 16000, 1024)
    _ap = _ap.astype(np.float)
    return pw.synthesize(_f0, _sp, _ap, 16000, frame_period=10)
