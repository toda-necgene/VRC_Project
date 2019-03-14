"""
製作者:TODA
リアルタイム変換
"""
import atexit
import glob
import os
from multiprocessing import Queue, Process, freeze_support

import chainer
import numpy as np
import pyaudio as pa
import pyworld as pw

from model import Generator
from load_setting import load_setting_from_json

def encode(data):
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
    _f0, _t = pw.dio(data, sampleing_rate)
    _f0 = pw.stonemask(data, _f0, _t, sampleing_rate)
    _sp = pw.cheaptrick(data, _f0, _t, sampleing_rate)
    _ap = pw.d4c(data, _f0, _t, sampleing_rate)
    return _f0, np.clip((np.log(_sp) + 20) / 20, -1.0, 1.0).astype(np.float32), _ap

def decode(_f0, _sp, _ap):
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
    _sp = np.exp(_sp * 20 - 20).astype(np.float)
    return pw.synthesize(_f0, _sp, _ap, 16000)

def load(checkpoint_dir, model):
    """
    モデルのロード
    変数の初期化も同時に行う
    Returns
    -------
    Flagment: bool
        うまくいけばTrue
        ファイルがなかったらFalse
    """
    print(" [I] Reading checkpoint...")
    if os.path.exists(checkpoint_dir):
        print(" [I] file found.")
        _ab = list(glob.glob(checkpoint_dir+"/gen_ab.npz"))[0]
        chainer.serializers.load_npz(_ab, model)
        return True
    print(" [I] dir not found:"+checkpoint_dir)
    return False

def process(queue_in, queue_out, args_model):
    """
    推論（生成）プロセスを行う
    マルチスレッドで動作
    Parameters
    ----------
    queue_in: Queue
    （こちら側からみて）データを受け取り用のキュー
    queue_out: Queue
    （こちら側からみて）データを送る用のキュー
    args_model: dict
    設定の辞書オブジェクト
    """
    net = Generator()
    load(args_model["name_save"], net)
    f0_parameters = np.load("./voice_profile.npy")
    queue_out.put("ok")
    while True:
        if not queue_in.empty():
            _ins = queue_in.get()
            _inputs = np.frombuffer(_ins, dtype=np.int16) / 32767.0
            _inputs = np.clip(_inputs, -1.0, 1.0)
            _f0, _sp, _ap = encode(_inputs.copy())
            _data = _sp.transpose((1, 0)).reshape(1, 513, -1, 1).astype(np.float32)
            _output = net(_data)
            _output = _output.data[0, :, :, 0]
            _output = np.transpose(_output, (1, 0))
            _response = decode((_f0 / f0_parameters[0]) * f0_parameters[1], _output, _ap)
            _response = (np.clip(_response, -1.0, 1.0).reshape(-1) * 32767)
            _response = _response.astype(np.int16)
            _response = _response.tobytes()
            queue_out.put(_response)
if __name__ == '__main__':
    args = load_setting_from_json("setting.json")
    fs = 16000
    channels = 1
    q_in = Queue()
    q_out = Queue()

    p_in = pa.PyAudio()
    p = Process(target=process, args=(q_in, q_out, args))
    p.start()
    while True:
        if not q_out.empty():
            _message = q_out.get()
            if _message == "ok":
                break
    print("Started")
    stream = p_in.open(format=pa.paInt16,
                       channels=1,
                       rate=fs,
                       frames_per_buffer=args["input_size"],
                       input=True,
                       output=True)
    stream.start_stream()
    _output_wave_dammy = np.zeros(args["input_size"], dtype=np.int16).tobytes()
    def terminate():
        """
        緊急終了用プロセス
        """
        stream.stop_stream()
        stream.close()
        q_in.close()
        q_out.close()
        p.terminate()
        p_in.terminate()
        print("Stream stops")
        freeze_support()
        exit(-1)
    atexit.register(terminate)
    while stream.is_active():
        _inputs_source = stream.read(args["input_size"])
        q_in.put(_inputs_source)
        _output_wave = _output_wave_dammy
        if not q_out.empty():
            _output_wave = q_out.get()
        stream.write(_output_wave)
