"""
製作者:TODA
リアルタイム変換
"""
import atexit
import os
import time
from multiprocessing import Queue, Process, freeze_support

import chainer
import numpy as np
import pyaudio as pa

from vrc_project.model import Generator
from vrc_project.setting_loader import load_setting_from_json
from vrc_project.world_and_wave import wave2world, world2wave

def load(checkpoint_dir, m_1):
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
        chainer.serializers.load_npz(checkpoint_dir+"/gen_ab1.npz", m_1)
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
    net_1 = Generator()
    load(args_model["name_save"], net_1)
    f0_parameters = np.load(args_model["name_save"]+"/voice_profile.npz")
    queue_out.put("ok")
    while True:
        if not queue_in.empty():
            _ins = queue_in.get()
            _inputs = np.clip(_ins, -1.0, 1.0)
            _f0, _sp, _ap = wave2world(_inputs.copy())
            _data = np.transpose(_sp, (1, 0))
            _data = _data.reshape([1, 64, 200, 1])
            _output = net_1(_data)
            _output = _output[0].data[0, :, :, 0]
            _output = np.clip(np.transpose(_output, (1, 0)), -20.0, 1.0)
            _response = world2wave((_f0 - f0_parameters["pre_sub"]) * f0_parameters["pitch_rate"] + f0_parameters["post_add"], _output, _ap)
            _response = (np.clip(_response, -1.0, 1.0).reshape(-1) * 32767)
            _response = _response.astype(np.int16)
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
                       frames_per_buffer=args["input_size"]//4,
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
    _input_holder = np.zeros(args["input_size"])
    while stream.is_active():
        tts = time.time()
        _inputs_source = stream.read(args["input_size"]//4)
        _inputs = np.frombuffer(_inputs_source, dtype=np.int16) / 32767.0
        _input_holder = np.concatenate([_input_holder, _inputs])[-args["input_size"]:]
        q_in.put(_input_holder)
        _output_wave = _output_wave_dammy
        if not q_out.empty():
            _output_wave = q_out.get()[-args["input_size"]//4:].tobytes()
        stream.write(_output_wave)
        print("wave_process_time:", time.time()-tts, "buffer", q_in.qsize())
