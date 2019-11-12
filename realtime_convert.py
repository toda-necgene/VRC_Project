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

noise_gate_rate = 0.1
gain = 1.8

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
        chainer.serializers.load_npz(checkpoint_dir+"/gen_ab.npz", m_1)
        return True
    print(" [I] dir not found:"+checkpoint_dir)
    return False

def process(queue_in, queue_out, args_model, noise_gate, noise_defact, noise_gate_rate):
    """
    推論（生成）プロセスを行う
    マルチスレッドで動作
    Parameters
    ----------
    queue_in: Queue
    （こちら側からみて）データ受信用のキュー
    queue_out: Queue
    （こちら側からみて）データ送信用のキュー
    args_model: dict
    設定の辞書オブジェクト
    """
    net_1 = Generator()
    load(args_model["name_save"], net_1)
    f0_parameters = np.load(args_model["name_save"]+"/voice_profile.npz")
    chainer.using_config("train", False)
    chainer.config.autotune = True
    queue_out.put("ok")
    while True:
        if not queue_in.empty():
            _ins = queue_in.get()
            _sp = _ins[1]
            _data = (_sp - noise_gate * noise_gate_rate + noise_defact * noise_gate_rate).reshape([1, 200, 1025, 1])    
            _output = net_1(_data)
            _output = _output[0].data[:, :, 0]
            _f0 = _ins[0]
            _f0 = ((_f0 - f0_parameters["pre_sub"]) * f0_parameters["pitch_rate"] + f0_parameters["post_add"]) * np.sign(_f0)
            _f0 = _f0.reshape(-1)
            _response = world2wave(_f0, _output, _ins[2])
            _response = (np.clip(_response, -1.0, 1.0).reshape(-1) * 32767)
            _response = _response.astype(np.int16)
            queue_out.put(_response)
if __name__ == '__main__':
    args = load_setting_from_json("setting.json")
    fs = 44100
    channels = 1
    term_sec = 8192
    q_in = Queue()
    q_out = Queue()
    p_in = pa.PyAudio()
    stream = p_in.open(format=pa.paInt16,
                       channels=1,
                       rate=fs,
                       frames_per_buffer=term_sec*2,
                       input=True,
                       output=True)
    stream.start_stream()
    let = np.zeros([term_sec*3])
    for n in range(3):
        _inputs_source = stream.read(term_sec)
        _inputs = np.frombuffer(_inputs_source, dtype=np.int16) / 32767.0
        let[n*term_sec:term_sec+n*term_sec] = _inputs
    _f0, _sp, _ap = wave2world(let.astype(np.float))
    noise_gate = np.mean(_sp, axis=0, keepdims=True)
    noise_defact = np.mean(noise_gate)
    p = Process(target=process, args=(q_in, q_out, args, noise_gate, noise_defact, noise_gate_rate))
    p.start()
    while True:
        if not q_out.empty():
            _message = q_out.get()
            if _message == "ok":
                break
    
    print("Started")
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
    _output_wave_dammy = np.zeros(term_sec, dtype=np.int16)
    wave_holder = np.zeros([args["input_size"]]) 
    while stream.is_active():
        tts = time.time()
        _inputs_source = stream.read(term_sec)
        _inputs = np.frombuffer(_inputs_source, dtype=np.int16) / 32767.0
        wave_holder = np.append(wave_holder, _inputs)[-args["input_size"]:]
        _inputs_wave = np.clip(wave_holder* gain, -1.0, 1.0)
        _f0, _sp, _ap = wave2world(_inputs_wave)
        if np.max(_inputs_wave) >= 0.2:
            q_in.put([_f0, _sp, _ap])
        _output_wave = _output_wave_dammy
        if not q_out.empty():
            _output_wave = q_out.get()[-term_sec:]
        _output = _output_wave.tobytes()
        stream.write(_output)
        pow_in = np.max(np.abs(_inputs_wave))
        pow_out = np.mean(np.abs(_output_wave/32767))
        print("\r audio-power input:{:<6.4f}, output:{:<6.4f} wave_process_time:{:<6.4f} queue_length{:0>3}".format(pow_in, pow_out, time.time()-tts, q_in.qsize()), end="")
        