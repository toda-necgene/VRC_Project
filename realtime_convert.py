"""
製作者:TODA
リアルタイム変換
"""
import atexit
import os
import time
from multiprocessing import Queue, Process, freeze_support

import torch
import numpy as np
import pyaudio as pa

from setting import get_setting
from core.model import Generator
from core.world_and_wave import world2wave, wave2world_lofi, wave2world_hifi

# experimental
noise_gate_rate = 0.0
gain = 1.0

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
        print(" [O] file found.")
        m_1.load_state_dict(torch.load(checkpoint_dir+"/gen_ab.pth"))
        return True
    print(" [X] dir not found:"+checkpoint_dir)
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
    net_1.eval()
    load(args_model["name_save"], net_1)
    f0_parameters = np.load(args_model["name_save"]+"/voice_profile.npz")
    queue_out.put("ok")
    while True:
        if not queue_in.empty():
            _ins = queue_in.get()
            _sp = _ins[1]
            _data = (_sp - noise_gate * noise_gate_rate + noise_defact * noise_gate_rate).reshape([1, 200, 1025, 1])
            _data = torch.Tensor(_data.transpose([0, 2, 1, 3]))
            _output = net_1(_data)
            _output = _output[0].detach().numpy()[:, :, 0].transpose([1, 0])
            _f0 = _ins[0]
            _f0 = ((_f0 - f0_parameters["pre_sub"]) * f0_parameters["pitch_rate"] + f0_parameters["post_add"]) * np.sign(_f0)
            _f0 = _f0.reshape(-1)
            _response = world2wave(_f0, _output, _ins[2])
            _response = (np.clip(_response, -1.0, 1.0).reshape(-1) * 32767)
            _response = _response.astype(np.int16)
            queue_out.put(_response)
if __name__ == '__main__':
    args = get_setting()
    fs = 44100
    term_sec = 44100
    q_in = Queue()
    q_out = Queue()
    paudio = pa.PyAudio()
    wave2world_process = wave2world_lofi if args["f0_estimation_plan"] is "dio" else wave2world_hifi
    di = 0
    do = 0
    input_devices = []
    input_device_indexis = []
    output_devices = []
    output_device_indexis = []

    for n in range(paudio.get_device_count()):
        d = paudio.get_device_info_by_index(n)
        if d["maxInputChannels"]!=0:
            ns = {"name":d["name"], "samplerate": d["defaultSampleRate"]}
            input_devices.append(ns)
            input_device_indexis.append(n)
        elif d["maxOutputChannels"]!=0:
            ns = {"name":d["name"], "samplerate": d["defaultSampleRate"]}
            output_devices.append(ns)
            output_device_indexis.append(n)
    print("[I] Input device list")
    print("---------------")
    for i,n in enumerate(input_devices):
        print("[{}]".format(i),n)
    print("---------------")
    print("[?] input device select(number)")
    di = int(input())
    ddi = input_device_indexis[di]
    print("[I] Output device list")
    print("---------------")
    for i,n in enumerate(output_devices):
        print("[{}]".format(i), n)
    print("---------------")
    print("[?] output device select(number)")
    do = int(input())
    ddo = output_device_indexis[do]
    print("[I] input selected({})".format(paudio.get_device_info_by_index(ddi)["name"]))
    print("[I] output selected({})".format(paudio.get_device_info_by_index(ddo)["name"]))
    stream = paudio.open(format=pa.paInt16,
                       channels=1,
                       rate=fs,
                       frames_per_buffer=term_sec,
                       input=True,
                       output=True,
                       input_device_index=ddi,
                       output_device_index=ddo)
    stream.start_stream()
    let = np.zeros([term_sec*3])
    for n in range(3):
        _inputs_source = stream.read(term_sec)
        _inputs = np.frombuffer(_inputs_source, dtype=np.int16) / 32767.0
        let[n*term_sec:term_sec+n*term_sec] = _inputs
    _f0, _sp, _ap = wave2world_lofi(let.astype(np.float))
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
        q_out.close()
        p.terminate()
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
        _inputs_wave = wave_holder* gain
        _f0, _sp, _ap = wave2world_process(_inputs_wave)
        if np.max(_inputs_wave) >= 0.0:
            q_in.put([_f0, _sp, _ap])
        _output_wave = _output_wave_dammy
        if not q_out.empty():
            _output_wave = q_out.get()[-term_sec:]
        _output = _output_wave.tobytes()
        stream.write(_output)
        pow_in = np.max(np.abs(_inputs_wave))
        pow_out = np.mean(np.abs(_output_wave/32767))
        print("\r audio-power input:{:<6.4f}, output:{:<6.4f} wave_process_time:{:<6.4f} queue_length{:0>3}".format(pow_in, pow_out, time.time()-tts, q_in.qsize()), end="")

