"""
    製作者:ixiid
    改変者:TODA
    データセットを作成するモジュールです
"""
import os
import wave
import glob
import numpy as np
import pyworld.pyworld as pw

def create_dataset(_term, _chunk=1024):
    """
    データセットを作成します
    """

    INPUT_NAMES = ["A", "B"]
    WAVE_INPUT_DIR = os.path.join("dataset", "source")
    TRAIN_DIR = os.path.join(".", "dataset", "train")

    pitch = dict()
    dataset_to_return = list()
    for name in INPUT_NAMES:
        wave_input_file_names = os.path.join(WAVE_INPUT_DIR, name)
        files = sorted(glob.glob(os.path.join(wave_input_file_names, "*.wav")))
        memory_spec_env = list()
        _ff = list()
        for file in files:
            print(" [*] パッチデータに変換を開始します。 :", file)
            dms = []
            _wf = wave.open(file, 'rb')
            dds = _wf.readframes(_chunk)
            while dds != b'':
                dms.append(dds)
                dds = _wf.readframes(_chunk)
            dms = b''.join(dms)
            data = np.frombuffer(dms, 'int16')
            data_real = (data / 32767).reshape(-1).astype(np.float)
            times = data_real.shape[0] // _term + 1
            if data_real.shape[0] % _term == 0:
                times -= 1
            for i in range(times):
                startpos = _term * i + data_real.shape[0] % _term
                data_real_current_use = data_real[max(startpos - _term, 0):startpos].copy()
                _padiing_size = _term - data_real_current_use.shape[0]
                if _padiing_size > 0:
                    data_real_current_use = np.pad(data_real_current_use, (_padiing_size, 0), "constant")
                _f0, _t = pw.dio(data_real_current_use, 16000)
                f0_estimation = pw.stonemask(data_real_current_use, _f0, _t, 16000)
                spec_env = pw.cheaptrick(data_real_current_use, f0_estimation, _t, 16000)
                f0_estimation = f0_estimation[f0_estimation > 0.0]
                if f0_estimation.shape[0] != 0:
                    _ff.extend(f0_estimation)
                spec_env = np.transpose(spec_env, [1, 0])
                memory_spec_env.append(np.clip((np.log(spec_env) + 15.0) / 20, -1.0, 1.0))
        _m = np.asarray(memory_spec_env, dtype=np.float32)
        dataset_to_return.append(_m)
        np.save(os.path.join(TRAIN_DIR, name + ".npy"), _m)
        print(" [*] " + name + "データ変換完了")
        pitch[name] = {}
        pitch[name]["mean"] = np.mean(_ff)
        pitch[name]["var"] = np.std(_ff)

    pitch_mean_s = pitch[INPUT_NAMES[0]]["mean"]
    pitch_var_s = pitch[INPUT_NAMES[0]]["var"]
    pitch_mean_t = pitch[INPUT_NAMES[1]]["mean"]
    pitch_var_t = pitch[INPUT_NAMES[1]]["var"]

    plof = np.asarray([pitch_mean_s, pitch_mean_t, pitch_var_t / pitch_var_s])
    np.save(os.path.join(".", "voice_profile.npy"), plof)
    return dataset_to_return[0], dataset_to_return[1]
