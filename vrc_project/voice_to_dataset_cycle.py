"""
    製作者:ixsiid
    改変者:TODA
    データセットを作成するモジュールです
"""
import os
import wave
import glob
import numpy as np
from vrc_project.world_and_wave import wave2world
def create_dataset(_term, _chunk=1024, delta=0):
    """
    データセットを作成します
    """

    INPUT_NAMES = ["A", "B"]
    WAVE_INPUT_DIR = os.path.join("dataset", "train")
    OUTPUT_DIR = os.path.join(".", "dataset", "patch")

    pitch = dict()
    dataset_to_return = list()
    for name in INPUT_NAMES:
        wave_input_file_names = os.path.join(WAVE_INPUT_DIR, name)
        files = sorted(glob.glob(os.path.join(wave_input_file_names, "*.wav")))
        memory_spec_env = list()
        _ff = list()
        for file in files:
            print(" [*] converting wave to patchdata :", file)
            dms = []
            _wf = wave.open(file, 'rb')
            dds = _wf.readframes(_chunk)
            while dds != b'':
                dms.append(dds)
                dds = _wf.readframes(_chunk)
            dms = b''.join(dms)
            data = np.frombuffer(dms, 'int16')
            data_real = (data / 32767).reshape(-1)
            _step = _term
            if delta > 0:
                _step = delta
            times = data_real.shape[0] // _step + 1
            if data_real.shape[0] % _step == 0:
                times -= 1
            for i in range(times):
                startpos = _step * i + data_real.shape[0] % _step
                data_real_current_use = data_real[max(startpos - _term, 0):startpos].copy()
                _padiing_size = _term - data_real_current_use.shape[0]
                if _padiing_size > 0:
                    data_real_current_use = np.pad(data_real_current_use, (_padiing_size, 0), "constant")
                f0_estimation, spec_env, _ = wave2world(data_real_current_use)
                f0_estimation = f0_estimation[f0_estimation > 0.0]
                if f0_estimation.shape[0] != 0:
                    _ff.extend(f0_estimation)
                spec_env = np.transpose(spec_env, [1, 0]).reshape(513, spec_env.shape[0], 1)
                # ap = np.transpose(ap, [1, 0]).reshape(513, ap.shape[0], 1)
                # spec = np.concatenate([spec_env, ap], axis=2).reshape(ap.shape[0], ap.shape[1], 2)
                memory_spec_env.append(spec_env)
        _m = np.asarray(memory_spec_env, dtype=np.float32)
        dataset_to_return.append(_m)
        np.save(os.path.join(OUTPUT_DIR, name + ".npy"), _m)
        print(" [I] " + name + " directory has been finished successfully.")
        pitch[name] = dict()
        pitch[name]["mean"] = np.mean(_ff)
        pitch[name]["var"] = np.var(_ff)
    '''
    基本周波数F0の変換に使用するパラメータの割り出し
    ちなみに計算式は
    $$$
    F_(0t) = (F_(0s) - mean(F_(0s))) / var(F_(0s)) * var(F_(0t)) + mean(F_(0t))
    $$$
    意味:平均と分散の振り直し
    F0の分布はおおよそ標準分布であるため
    スケールについても議論の余地あり
    '''
    pitch_mean_s = pitch[INPUT_NAMES[0]]["mean"]
    pitch_var_s = pitch[INPUT_NAMES[0]]["var"]
    pitch_mean_t = pitch[INPUT_NAMES[1]]["mean"]
    pitch_var_t = pitch[INPUT_NAMES[1]]["var"]
    np.savez(os.path.join(".", "voice_profile.npz"), pre_sub=pitch_mean_s, pitch_rate=pitch_var_t/pitch_var_s, post_add=pitch_mean_t)
    return dataset_to_return[0], dataset_to_return[1]
