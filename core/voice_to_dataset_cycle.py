"""
    製作者:ixsiid
    改変者:TODA
    データセットを作成するモジュールです
"""
import os
import wave
import glob
import numpy as np
from core.world_and_wave import load_wave_file
def create_dataset(wave2world_function):
    """
    create patch files
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
            data_real = load_wave_file(file) / 32767
            f0_estimation, spec_env, _ = wave2world_function(data_real)
            f0_estimation = f0_estimation[f0_estimation > 0.0]
            _ff.extend(f0_estimation)
            spec_env = spec_env.reshape(spec_env.shape[0], spec_env.shape[1], 1)
            memory_spec_env.append(spec_env)
        _m = np.asarray(memory_spec_env, dtype=np.float32).reshape(-1, spec_env.shape[1], 1)
        dataset_to_return.append(_m)
        np.save(os.path.join(OUTPUT_DIR, name + ".npy"), _m)
        print(" [I] voice in " + name + " directory has been finished successfully.")
        pitch[name] = dict()
        pitch[name]["mean"] = np.mean(_ff)
        pitch[name]["std"] = np.std(_ff)
    pitch_mean_s = pitch[INPUT_NAMES[0]]["mean"]
    pitch_std_s = pitch[INPUT_NAMES[0]]["std"]
    pitch_mean_t = pitch[INPUT_NAMES[1]]["mean"]
    pitch_std_t = pitch[INPUT_NAMES[1]]["std"]
    np.savez(os.path.join(".", "voice_profile.npz"), pre_sub=pitch_mean_s, pitch_rate=pitch_std_t/pitch_std_s, post_add=pitch_mean_t)
    return dataset_to_return[0], dataset_to_return[1]
