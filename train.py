"""
製作者:TODA
学習スクリプト
設定パラメーターの意味はsetting_loader.pyを参照
"""
import os
import shutil
import wave
import chainer
import numpy as np
from vrc_project.model import Discriminator, Generator
from vrc_project.seq_dataset import SeqData
from vrc_project.updater import CycleGANUpdater
from vrc_project.voice_to_dataset_cycle import create_dataset
from vrc_project.setting_loader import load_setting_from_json
from vrc_project.eval import TestModel
from vrc_project.notify import LineNotify


def load_model_from_npz(_checkpoint_dir, _trainer):
    """
    Parameters
    ----------
    _checkpoint_dir: str
    _trainer: chainer.training.trainer
    Returns
    -------
    flag: bool
    """
    print(" [*] Reading checkpoint...")
    if os.path.exists(_checkpoint_dir) and os.path.exists(_checkpoint_dir+"/snapshot.npz"):
        print(" [I] checkpoint is found. loading file name : %s " % (_checkpoint_dir))
        chainer.serializers.load_npz(_checkpoint_dir+"/snapshot.npz", _trainer)
        print(" [I] loaded checkpoint successfully.")
        return True
    elif not os.path.exists(_checkpoint_dir):
        os.makedirs(_checkpoint_dir)
    print(" [I] checkpoint is not found.")
    return False
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


def dataset_pre_process_controler(args):
    """
    Parameters
    ----------
    _args: dict
    Returns
    -------
    _train_iter_a: chainer.iterators.Iterator
    _train_iter_b: chainer.iterators.Iterator
    _voice_profile: dict (float64)
        f0 parameters.
        keys: (pre_sub, pitch_rate, postad)
    _length_sp: int
    """
    _sounds_a = None
    _sounds_b = None
    if not (args["use_old_dataset"] and os.path.exists("./dataset/patch/A.npy") and os.path.exists("./dataset/patch/B.npy")):
        _sounds_a, _sounds_b = create_dataset(args["input_size"])
    else:
        # preparing training-data
        print(" [*] loading data-set ...")
        _sounds_a = np.load("./dataset/patch/A.npy")
        _sounds_b = np.load("./dataset/patch/B.npy")
        print(" [I] loaded data-set successfully.")
    _length_sp = 200
    if args["gpu"] >= 0:
        _sounds_a = chainer.backends.cuda.to_gpu(_sounds_a)
        _sounds_b = chainer.backends.cuda.to_gpu(_sounds_b)
    _train_iter_a = chainer.iterators.MultithreadIterator(SeqData(_sounds_a, 200), args["batch_size"], shuffle=True, n_threads=4)
    _train_iter_b = chainer.iterators.MultithreadIterator(SeqData(_sounds_b, 200), args["batch_size"], shuffle=True, n_threads=4)
    # f0 parameters(基本周波数F0の変換に使用する定数。詳しくは./vrc_project/voice_to_dataset_cycle.py L65周辺)
    _voice_profile = np.load("./voice_profile.npz")
    if not os.path.exists(args["name_save"]):
        os.mkdir(args["name_save"])
    shutil.copy("./voice_profile.npz", args["name_save"]+"/voice_profile.npz")
    return _train_iter_a, _train_iter_b, _voice_profile, _length_sp

if __name__ == '__main__':
    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(512*1024*1024)
    _args = load_setting_from_json("setting.json")
    if  _args["wave_otp_dir"] is not "False":
        _args["wave_otp_dir"] = _args["wave_otp_dir"] + _args["model_name"] +  _args["version"]+"/"
        if not os.path.exists(_args["wave_otp_dir"]):
            os.makedirs(_args["wave_otp_dir"])
    train_iter_a, train_iter_b, voice_profile, length_sp = dataset_pre_process_controler(_args)
    g_a_to_b = Generator()
    g_b_to_a = Generator()
    d_a = Discriminator()
    if _args["gpu"] >= 0:
        chainer.cuda.Device(_args["gpu"]).use()
        g_a_to_b.to_gpu()
        g_b_to_a.to_gpu()
        d_a.to_gpu()
    g_optimizer_ab = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5).setup(g_a_to_b)
    g_optimizer_ba = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5).setup(g_b_to_a)
    d_optimizer_a = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5).setup(d_a)
    # main training
    updater = CycleGANUpdater(
        model={"main":g_a_to_b, "inverse":g_b_to_a, "disa":d_a},
        max_itr=_args["train_iteration"],
        f0_param=voice_profile,
        iterator={"main":train_iter_a, "data_b":train_iter_b},
        optimizer={"gen_ab":g_optimizer_ab, "gen_ba":g_optimizer_ba, "disa":d_optimizer_a},
        device=_args["gpu"])
    _trainer = chainer.training.Trainer(updater, (_args["train_iteration"], "iteration"), out=_args["name_save"])
    load_model_from_npz(_args["name_save"], _trainer)
    display_interval = (_args["log_interval"], 'iteration')
    if _args["test"]:
        test = load_wave_file("./dataset/test/test.wav") / 32767.0
        _label_sample = load_wave_file("./dataset/test/label.wav") / 32767.0
        _trainer.extend(TestModel(_trainer, _args, [test, _label_sample, voice_profile], length_sp, "itrs"), trigger=display_interval)
        if _args["line_notify"]:
            with open("line_api_token.txt", "rb") as s:
                key = s.readline().decode("utf8")
                tri = chainer.training.triggers.ManualScheduleTrigger([100, 500, 1000, 5000, 10000, 15000], "iteration")
                _trainer.extend(LineNotify(_trainer, key), trigger=tri)
    _trainer.extend(chainer.training.extensions.snapshot(filename='snapshot.npz', num_retain=2), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_a_to_b, 'gen_ab.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.LogReport(trigger=display_interval))
    _trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))
    rep_list = ['iteration', 'D_B_FAKE', 'G_AB__GAN', 'G_ABA_CYC', "env_test_loss", "test_loss"]
    _trainer.extend(chainer.training.extensions.PrintReport(rep_list), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.PlotReport(["env_test_loss"], filename="env.png"), trigger=display_interval)
    _trainer.run()
    print(" [*] All over.")
    