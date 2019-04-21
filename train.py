"""
製作者:TODA


実行すれば学習ができる。
設定はsetting.jsonを利用する。
設定のパラメーターはsetting_loader.pyを参照
"""
import os
import shutil
import wave
import chainer
import numpy as np
from vrc_project.model import Discriminator, Generator
from vrc_project.updater import CycleGANUpdater
from vrc_project.voice_to_dataset_cycle import create_dataset
from vrc_project.setting_loader import load_setting_from_json
from vrc_project.eval import TestModel
from vrc_project.noisy_dataset import Noisy_dataset


def load_model_from_npz(_checkpoint_dir, _trainer):
    """
    モデルのロード
    Parameters
    ----------
    _checkpoint_dir: str
        モデルのロード対象ディレクトリ
    _trainer: chainer.training.trainer
        ロードさせる対象のtrainer
    Returns
    -------
    ResultFlagment: bool
        ロードが完了したか
    """
    print(" [I] Reading checkpoint...")
    if os.path.exists(_checkpoint_dir) and os.path.exists(_checkpoint_dir+"/snapshot.npz"):
        print(" [I] load file name : %s " % (_checkpoint_dir))
        chainer.serializers.load_npz(_checkpoint_dir+"/snapshot.npz", _trainer)
        chainer.serializers.load_npz(_checkpoint_dir+"/gen_ab1.npz", _trainer.updater.gen_ab1)
        chainer.serializers.load_npz(_checkpoint_dir+"/gen_ba1.npz", _trainer.updater.gen_ba1)
        chainer.serializers.load_npz(_checkpoint_dir+"/dis_a.npz", _trainer.updater.disa)
        chainer.serializers.load_npz(_checkpoint_dir+"/dis_b.npz", _trainer.updater.disb)
        return True
    elif os.path.exists(_checkpoint_dir):
        return False
    else:
        os.makedirs(_checkpoint_dir)
        return False

def wave_read(_path_to_file):
    """
    #音声を読み込みます
     Parameters
    ----------
    _path_to_file: str
        ファイルまでのパス
    Returns
    -------
    ans: np.ndarray
        音声
        ValueRange  : [-32767,32767]
        dtype       : int16
    """
    wave_data = wave.open(_path_to_file, "rb")
    ans_data = np.zeros([1], dtype=np.int16)
    dds = wave_data.readframes(1024)
    while dds != b'':
        ans_data = np.append(ans_data, np.frombuffer(dds, "int16"))
        dds = wave_data.readframes(1024)
    wave_data.close()
    ans_data = ans_data[1:]
    return ans_data


def dataset_pre_process(_args):
    """
    データセットが存在し新規に作らない設定ならば読み込み
    そのほかならば作成する。
    Parameters
    ----------
    _args: dict
        設定パラメータ
    Returns
    -------
    _train_iter_a: chainer.iterators.Iterator
        話者Aデータイテレータ
    _train_iter_b: chainer.iterators.Iterator
        話者Bデータイテレータ
    _voice_profile: dict of float
        f0（基本周波数）に関するデータ
    _length_sp: int
        データ長（時間軸方向）
    """
    _sounds_a = None
    _sounds_b = None
    if not (args["use_old_dataset"] and os.path.exists("./dataset/patch/A.npy") and os.path.exists("./dataset/patch/B.npy")):
        _sounds_a, _sounds_b = create_dataset(args["input_size"])
    else:
        # preparing training-data
        print(" [I] loading data-set ...")
        _sounds_a = np.load("./dataset/patch/A.npy")
        _sounds_b = np.load("./dataset/patch/B.npy")
        print(" [I] loaded data-set !")
    _length_sp = _sounds_a.shape[2]
    if args["gpu"] >= 0:
        _sounds_a = chainer.backends.cuda.to_gpu(_sounds_a)
        _sounds_b = chainer.backends.cuda.to_gpu(_sounds_b)
    _train_iter_a = chainer.iterators.MultithreadIterator(Noisy_dataset(_sounds_a, 0.0005), args["batch_size"], shuffle=True)
    _train_iter_b = chainer.iterators.MultithreadIterator(Noisy_dataset(_sounds_b, 0.0005), args["batch_size"], shuffle=True)
    # loading f0 parameters
    _voice_profile = np.load("./voice_profile.npz")
    if not os.path.exists(args["name_save"]):
        os.mkdir(args["name_save"])
    shutil.copy("./voice_profile.npz", args["name_save"]+"/voice_profile.npz")
    return _train_iter_a, _train_iter_b, _voice_profile, _length_sp
def define_model(_args, _train_data_a, _train_data_b):
    """
    学習モデルの定義

    Parameters
    ----------
    _args: dict
        設定パラメーター
    _train_data_a: chainer.iterators.Iterator
        話者Aのイテレーター
    _train_data_b: chainer.iterators.Iterator
        話者Bのイテレーター
    Returns
    -------
    _trainer: chainer.training.trainer
        trainerオブジェクト
    """
    #creating models (if you want to view more code then ./model.py)
    g_a_to_b1 = Generator()
    g_b_to_a1 = Generator()
    d_a = Discriminator()
    d_b = Discriminator()
    if _args["gpu"] >= 0:
        chainer.cuda.Device(_args["gpu"]).use()
        g_a_to_b1.to_gpu()
        g_b_to_a1.to_gpu()
        d_a.to_gpu()
        d_b.to_gpu()
    # Optimizers
    g_optimizer_ab1 = chainer.optimizers.Adam(alpha=2e-4, beta1=0.0, beta2=0.9).setup(g_a_to_b1)
    g_optimizer_ba1 = chainer.optimizers.Adam(alpha=2e-4, beta1=0.0, beta2=0.9).setup(g_b_to_a1)
    d_optimizer_a = chainer.optimizers.Adam(alpha=2e-4, beta1=0.0, beta2=0.9).setup(d_a)
    d_optimizer_b = chainer.optimizers.Adam(alpha=2e-4, beta1=0.0, beta2=0.9).setup(d_b)
    updater = CycleGANUpdater(
        model={"main":g_a_to_b1, "inverse":g_b_to_a1, "disa":d_a, "disb":d_b},
        max_itr=_args["train_iteration"],
        iterator={"main":_train_data_a, "data_b":_train_data_b},
        optimizer={"gen_ab1":g_optimizer_ab1, "gen_ba1":g_optimizer_ba1, "disa":d_optimizer_a, "disb":d_optimizer_b},
        device=_args["gpu"])
    checkpoint_dir = _args["name_save"]
    _trainer = chainer.training.Trainer(updater, (_args["train_iteration"], "iteration"), out=checkpoint_dir)
    # loading net
    load_model_from_npz(checkpoint_dir, _trainer)
    display_interval = (_args["log_interval"], 'iteration')
    if _args["test"]:
        test = wave_read("./dataset/test/test.wav") / 32767.0
        _label_sample = wave_read("./dataset/test/label.wav") / 32767.0
        _trainer.extend(TestModel(_trainer, _args["wave_otp_dir"], test, voice_profile, length_sp, _label_sample), trigger=display_interval)
    # save snapshot
    _trainer.extend(chainer.training.extensions.snapshot(filename='snapshot.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_a_to_b1, 'gen_ab1.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_b_to_a1, 'gen_ba1.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(d_a, 'dis_a.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(d_b, 'dis_b.npz'), trigger=display_interval)
    # learning rate decay
    # decay_timming_seco = (_args["train_iteration"]*0.5, 'iteration')
    # _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.5, optimizer=updater.get_optimizer("gen_ab1")), trigger=decay_timming_seco)
    # _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.5, optimizer=updater.get_optimizer("gen_ba1")), trigger=decay_timming_seco)
    # _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.5, optimizer=updater.get_optimizer("disa")), trigger=decay_timming_seco)
    # _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.5, optimizer=updater.get_optimizer("disb")), trigger=decay_timming_seco)
    # logging
    _trainer.extend(chainer.training.extensions.LogReport(trigger=display_interval))
    # console output
    _trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))
    return _trainer

if __name__ == '__main__':
    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(512*1024*1024)
    args = load_setting_from_json("setting.json")
    if  args["wave_otp_dir"] is not "False":
        args["wave_otp_dir"] = args["wave_otp_dir"] + args["model_name"] +  args["version"]+"/"
        if not os.path.exists(args["wave_otp_dir"]):
            os.makedirs(args["wave_otp_dir"])
    train_iter_a, train_iter_b, voice_profile, length_sp = dataset_pre_process(args)
    trainer = define_model(args, train_iter_a, train_iter_b)
    print(" [I] Train Started")
    # run tarining
    trainer.run()
