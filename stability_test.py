"""
製作者:TODA
モデルの収束安定度を確認するためのスクリプト
csv出力対応
"""
import os
import shutil
import wave
import chainer
import numpy as np
from tqdm import trange
from vrc_project.model import Discriminator, Generator
from vrc_project.seq_dataset import SeqData
from vrc_project.updater import CycleGANUpdater
from vrc_project.voice_to_dataset_cycle import create_dataset
from vrc_project.setting_loader import load_setting_from_json
from vrc_project.eval import TestModel
from vrc_project.notify  import send_msg

test_size = 50
g_al_decay = "normal"
d_al_decay = "normal"
g_alpha = 2e-4
d_alpha = 2e-4
g_beta1 = 0.5
d_beta1 = 0.5

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
        _sounds_a, _sounds_b = create_dataset(args["input_size"], delta=args["input_size"])
    else:
        # preparing training-data
        _sounds_a = np.load("./dataset/patch/A.npy")
        _sounds_b = np.load("./dataset/patch/B.npy")
    _length_sp = 200
    if args["gpu"] >= 0:
        _sounds_a = chainer.backends.cuda.to_gpu(_sounds_a)
        _sounds_b = chainer.backends.cuda.to_gpu(_sounds_b)
    _train_iter_a = chainer.iterators.MultithreadIterator(SeqData(_sounds_a, 200), args["batch_size"], shuffle=True, n_threads=2)
    _train_iter_b = chainer.iterators.MultithreadIterator(SeqData(_sounds_b, 200), args["batch_size"], shuffle=True, n_threads=2)
    # f0 parameters(基本周波数F0の変換に使用する定数。詳しくは./vrc_project/voice_to_dataset_cycle.py L65周辺)
    _voice_profile = np.load("./voice_profile.npz")
    if not os.path.exists(args["name_save"]):
        os.mkdir(args["name_save"])
    shutil.copy("./voice_profile.npz", args["name_save"]+"/voice_profile.npz")
    return _train_iter_a, _train_iter_b, _voice_profile, _length_sp
MAX_ITER = 800
_args = dict()
def test_train(step, name):
    """
      短期学習を行う
      Returns
      -----------
      best_score: double
      best_iter: int
    """
    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(512*1024*1024)
    train_iter_a, train_iter_b, voice_profile, length_sp = dataset_pre_process_controler(_args)
    g_a_to_b = Generator()
    g_b_to_a = Generator()
    d_a = Discriminator()
    if _args["gpu"] >= 0:
        chainer.cuda.Device(_args["gpu"]).use()
        g_a_to_b.to_gpu()
        g_b_to_a.to_gpu()
        d_a.to_gpu()
    g_optimizer_ab = chainer.optimizers.Adam(alpha=g_alpha, beta1=g_beta1).setup(g_a_to_b)
    g_optimizer_ba = chainer.optimizers.Adam(alpha=g_alpha, beta1=g_beta1).setup(g_b_to_a)
    d_optimizer_a = chainer.optimizers.Adam(alpha=d_alpha, beta1=d_beta1).setup(d_a)
    # main training
    updater = CycleGANUpdater(
        model={"main":g_a_to_b, "inverse":g_b_to_a, "disa":d_a},
        max_itr=MAX_ITER,
        iterator={"main":train_iter_a, "data_b":train_iter_b},
        optimizer={"gen_ab":g_optimizer_ab, "gen_ba":g_optimizer_ba, "disa":d_optimizer_a},
        device=_args["gpu"])
    term_interval = (100, 'iteration')
    tr = chainer.training.triggers.EarlyStoppingTrigger(monitor="env_test_loss", patients=5, check_trigger=term_interval, max_trigger=(MAX_ITER, "iteration"), mode="min")
    _trainer = chainer.training.Trainer(updater, tr, out=_args["name_save"])
    test = load_wave_file("./dataset/test/test.wav") / 32767.0
    _label_sample = load_wave_file("./dataset/test/label.wav") / 32767.0
    tm = TestModel(_trainer, _args, [test, _label_sample, voice_profile], length_sp, True)
    _trainer.extend(tm, trigger=term_interval)
    _trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10), trigger=(10, "iteration"))
    _log = chainer.training.extensions.LogReport(filename=name+"_test_"+str(step), trigger=term_interval)
    _trainer.extend(_log)
    _trainer.run()
    score_list = list()
    for i in _log.log:
        score_list.append(i["env_test_loss"])
    best = min(score_list)
    index = score_list.index(best)
    return best, index, score_list
if __name__ == '__main__':
    _args = load_setting_from_json("setting.json")
    select = ["1_Gscale"]
    for k in select:
        g_al_decay = "normal"
        d_al_decay = "normal"
        scores = list()
        simple = list()
        for i in trange(test_size, desc="test stage"):
            best_score, best_index, score_ls = test_train(i, str(k))
            scores.append([float(best_score), int(best_index), *score_ls])
            simple.append(float(best_score))
        s = np.asarray(simple)
        print('+'+'-'*10+'+')
        print(simple)
        print("!result_profile!")
        _me = float(np.mean(s))
        _st = float(np.std(s))
        print("score details mean:%f std:%f"%(_me, _st))
        with open(_args["name_save"]+"/test_processs"+str(k)+".csv", "a") as f:
            for ss in scores:
                for sss in ss[:-1]:
                    f.write(str(sss)+",")
                f.write(str(ss[-1])+"\n")
        with open(_args["name_save"]+"/test_result"+str(k)+".csv", "a") as f:
            for i, ss in enumerate(scores):
                f.write(str(ss[0]))
                if i != len(scores)-1:
                    f.write(",")
        with open("line_api_token.txt", "rb") as s:
            key = s.readline().decode("utf8")
        send_msg(key, "%s Finished. mean:%f std:%f"%(str(k), _me, _st))
    print("[*] all_finish")
    