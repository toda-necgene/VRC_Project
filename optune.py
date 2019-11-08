"""
製作者:TODA
optunaを用いたハイパーパラメータの探索
"""
import os
import shutil
import wave
import chainer
import numpy as np
import optuna
from optuna.integration.chainer import ChainerPruningExtension
from vrc_project.model import Discriminator, Generator
from vrc_project.seq_dataset import SeqData
from vrc_project.updater import CycleGANUpdater
from vrc_project.voice_to_dataset_cycle import create_dataset
from vrc_project.setting_loader import load_setting_from_json
from vrc_project.eval import TestModel
from vrc_project.notify  import send_msg


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
        _sounds_a = np.load("./dataset/patch/A.npy")
        _sounds_b = np.load("./dataset/patch/B.npy")
    _length_sp = 200
    if args["gpu"] >= 0:
        _sounds_a = chainer.backends.cuda.to_gpu(_sounds_a)
        _sounds_b = chainer.backends.cuda.to_gpu(_sounds_b)
    _train_iter_a = chainer.iterators.MultithreadIterator(SeqData(_sounds_a, 200), args["batch_size"], shuffle=True, n_threads=2)
    _train_iter_b = chainer.iterators.MultithreadIterator(SeqData(_sounds_b, 200), args["batch_size"], shuffle=True, n_threads=2)
    _voice_profile = np.load("./voice_profile.npz")
    if not os.path.exists(args["name_save"]):
        os.mkdir(args["name_save"])
    shutil.copy("./voice_profile.npz", args["name_save"]+"/voice_profile.npz")
    return _train_iter_a, _train_iter_b, _voice_profile, _length_sp
MAX_ITER = 2000
_args = dict()
def objective(trials):
    """
    対象関数
    設定したパラメータを用いて実験する
    """
    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(512*1024*1024)
    train_iter_a, train_iter_b, voice_profile, length_sp = dataset_pre_process_controler(_args)
    g_la = 9
    g_al_decay = 1.0
    d_al_decay = 1.0
    cyc_lambda = trials.suggest_int("cyc_lambda", 10, 500)
    g_ch = 256
    d_ch = [64, 128, 256, 512]
    g_a_to_b = Generator(chs=g_ch, layers=g_la)
    g_b_to_a = Generator(chs=g_ch, layers=g_la)
    d_a = Discriminator(chs=d_ch)
    if _args["gpu"] >= 0:
        import cupy as cp
        cp.seed = (100)
        chainer.using_config('cudnn_deterministic', True)
        chainer.cuda.Device(_args["gpu"]).use()
        g_a_to_b.to_gpu()
        g_b_to_a.to_gpu()
        d_a.to_gpu()
    else:
        np.seed(100)
    g_optimizer_ab = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5).setup(g_a_to_b)
    g_optimizer_ba = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5).setup(g_b_to_a)
    d_optimizer_a = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5).setup(d_a)
    # main training
    updater = CycleGANUpdater(
        model={"main":g_a_to_b, "inverse":g_b_to_a, "disa":d_a},
        max_itr=MAX_ITER,
        cyc_lambda=cyc_lambda,
        iterator={"main":train_iter_a, "data_b":train_iter_b},
        optimizer={"gen_ab":g_optimizer_ab, "gen_ba":g_optimizer_ba, "disa":d_optimizer_a},
        device=_args["gpu"])
    term_interval = (100, 'iteration')
    tr = chainer.training.triggers.EarlyStoppingTrigger(monitor="env_test_loss", patients=5, check_trigger=term_interval, max_trigger=(MAX_ITER, "iteration"), mode="min")
    _trainer = chainer.training.Trainer(updater, tr, out=_args["name_save"])
    test = load_wave_file("./dataset/test/test.wav") / 32767.0
    _label_sample = load_wave_file("./dataset/test/label.wav") / 32767.0
    tm = TestModel(_trainer, _args, [test, _label_sample, voice_profile], length_sp, None)
    _trainer.extend(tm, trigger=term_interval)
    _log = chainer.training.extensions.LogReport(trigger=term_interval)
    _trainer.extend(_log)
    _trainer.extend(chainer.training.extensions.PrintReport(["env_test_loss"]), trigger=term_interval)
    decay_timming = (400, "iteration")
    if g_al_decay != 1.0:
        _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', g_al_decay, optimizer=updater.get_optimizer("gen_ab")), trigger=decay_timming)
        _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', g_al_decay, optimizer=updater.get_optimizer("gen_ba")), trigger=decay_timming)
    if d_al_decay != 1.0:
        _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', d_al_decay, optimizer=updater.get_optimizer("disa")), trigger=decay_timming)
    _trainer.extend(ChainerPruningExtension(trials, 'env_test_loss', term_interval))
    _trainer.run()
    score_list = list()
    for i in _log.log:
        score_list.append(i["env_test_loss"])
    score = tm.convert()[1]
    best = min(score_list)
    index = score_list.index(best)*100+100
    trials.set_user_attr("score", best)
    trials.set_user_attr("iter", index)
    return score
if __name__ == '__main__':
    _args = load_setting_from_json("setting.json")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=5))
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print('+'+'-'*10+'+')
    print("!result_profile!")
    print('+'+'-'*10+'+')
    print('Value: ', trial.value)
    print('-'*10)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    print('-'*10)
    print('  score detail:')
    l = list()
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))
        l.append(value)
    best_score = min(l)
    print('-'*10)
    with open("line_api_token.txt", "rb") as s:
        key = s.readline().decode("utf8")
    send_msg(key, "Finished. score:%d" % best_score)
    study.trials_dataframe().to_csv(_args["name_save"]+"/study_result.csv")
    print("[*] all_finish")
    