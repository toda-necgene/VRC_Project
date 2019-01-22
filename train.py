"""
製作者:TODA


実行すれば学習ができる。
設定はsetting.jsonを利用する。
"""
import os
import time
import json
from datetime import datetime

import wave
import pyaudio
import pyworld.pyworld as pw
import numpy as np
import chainer
import matplotlib.pyplot as plt
from model import Discriminator, Generator
from updater import CycleGANUpdater
from voice_to_dataset_cycle import create_dataset


class Model:
    """
    学習用モデルの定義です。
    """
    def __init__(self, path_setting):
        self.start_time = time.time()
        self.tt_list = list()
        self.test = None
        self.label = None
        self.label_spec = None
        self.label_spec_v = None
        self.sounds_r = None
        self.sounds_t = None
        self.loop_num = 0

        # setting default parameters

        self.args = dict()

        # name options
        self.args["model_name"] = "VRC"
        self.args["version"] = "1.0.0"
        # saving options
        self.args["checkpoint_dir"] = "./trained_models"
        self.args["wave_otp_dir"] = "./harvests"
        #training-data options
        self.args["use_old_dataset"] = False
        self.args["train_data_dir"] = "./dataset/train"
        self.args["test_data_dir"] = "./dataset/test"
        # learning details output options
        self.args["test"] = True
        self.args["real_sample_compare"] = False
        # learning options
        self.args["batch_size"] = 128
        self.args["weight_cycle"] = 10.0
        self.args["train_iteration"] = 600000
        self.args["start_epoch"] = 0
        self.args["learning_rate"] = 8e-7
        self.args["test_interval"] = 1
        # architecture option
        self.args["input_size"] = 4096
        self.args["gpu"] = -1


        # loading json setting file
        # (more codes ./setting.json. manual is exist in ./setting-example.json)
        with open(path_setting, "r") as setting_raw_txt:
            try:
                json_loaded = json.load(setting_raw_txt)
                keys = json_loaded.keys()
                for j in keys:
                    data = json_loaded[j]
                    keys2 = data.keys()
                    for k in keys2:
                        if k in self.args:
                            if isinstance(self.args[k], type(data[k])):
                                self.args[k] = data[k]
                            else:
                                print(" [W] Argumet \"" + k + "\" is incorrect data type. Please change to \"" + str(type(self.args[k])) + "\"")
                        elif k[0] == "#":
                            pass
                        else:
                            print(" [W] Argument \"" + k + "\" is not exsits.")
            except json.JSONDecodeError as er_message:
                print(" [W] JSONDecodeError: ", er_message)
                print(" [W] Use default setting")
        # shapes properties
        self.input_size_model = [self.args["batch_size"], 52, 513]
        self.input_size_test = [1, 52, 513]

        # initializing harvest directory
        self.args["name_save"] = self.args["model_name"] + self.args["version"]
        if  self.args["wave_otp_dir"] is not "False":
            self.args["wave_otp_dir"] = self.args["wave_otp_dir"] + self.args["name_save"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])

        sounds_a = None
        sounds_b = None
        # create or load data-set
        if not self.args["use_old_dataset"]:
            sounds_a, sounds_b = create_dataset(self.args["input_size"])
        else:
            # preparing training-data
            batch_files = self.args["train_data_dir"]+'/A.npy'
            batch_files2 = self.args["train_data_dir"]+'/B.npy'
            print(" [I] loading data-set ...")
            sounds_a = np.load(batch_files)
            sounds_b = np.load(batch_files2)
            if self.args["gpu"] >= 0:
                sounds_a = chainer.backends.cuda.to_gpu(sounds_a)
                sounds_b = chainer.backends.cuda.to_gpu(sounds_b)
            print(" [I] loaded data-set !")
        train_iter_a = chainer.iterators.SerialIterator(sounds_a, self.args["batch_size"])
        train_iter_b = chainer.iterators.SerialIterator(sounds_b, self.args["batch_size"])
        # times on an epoch
        train_data_num = min(sounds_a.shape[0], sounds_b.shape[0])
        self.loop_num = train_data_num // self.args["batch_size"]
        print(" [I] %d data loaded!!" % train_data_num)

        # load test-data
        if self.args["test"]:
            self.test = wave_read(self.args["test_data_dir"]+'/test.wav')
            if self.args["real_sample_compare"]:
                self.label = wave_read(self.args["test_data_dir"] + '/label.wav')
                self.label_power_spec = fft(self.label[800:156000]/32767)
        # loading f0 parameters
        self.voice_profile = np.load("./voice_profile.npy")
        #creating generator (if you want to view more codes then ./model.py)
        self.g_a_to_b = Generator()
        self.g_b_to_a = Generator()
        #creating discriminator (if you want to view more codes then ./model.py)
        self.d_a_and_b = Discriminator()
        # Optimizers
        def make_optimizer(model, alpha=0.0002, beta1=0.5):
            optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
            optimizer.setup(model)
            optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')
            return optimizer
        g_optimizer_ab = make_optimizer(self.g_a_to_b, self.args["learning_rate"])
        g_optimizer_ba = make_optimizer(self.g_b_to_a, self.args["learning_rate"])
        d_optimizer = make_optimizer(self.d_a_and_b, self.args["learning_rate"])
        self.updater = CycleGANUpdater(
            _models=(self.g_a_to_b, self.g_b_to_a, self.d_a_and_b),
            iterator={"main":train_iter_a, "data_b":train_iter_b},
            optimizer={"gen_ab":g_optimizer_ab, "gen_ba":g_optimizer_ba, "dis":d_optimizer},
            device=self.args["gpu"])
    def train(self):
        """
        学習を実行する。

        """
        model_dir = self.args["name_save"]
        checkpoint_dir = os.path.join(self.args["checkpoint_dir"], model_dir)
        trainer = chainer.training.Trainer(self.updater, (self.args["train_iteration"], "iteration"), out=checkpoint_dir)
        # loading net
        if self.load(checkpoint_dir, trainer):
            print(" [I] Load success.")
        else:
            print(" [I] Load failed.")
        snapshot_interval = (self.args["test_interval"], 'epoch')
        display_interval = (self.args["test_interval"], 'epoch')
        trainer.extend(chainer.training.extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
        trainer.extend(chainer.training.extensions.snapshot_object(self.g_a_to_b, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
        trainer.extend(chainer.training.extensions.LogReport(trigger=display_interval))
        trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'iteration', 'gen_ab/loss', 'gen_ba/loss', 'dis/loss', "score_pow_spec"]), trigger=display_interval)
        trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))
        if self.args["test"]:
            trainer.extend(
                TestModel(trainer, self.args["wave_otp_dir"], self.g_a_to_b, self.test,
                          self.label_power_spec, self.args["real_sample_compare"], self.voice_profile),
                trigger=snapshot_interval)
        # run tarining
        print(" [I] Train Started")
        trainer.run()
    def load(self, _checkpoint_dir, _trainer):
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
        if os.path.exists(_checkpoint_dir):
            chainer.serializers.load_npz(_checkpoint_dir, _trainer)
            return True
        else:
            os.makedirs(_checkpoint_dir)
            return False


class TestModel(chainer.training.Extension):
    """
    テストを行うExtention
    """
    def __init__(self, trainer, direc, model, source, label_spec, real_sample_compare, voice_profile):
        """
        変数の初期化と事前処理
        """
        self.dir = direc
        self.source_f0, self.source_sp, self.source_ap = encode(source.astype(np.float64))
        self.source_f0 = (self.source_f0-voice_profile[0])*voice_profile[2]+voice_profile[1]
        self.source_sp = np.transpose(self.source_sp, [1, 0]).astype(np.float32)
        padding_size = 52 - self.source_sp.shape[1] % 52
        self.source_sp = np.pad(self.source_sp, ((0, 0), (padding_size, 0)), "constant")
        self.source_sp = self.source_sp.reshape(-1, 513, 52)
        self.source_sp = chainer.backends.cuda.to_gpu(self.source_sp)
        self.model = model
        self.length = self.source_f0.shape[0]
        self.wave_len = source.shape[0]
        self.real_sample_compare = real_sample_compare
        self.label_spec = np.mean(label_spec, axis=0)
        self.label_spec_v = np.std(label_spec, axis=0)
        super(TestModel, self).initialize(trainer)
    def convert(self):
        """
        変換用関数
        """
        #function of test
        #to convert wave file
        chainer.config.train = False
        # run net
        result = self.model(self.source_sp)
        result = chainer.backends.cuda.to_cpu(result)
        # post-process
        # WORLD to wave
        result_wave = decode(self.source_f0, result.reshape(-1, 513)[:self.length], self.source_ap)
        result_wave_fixed = np.clip(result_wave, -1.0, 1.0)
        otp = result_wave_fixed.reshape(-1).astype(np.int16)
        head_cut_num = otp.shape[0]-self.wave_len
        if head_cut_num > 0:
            otp = otp[head_cut_num:]
        chainer.config.train = True
        return otp

    def __call__(self, trainer):
        """
        呼び出し関数
        やっていること
        - パワースペクトラムの比較
        - 音声/画像の保存
        """
        # testing
        out_put = self.convert()

        # fixing harvests types
        out_puts = (out_put.copy() * 32767).astype(np.int16)

        # calculating power spectrum
        image_power_spec = fft(out_put[800:156000])
        if self.real_sample_compare:
            spec_m = np.mean(image_power_spec, axis=0)
            spec_v = np.std(image_power_spec, axis=0)
            diff = spec_m-self.label_spec
            diff2 = spec_v-self.label_spec_v
            score = np.mean(diff*diff+diff2*diff2)
            chainer.report({"score_pow_spec":score})
        # saving fake spectrum
        plt.clf()
        plt.subplot(3, 1, 1)
        insert_image = np.transpose(image_power_spec, (1, 0))
        plt.imshow(insert_image, vmin=-15, vmax=5, aspect="auto")
        plt.subplot(3, 1, 2)
        plt.plot(out_put[800:156000])
        plt.ylim(-1, 1)
        if self.real_sample_compare:
            plt.subplot(3, 1, 3)
            plt.plot(diff * diff + diff2 * diff2)
            plt.ylim(0, 100)
        name_save = "%s%04d.png" % (self.dir, trainer.epoch)
        plt.savefig(name_save)
        name_save = "./latest.png"
        plt.savefig(name_save)
        #saving fake waves
        pa_ins = pyaudio.PyAudio()
        path_save = self.dir + datetime.now().strftime("%m-%d_%H-%M-%S") + "_" + str(trainer.epoch)
        voiced = out_puts.astype(np.int16)[800:156000]
        wave_data = wave.open(path_save + ".wav", 'wb')
        wave_data.setnchannels(1)
        wave_data.setsampwidth(pa_ins.get_sample_size(pyaudio.paInt16))
        wave_data.setframerate(16000)
        wave_data.writeframes(voiced.reshape(-1).tobytes())
        wave_data.close()
        pa_ins.terminate()
def encode(data):
    """
    #音声をWorldに変換します

    Parameters
    ----------
    data : ndarray
        入力データ
        SamplingRate: 16000
        ValueRange  : [-1.0,1.0]
        dtype       : float64
    Returns
    -------
    World: list(3items)
        出力
        _f0 : f0 estimation
        Shape(N)
        dtype       : float64
        _sp : spectram envelobe
        Shape(N,513)
        dtype       : float64
        _ap : aperiodicity
        Shape(N,513)
        dtype       : float64
    """
    sampleing_rate = 16000
    _f0, _t = pw.dio(data, sampleing_rate)
    _f0 = pw.stonemask(data, _f0, _t, sampleing_rate)
    _sp = pw.cheaptrick(data, _f0, _t, sampleing_rate)
    _ap = pw.d4c(data, _f0, _t, sampleing_rate)
    return _f0.astype(np.float64), np.clip((np.log(_sp)+15)/20, -1.0, 1.0).astype(np.float64), _ap.astype(np.float64)

def decode(_f0, _sp, _ap):
    """
    #Worldを音声に変換します
    Parameters
    ----------
    _f0 : np.ndarray
        _f0 estimation
        Shape(N)
        dtype       : float64
    _sp : np.ndarray
        spectram envelobe
        Shape(N,513)
        dtype       : float64
    _ap : np.ndarray
        aperiodicity
        Shape(N,513)
        dtype       : float64
    Returns
    -------
    World: list(3items)
        #出力
        SamplingRate: 16000
        ValueRange  : [-1.0,1.0]
        dtype       : float64
    """
    _sp = np.exp(_sp * 20 - 15).astype(np.float)
    return pw.synthesize(_f0, _sp, _ap, 16000)


def wave_read(path_to_file):
    """
    #音声を読み込みます
     Parameters
    ----------
    path_to_file : string
        #ファイルまでのパス
    Returns
    -------
    ans: ndarray
        #音声
        ValueRange  : [-32767,32767]
        dtype       : int16
    """
    wave_data = wave.open(path_to_file, "rb")
    ans_data = np.zeros([1], dtype=np.int16)
    dds = wave_data.readframes(1024)
    while dds != b'':
        ans_data = np.append(ans_data, np.frombuffer(dds, "int16"))
        dds = wave_data.readframes(1024)
    wave_data.close()
    ans_data = ans_data[1:]
    return ans_data

def fft(data):
    """
    # stftを計算

     Parameters
    ----------
    data : np.ndarray
        音声データ
        ValueRange  : [-1.0,1.0]
        dtype       : float64
    Returns
    -------
    spec_po: np.ndarray
        パワースペクトラム
        power-spectram
        Shape : (n,512)
    """
    time_ruler = data.shape[0] // 512
    if data.shape[0] % 512 == 0:
        time_ruler -= 1
    window = np.hamming(1024)
    pos = 0
    wined = np.zeros([time_ruler, 1024])
    for fft_index in range(time_ruler):
        frame = data[pos:pos + 1024]
        padding_size = 1024-frame.shape[0]
        if padding_size > 0:
            frame = np.pad(frame, (0, padding_size), "constant")
        wined[fft_index] = frame * window
        pos += 512
    fft_r = np.fft.fft(wined, n=1024, axis=1)
    spec_re = fft_r.real.reshape(time_ruler, -1)
    spec_im = fft_r.imag.reshape(time_ruler, -1)
    spec_po = np.log(np.power(spec_re, 2) + np.power(spec_im, 2) + 1e-24).reshape(time_ruler, -1)[:, 512:]
    return np.clip(spec_po, -15.0, 5.0)


if __name__ == '__main__':
    NET = Model("./setting.json")
    NET.train()
