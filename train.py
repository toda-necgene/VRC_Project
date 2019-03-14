"""
製作者:TODA


実行すれば学習ができる。
設定はsetting.jsonを利用する。
"""
import os
import time
import glob

import wave
import chainer
import pyworld.pyworld as pw
import numpy as np
import matplotlib.pyplot as plt
from model import Discriminator, Generator, Encoder, Decoder
from updater import CycleGANUpdater
from voice_to_dataset_cycle import create_dataset
from load_setting import load_setting_from_json

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

        # load parameters
        self.args = load_setting_from_json(path_setting)
        # configure dtype
        chainer.global_config.autotune = True
        chainer.cuda.set_max_workspace_size(512*1024*1024)
        if  self.args["wave_otp_dir"] is not "False":
            self.args["wave_otp_dir"] = self.args["wave_otp_dir"] + self.args["model_name"] +  self.args["version"]+"/"
            if not os.path.exists(self.args["wave_otp_dir"]):
                os.makedirs(self.args["wave_otp_dir"])

        sounds_a = None
        sounds_b = None
        # create or load data-set
        if not (self.args["use_old_dataset"] and os.path.exists(self.args["train_data_dir"]+'/A.npy') and self.args["train_data_dir"]+'/B.npy'):
            sounds_a, sounds_b = create_dataset(self.args["input_size"])
        else:
            # preparing training-data
            print(" [I] loading data-set ...")
            sounds_a = np.load(self.args["train_data_dir"]+'/A.npy')
            sounds_b = np.load(self.args["train_data_dir"]+'/B.npy')
            print(" [I] loaded data-set !")
        self.length_sp = sounds_a.shape[2]
        if self.args["gpu"] >= 0:
            sounds_a = chainer.backends.cuda.to_gpu(sounds_a)
            sounds_b = chainer.backends.cuda.to_gpu(sounds_b)
        train_iter_a = chainer.iterators.MultithreadIterator(sounds_a, self.args["batch_size"], shuffle=True)
        train_iter_b = chainer.iterators.MultithreadIterator(sounds_b, self.args["batch_size"], shuffle=True)
        # times on an epoch
        train_data_num = min(sounds_a.shape[0], sounds_b.shape[0])
        self.loop_num = train_data_num // self.args["batch_size"]
        print(" [I] %d data loaded!!" % train_data_num)
        # load test-data
        if self.args["test"]:
            self.test = wave_read(self.args["test_data_dir"]+'/test.wav') / 32767.0
            if self.args["real_sample_compare"]:
                self.label = wave_read(self.args["test_data_dir"] + '/label.wav')
                self.label_power_spec = fft(self.label[800:156000]/32767)
                plt.clf()
                plt.subplot(2, 1, 1)
                insert_image = np.transpose(self.label_power_spec, (1, 0))
                plt.imshow(insert_image, vmin=-0.25, vmax=1, aspect="auto")
                plt.subplot(2, 1, 2)
                plt.plot(self.label[800:156000] / 32767)
                plt.ylim(-1, 1)
                name_save = "%slabel.png" % (self.args["wave_otp_dir"])
                plt.savefig(name_save)
        # loading f0 parameters
        self.voice_profile = np.load("./voice_profile.npy")
        #creating generator (if you want to view more codes then ./model.py)
        self.g_en = Encoder()
        self.g_de = Decoder()
        self.g_a_to_b1 = Generator()
        self.g_b_to_a1 = Generator()
        self.g_a_to_b2 = Generator()
        self.g_b_to_a2 = Generator()
        #creating discriminator (if you want to view more codes then ./model.py)
        self.d_a = Discriminator()
        self.d_b = Discriminator()
        if self.args["gpu"] >= 0:
            chainer.cuda.Device(self.args["gpu"]).use()
            self.g_en.to_gpu()
            self.g_de.to_gpu()
            self.g_a_to_b1.to_gpu()
            self.g_b_to_a1.to_gpu()
            self.g_a_to_b2.to_gpu()
            self.g_b_to_a2.to_gpu()
            self.d_a.to_gpu()
            self.d_b.to_gpu()
        # Optimizers
        def make_optimizer_g(model):
            optimizer = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5)
            optimizer.setup(model)
            return optimizer
        def make_optimizer_d(model):
            optimizer = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5)
            optimizer.setup(model)
            return optimizer
        g_optimizer_en = make_optimizer_g(self.g_en)
        g_optimizer_de = make_optimizer_g(self.g_de)
        g_optimizer_ab1 = make_optimizer_g(self.g_a_to_b1)
        g_optimizer_ba1 = make_optimizer_g(self.g_b_to_a1)
        g_optimizer_ab2 = make_optimizer_g(self.g_a_to_b2)
        g_optimizer_ba2 = make_optimizer_g(self.g_b_to_a2)
        d_optimizer_a = make_optimizer_d(self.d_a)
        d_optimizer_b = make_optimizer_d(self.d_b)
        self.updater = CycleGANUpdater(
            model={"main":self.g_a_to_b1, "inverse":self.g_b_to_a1, "main2":self.g_a_to_b2, "inverse2":self.g_b_to_a2, "encoder":self.g_en, "decoder":self.g_de, "disa":self.d_a, "disb":self.d_b},
            max_itr=self.args["train_iteration"],
            iterator={"main":train_iter_a, "data_b":train_iter_b},
            optimizer={"gen_ab1":g_optimizer_ab1, "gen_ba1":g_optimizer_ba1, "gen_ab2":g_optimizer_ab2, "gen_ba2":g_optimizer_ba2, "gen_en":g_optimizer_en, "gen_de":g_optimizer_de, "disa":d_optimizer_a, "disb":d_optimizer_b},
            device=self.args["gpu"])
    def train(self):
        """
        学習を実行する。

        """
        checkpoint_dir = self.args["name_save"]
        trainer = chainer.training.Trainer(self.updater, (self.args["train_iteration"], "iteration"), out=checkpoint_dir)
        # loading net
        if self.load(checkpoint_dir, trainer):
            print(" [I] Load success.")
        else:
            print(" [I] Load failed.")
        display_interval = (self.args["log_interval"], 'iteration')
        if self.args["test"]:
            trainer.extend(
                TestModel(trainer, self.args["wave_otp_dir"], self.test, self.label_power_spec, self.args["real_sample_compare"], self.voice_profile, self.length_sp),
                trigger=display_interval)
        # save snapshot
        trainer.extend(chainer.training.extensions.snapshot(filename='snapshot.npz'), trigger=display_interval)
        trainer.extend(chainer.training.extensions.snapshot_object(self.g_a_to_b1, 'gen_ab1.npz'), trigger=display_interval)
        trainer.extend(chainer.training.extensions.snapshot_object(self.g_b_to_a1, 'gen_ba1.npz'), trigger=display_interval)
        trainer.extend(chainer.training.extensions.snapshot_object(self.g_a_to_b2, 'gen_ab2.npz'), trigger=display_interval)
        trainer.extend(chainer.training.extensions.snapshot_object(self.g_b_to_a2, 'gen_ba2.npz'), trigger=display_interval)
        trainer.extend(chainer.training.extensions.snapshot_object(self.d_a, 'dis_a.npz'), trigger=display_interval)
        trainer.extend(chainer.training.extensions.snapshot_object(self.d_b, 'dis_b.npz'), trigger=display_interval)
        # learning rate decay
        # decay_timming = chainer.training.triggers.ManualScheduleTrigger([self.args["train_iteration"]*0.5, self.args["train_iteration"]*0.75], 'iteration')
        # trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=self.updater.get_optimizer("gen_ab")), trigger=decay_timming)
        # trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=self.updater.get_optimizer("gen_ba")), trigger=decay_timming)
        # trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=self.updater.get_optimizer("disa")), trigger=decay_timming)
        # trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=self.updater.get_optimizer("disb")), trigger=decay_timming)
        # logging
        trainer.extend(chainer.training.extensions.LogReport(trigger=display_interval))
        # console output
        trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))
        trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'gen_ab1/accuracy']), trigger=display_interval)
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
            _files = list(glob.glob(_checkpoint_dir+"/snapshot*.npz"))
            if len(_files) > 0:
                print(" [I] load file name : %s " % (_files[0]))
                chainer.serializers.load_npz(_files[0], _trainer)
                _ab = list(glob.glob(_checkpoint_dir+"/gen_ab.npz"))[0]
                chainer.serializers.load_npz(_ab, _trainer.updater.gen_ab)
                _ba = list(glob.glob(_checkpoint_dir+"/gen_ba.npz"))[0]
                chainer.serializers.load_npz(_ba, _trainer.updater.gen_ba)
                _da = list(glob.glob(_checkpoint_dir+"/dis_a.npz"))[0]
                chainer.serializers.load_npz(_da, _trainer.updater.disa)
                _db = list(glob.glob(_checkpoint_dir+"/dis_b*.npz"))[0]
                chainer.serializers.load_npz(_db, _trainer.updater.disb)
                return True
            return False
        else:
            os.makedirs(_checkpoint_dir)
            return False


class TestModel(chainer.training.Extension):
    """
    テストを行うExtention
    """
    def __init__(self, trainer, direc, source, label_spec, real_sample_compare, voice_profile, length_sp):
        """
        変数の初期化と事前処理
        """
        self.dir = direc
        self.encoder = trainer.updater.gen_en
        self.decoder = trainer.updater.gen_de
        self.model = trainer.updater.gen_ab1
        self.model2 = trainer.updater.gen_ab2
        source_f0, source_sp, source_ap = encode(source.astype(np.float64))
        self.source_ap = source_ap
        self.length = source_f0.shape[0]
        padding_size = length_sp - source_sp.shape[0] % length_sp
        source_sp = np.pad(source_sp, ((padding_size, 0), (0, 0)), "edge").reshape(-1, length_sp, 513)
        source_sp = np.transpose(source_sp, [0, 2, 1]).astype(np.float32).reshape(-1, 513, length_sp, 1)
        padding_size = length_sp - source_ap.shape[0] % length_sp
        source_ap = np.pad(source_ap, ((padding_size, 0), (0, 0)), "edge").reshape(-1, length_sp, 513)
        source_ap = np.transpose(source_ap, [0, 2, 1]).astype(np.float32).reshape(-1, 513, length_sp, 1)
        source_pp = np.concatenate([source_sp, source_ap], axis=3)
        self.source_pp = chainer.backends.cuda.to_gpu(source_pp)
        self.source_f0 = (source_f0 - voice_profile[0]) * voice_profile[2] + voice_profile[1]
        self.wave_len = source.shape[0]
        self.real_sample_compare = real_sample_compare
        if real_sample_compare:
            seconds = label_spec.shape[0] // (8192 // 512)
            self.label_spec = np.zeros([seconds, 512, 2])
            for h in range(0, seconds):
                term = 8192 // 512
                self.label_spec[h, :, 0] = np.mean(label_spec[h*term:(h+1)*term], axis=0)
                self.label_spec[h, :, 1] = np.std(label_spec[h*term:(h+1)*term], axis=0)
        super(TestModel, self).initialize(trainer)
    def convert(self):
        """
        変換用関数
        """
        #function of test
        #to convert wave file
        chainer.using_config("train", False)
        # run netr
        result = self.encoder(self.source_pp)
        result1 = self.model(result)
        result2 = self.model2(result1)
        result1 = self.decoder(result1)
        result2 = self.decoder(result2)
        result1 = chainer.backends.cuda.to_cpu(result1.data)
        result2 = chainer.backends.cuda.to_cpu(result2.data)
        result1 = np.transpose(result1, [0, 2, 1, 3]).reshape(-1, 513, 2)[-self.length:]
        result2 = np.transpose(result2, [0, 2, 1, 3]).reshape(-1, 513, 2)[-self.length:]
        # post-process
        # WORLD to wave
        result_wave = decode(self.source_f0, result1[:, :, 0], self.source_ap)
        result_wave2 = decode(self.source_f0, result2[:, :, 0], self.source_ap)
        otp = result_wave.reshape(-1)
        otp2 = result_wave2.reshape(-1)
        head_cut_num = otp.shape[0]-self.wave_len
        if head_cut_num > 0:
            otp = otp[head_cut_num:]
            otp2 = otp2[head_cut_num:]
        chainer.using_config("train", True)
        return otp, otp2

    def __call__(self, trainer):
        """
        評価関数
        やっていること
        - パワースペクトラムの比較
        - 音声/画像の保存
        """
        # testing
        out_put1, out_put = self.convert()
        out_put_t = np.append(out_put, out_put1)
        out_puts = (out_put_t*32767).astype(np.int16)
        # calculating power spectrum
        image_power_spec = fft(out_put[800:156000])
        if self.real_sample_compare:
            term = 8192 // 512
            seconds = image_power_spec.shape[0] // (8192 // 512)
            spec_m = np.zeros([seconds, 512, 2])
            for h in range(0, seconds):
                spec_m[h, :, 0] = np.mean(image_power_spec[h*term:(h+1)*term], axis=0)
                spec_m[h, :, 1] = np.std(image_power_spec[h*term:(h+1)*term], axis=0)
            diff = np.sum(spec_m * self.label_spec) / (np.linalg.norm(spec_m) * np.linalg.norm(self.label_spec) + 1e-8)
            score = float(diff)
            chainer.report({"accuracy": score}, self.model)
        plt.clf()
        plt.subplot(2, 1, 1)
        insert_image = np.transpose(image_power_spec, (1, 0))
        plt.imshow(insert_image, vmin=-0.25, vmax=1, aspect="auto")
        plt.subplot(2, 1, 2)
        plt.plot(out_put[800:156000])
        plt.ylim(-1, 1)
        name_save = "%s%04d.png" % (self.dir, trainer.updater.epoch)
        plt.savefig(name_save)
        name_save = "./latest.png"
        plt.savefig(name_save)
        #saving fake waves
        path_save = self.dir + str(trainer.updater.epoch)
        voiced = out_puts.astype(np.int16)
        wave_data = wave.open(path_save + ".wav", 'wb')
        wave_data.setnchannels(1)
        wave_data.setsampwidth(2)
        wave_data.setframerate(16000)
        wave_data.writeframes(voiced.reshape(-1).tobytes())
        wave_data.close()
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
    return _f0, np.clip((np.log(_sp) + 20) / 20, -1.0, 1.0).astype(np.float32), _ap

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
    _sp = np.exp(_sp * 20 - 20).astype(np.float)
    _ap = _ap.astype(np.float)
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
    spec_po = np.clip((spec_po + 15) / 20, -1.0, 1.0)
    return spec_po


if __name__ == '__main__':
    Model("./setting.json").train()
