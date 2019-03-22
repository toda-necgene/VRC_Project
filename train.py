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
import matplotlib.pyplot as plt
from vrc_project.model import Discriminator, Generator, Encoder, Decoder
from vrc_project.updater import CycleGANUpdater
from vrc_project.voice_to_dataset_cycle import create_dataset
from vrc_project.setting_loader import load_setting_from_json
from vrc_project.world_and_wave import wave2world, world2wave


class TestModel(chainer.training.Extension):
    """
    テストを行うExtention
    """
    def __init__(self, _trainer, _direc, _source, _label_spec, _voice_profile, _sp_input_length):
        """
        変数の初期化と事前処理
        Parameters
        ----------
        _trainer: chainer.training.trainer
            評価用トレーナ
        _drec: str
            ファイル出力ディレクトリパス
        _source: np.ndarray
            変換元音声
            range: [-1.0, 1.0]
            dtype: float
        _label_spec: np.ndarray
            目標話者パラレルテストデータのパワースペクトラム
            Noneならば比較を省略
        _voice_profile: dict of int
            f0に関するデータ
        _sp_input_length: int
            モデルの入力データ長
        """
        self.dir = _direc
        self.encoder = _trainer.updater.gen_en
        self.decoder = _trainer.updater.gen_de
        self.model = _trainer.updater.gen_ab1
        self.model2 = _trainer.updater.gen_ab2
        source_f0, source_sp, source_ap = wave2world(_source.astype(np.float64))
        self.source_ap = source_ap
        self.length = source_f0.shape[0]
        padding_size = _sp_input_length - source_sp.shape[0] % _sp_input_length
        source_sp = np.pad(source_sp, ((padding_size, 0), (0, 0)), "edge").reshape(-1, _sp_input_length, 513)
        source_sp = np.transpose(source_sp, [0, 2, 1]).astype(np.float32).reshape(-1, 513, _sp_input_length, 1)
        padding_size = _sp_input_length - source_ap.shape[0] % _sp_input_length
        source_ap = np.pad(source_ap, ((padding_size, 0), (0, 0)), "edge").reshape(-1, _sp_input_length, 513)
        source_ap = np.transpose(source_ap, [0, 2, 1]).astype(np.float32).reshape(-1, 513, _sp_input_length, 1)
        source_pp = np.concatenate([source_sp, source_ap], axis=3)
        self.source_pp = chainer.backends.cuda.to_gpu(source_pp)
        self.source_f0 = (source_f0 - _voice_profile["pre_sub"]) * _voice_profile["pitch_rate"] + _voice_profile["post_add"]
        self.wave_len = _source.shape[0]
        self.real_sample_compare = _label_spec is not None
        if _label_spec is not None:
            seconds = _label_spec.shape[0] // (8192 // 512)
            self.label_spec = np.zeros([seconds, 512, 2])
            for h in range(0, seconds):
                term = 8192 // 512
                self.label_spec[h, :, 0] = np.mean(_label_spec[h*term:(h+1)*term], axis=0)
                self.label_spec[h, :, 1] = np.std(_label_spec[h*term:(h+1)*term], axis=0)
        super(TestModel, self).initialize(_trainer)
    def convert(self):
        """
        変換用関数
        Returns
        -------
        otp: np.ndarray
            変換後の音声波形データ
        """
        chainer.using_config("train", False)
        result = self.encoder(self.source_pp)
        result = self.model(result)
        result = self.model2(result)
        result = self.decoder(result)
        result = chainer.backends.cuda.to_cpu(result.data)
        result = np.transpose(result, [0, 2, 1, 3]).reshape(-1, 513, 2)[-self.length:]
        result_wave = world2wave(self.source_f0, result[:, :, 0], self.source_ap)
        otp = result_wave.reshape(-1)
        head_cut_num = otp.shape[0]-self.wave_len
        if head_cut_num > 0:
            otp = otp[head_cut_num:]
        chainer.using_config("train", True)
        return otp
    def __call__(self, _trainer):
        """
        評価関数
        やっていること
        - パワースペクトラムの比較
        - 音声/画像の保存
        Parameters
        ----------
        _trainer: chainer.training.trainer
            テストに使用するトレーナー
        """
        # testing
        out_put = self.convert()
        out_puts = (out_put*32767).astype(np.int16)
        # calculating power spectrum
        image_power_spec = fft(out_put[800:156000])
        if self.real_sample_compare:
            term = 8192 // 512
            seconds = image_power_spec.shape[0] // term
            spec_m = np.zeros([seconds, 512, 2])
            for h in range(0, seconds):
                spec_m[h, :, 0] = np.mean(image_power_spec[h*term:(h+1)*term], axis=0)
                spec_m[h, :, 1] = np.std(image_power_spec[h*term:(h+1)*term], axis=0)
            diff = np.sum(spec_m * self.label_spec) / (np.linalg.norm(spec_m) * np.linalg.norm(self.label_spec) + 1e-8)
            score = float(diff)
            chainer.report({"accuracy": score}, self.model)
        #saving fake power-spec image
        plt.clf()
        plt.subplot(2, 1, 1)
        _insert_image = np.transpose(image_power_spec, (1, 0))
        plt.imshow(_insert_image, vmin=-1, vmax=1, aspect="auto")
        plt.subplot(2, 1, 2)
        plt.plot(out_put[800:156000])
        plt.ylim(-1, 1)
        _name_save = "%s%04d.png" % (self.dir, _trainer.updater.epoch)
        plt.savefig(_name_save)
        _name_save = "./latest.png"
        plt.savefig(_name_save)
        #saving fake waves
        path_save = self.dir + str(_trainer.updater.epoch)
        voiced = out_puts.astype(np.int16)
        wave_data = wave.open(path_save + ".wav", 'wb')
        wave_data.setnchannels(1)
        wave_data.setsampwidth(2)
        wave_data.setframerate(16000)
        wave_data.writeframes(voiced.reshape(-1).tobytes())
        wave_data.close()

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
        chainer.serializers.load_npz(_checkpoint_dir+"/gen_en.npz", _trainer.updater.gen_en)
        chainer.serializers.load_npz(_checkpoint_dir+"/gen_de.npz", _trainer.updater.gen_de)
        chainer.serializers.load_npz(_checkpoint_dir+"/gen_ab1.npz", _trainer.updater.gen_ab1)
        chainer.serializers.load_npz(_checkpoint_dir+"/gen_ba1.npz", _trainer.updater.gen_ba1)
        chainer.serializers.load_npz(_checkpoint_dir+"/gen_ab2.npz", _trainer.updater.gen_ab2)
        chainer.serializers.load_npz(_checkpoint_dir+"/gen_ba2.npz", _trainer.updater.gen_ba2)
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

def fft(_data):
    """
    stftを計算

     Parameters
    ----------
    _data: np.ndarray
        音声データ
        range  : [-1.0,1.0]
        dtype  : float64
    Returns
    -------
    spec_po: np.ndarray
        パワースペクトラム
        power-spectram
        Shape : (n,512)
    """
    time_ruler = _data.shape[0] // 512
    if _data.shape[0] % 512 == 0:
        time_ruler -= 1
    window = np.hamming(1024)
    pos = 0
    wined = np.zeros([time_ruler, 1024])
    for fft_index in range(time_ruler):
        frame = _data[pos:pos + 1024]
        padding_size = 1024-frame.shape[0]
        if padding_size > 0:
            frame = np.pad(frame, (0, padding_size), "constant")
        wined[fft_index] = frame * window
        pos += 512
    fft_r = np.fft.fft(wined, n=1024, axis=1)
    spec_re = fft_r.real.reshape(time_ruler, -1)
    spec_im = fft_r.imag.reshape(time_ruler, -1)
    spec_po = np.log(np.power(spec_re, 2) + np.power(spec_im, 2) + 1e-24).reshape(time_ruler, -1)[:, 512:]
    spec_po = np.clip((spec_po) / 20, -1.0, 1.0)
    return spec_po

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
    _train_iter_a = chainer.iterators.MultithreadIterator(_sounds_a, args["batch_size"], shuffle=True)
    _train_iter_b = chainer.iterators.MultithreadIterator(_sounds_b, args["batch_size"], shuffle=True)
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
    g_en = Encoder()
    g_de = Decoder()
    g_a_to_b1 = Generator()
    g_b_to_a1 = Generator()
    g_a_to_b2 = Generator()
    g_b_to_a2 = Generator()
    d_a = Discriminator()
    d_b = Discriminator()
    if _args["gpu"] >= 0:
        chainer.cuda.Device(_args["gpu"]).use()
        g_en.to_gpu()
        g_de.to_gpu()
        g_a_to_b1.to_gpu()
        g_b_to_a1.to_gpu()
        g_a_to_b2.to_gpu()
        g_b_to_a2.to_gpu()
        d_a.to_gpu()
        d_b.to_gpu()
    # Optimizers
    g_optimizer_en = chainer.optimizers.Adam(alpha=2e-4, beta1=0.9).setup(g_en)
    g_optimizer_de = chainer.optimizers.Adam(alpha=2e-4, beta1=0.9).setup(g_de)
    g_optimizer_ab1 = chainer.optimizers.Adam(alpha=2e-4, beta1=0.9).setup(g_a_to_b1)
    g_optimizer_ba1 = chainer.optimizers.Adam(alpha=2e-4, beta1=0.9).setup(g_b_to_a1)
    g_optimizer_ab2 = chainer.optimizers.Adam(alpha=2e-4, beta1=0.9).setup(g_a_to_b2)
    g_optimizer_ba2 = chainer.optimizers.Adam(alpha=2e-4, beta1=0.9).setup(g_b_to_a2)
    d_optimizer_a = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5).setup(d_a)
    d_optimizer_b = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5).setup(d_b)
    updater = CycleGANUpdater(
        model={"main":g_a_to_b1, "inverse":g_b_to_a1, "main2":g_a_to_b2, "inverse2":g_b_to_a2, "encoder":g_en, "decoder":g_de, "disa":d_a, "disb":d_b},
        max_itr=_args["train_iteration"],
        iterator={"main":_train_data_a, "data_b":_train_data_b},
        optimizer={"gen_ab1":g_optimizer_ab1, "gen_ba1":g_optimizer_ba1, "gen_ab2":g_optimizer_ab2, "gen_ba2":g_optimizer_ba2, "gen_en":g_optimizer_en, "gen_de":g_optimizer_de, "disa":d_optimizer_a, "disb":d_optimizer_b},
        device=_args["gpu"])
    checkpoint_dir = _args["name_save"]
    _trainer = chainer.training.Trainer(updater, (_args["train_iteration"], "iteration"), out=checkpoint_dir)
    # loading net
    load_model_from_npz(checkpoint_dir, _trainer)
    display_interval = (_args["log_interval"], 'iteration')
    if _args["test"]:
        label_power_spec = None
        test = wave_read("./dataset/test/test.wav") / 32767.0
        if _args["real_sample_compare"]:
            label = wave_read("./dataset/test/label.wav")
            label_power_spec = fft(label[800:156000]/32767)
            plt.clf()
            plt.subplot(2, 1, 1)
            insert_image = np.transpose(label_power_spec, (1, 0))
            plt.imshow(insert_image, vmin=-0.25, vmax=1, aspect="auto")
            plt.subplot(2, 1, 2)
            plt.plot(label[800:156000] / 32767)
            plt.ylim(-1, 1)
            name_save = "%slabel.png" % (_args["wave_otp_dir"])
            plt.savefig(name_save)
        _trainer.extend(TestModel(_trainer, _args["wave_otp_dir"], test, label_power_spec, voice_profile, length_sp), trigger=display_interval)
    # save snapshot
    _trainer.extend(chainer.training.extensions.snapshot(filename='snapshot.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_en, 'gen_en.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_de, 'gen_de.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_a_to_b1, 'gen_ab1.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_b_to_a1, 'gen_ba1.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_a_to_b2, 'gen_ab2.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(g_b_to_a2, 'gen_ba2.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(d_a, 'dis_a.npz'), trigger=display_interval)
    _trainer.extend(chainer.training.extensions.snapshot_object(d_b, 'dis_b.npz'), trigger=display_interval)
    # learning rate decay
    decay_timming_prim = chainer.training.triggers.ManualScheduleTrigger([_args["train_iteration"]*0.5, _args["train_iteration"]*0.75, _args["train_iteration"]*0.9], 'iteration')
    decay_timming_seco = chainer.training.triggers.ManualScheduleTrigger([_args["train_iteration"]*0.75, _args["train_iteration"]*0.9], 'iteration')
    _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=updater.get_optimizer("gen_en")), trigger=decay_timming_prim)
    _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=updater.get_optimizer("gen_de")), trigger=decay_timming_prim)
    _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=updater.get_optimizer("gen_ab1")), trigger=decay_timming_prim)
    _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=updater.get_optimizer("gen_ba1")), trigger=decay_timming_prim)
    _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=updater.get_optimizer("gen_ab2")), trigger=decay_timming_seco)
    _trainer.extend(chainer.training.extensions.ExponentialShift('alpha', 0.1, optimizer=updater.get_optimizer("gen_ba2")), trigger=decay_timming_seco)
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
