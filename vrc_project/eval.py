"""
製作者:TODA

学習時に逐次行うテストクラスの定義
"""
import wave
import chainer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from vrc_project.world_and_wave import wave2world, world2wave

class TestModel(chainer.training.Extension):
    """
    テストを行うExtention
    """
    def __init__(self, _trainer, _direc, _source, _voice_profile, _sp_input_length, _label_sample):
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
        mpl.rcParams["agg.path.chunksize"] = 100000
        self.dir = _direc
        self.model = _trainer.updater.gen_ab
        self.target = _label_sample
        source_f0, source_sp, source_ap = wave2world(_source.astype(np.float64))
        _, self.source_sp_l, _ = wave2world(_label_sample.astype(np.float64))
        self.image_power_l = fft(_label_sample[800:156000])
        self.source_ap = source_ap
        self.length = source_f0.shape[0]
        padding_size = abs(_sp_input_length - source_sp.shape[0] % _sp_input_length)
        ch = source_sp.shape[1]
        source_sp = np.pad(source_sp, ((padding_size, 0), (0, 0)), "edge").reshape(-1, _sp_input_length, ch)
        source_sp = np.transpose(source_sp, [0, 2, 1]).astype(np.float32).reshape(-1, ch, _sp_input_length, 1)
        source_ap = np.pad(source_ap, ((padding_size, 0), (0, 0)), "edge").reshape(-1, _sp_input_length, 513)
        source_ap = np.transpose(source_ap, [0, 2, 1]).astype(np.float32).reshape(-1, 513, _sp_input_length, 1)
        # source_pp = np.concatenate([source_sp, source_ap], axis=3)
        self.source_pp = chainer.backends.cuda.to_gpu(source_sp)
        self.source_f0 = (source_f0 - _voice_profile["pre_sub"]) * np.sign(source_f0) * _voice_profile["pitch_rate"] + _voice_profile["post_add"] * np.sign(source_f0)
        self.wave_len = _source.shape[0]
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
        result, _ = self.model(self.source_pp)
        result = chainer.backends.cuda.to_cpu(result.data)
        ch = result.shape[1]
        result = np.transpose(result, [0, 2, 1, 3]).reshape(-1, ch)[-self.length:]
        score = np.mean((result - self.source_sp_l)**2)
        result_wave = world2wave(self.source_f0, result, self.source_ap)
        otp = result_wave.reshape(-1)
        head_cut_num = otp.shape[0]-self.wave_len
        if head_cut_num > 0:
            otp = otp[head_cut_num:]
        chainer.using_config("train", True)
        return otp, score, result
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
        out_put, score_raw, im_env = self.convert()
        out_puts = (out_put*32767).astype(np.int16)
        # calculating power spectrum
        image_power_spec = fft(out_put[800:156000])
        score_fft = np.mean((image_power_spec-self.image_power_l) ** 2)
        chainer.report({"env_test_loss": score_raw, "test_loss": score_fft})
        #saving fake power-spec image
        plt.figure(figsize=(8, 5))
        plt.subplot(2, 2, 1)
        _insert_image = np.transpose(image_power_spec, (1, 0))
        plt.imshow(_insert_image, aspect="auto")
        plt.colorbar()
        plt.subplot(2, 2, 2)
        _insert_image = np.transpose(self.image_power_l, (1, 0))
        plt.imshow(_insert_image, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar()
        plt.subplot(2, 2, 3)
        _insert_image = np.transpose((im_env-self.source_sp_l) ** 2, (1, 0))
        pcm = plt.pcolor(_insert_image, norm=mpl.colors.LogNorm(), cmap="jet")
        plt.colorbar(pcm, extend="both")
        plt.subplot(2, 2, 4)
        plt.ylim(-1, 1)
        plt.tick_params(labelbottom=False)
        plt.plot(out_put)
        table = plt.table(cellText=[["iteraiton", "fft_diff", "spenv_diff"], ["%d" % _trainer.updater.iteration, "%f" % score_fft, "%f" %score_raw]])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        plt.savefig("%s%04d.png" % (self.dir, _trainer.updater.iteration))
        plt.savefig("./latest.png")
        #saving fake waves
        path_save = self.dir + str(_trainer.updater.iteration)
        voiced = out_puts.astype(np.int16)
        wave_data = wave.open(path_save + ".wav", 'wb')
        wave_data.setnchannels(1)
        wave_data.setsampwidth(2)
        wave_data.setframerate(16000)
        wave_data.writeframes(voiced.reshape(-1).tobytes())
        wave_data.close()
        wave_data = wave.open("latest.wav", 'wb')
        wave_data.setnchannels(1)
        wave_data.setsampwidth(2)
        wave_data.setframerate(16000)
        wave_data.writeframes(voiced.reshape(-1).tobytes())
        wave_data.close()
        plt.clf()
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
    spec_po = np.log(np.power(spec_re, 2) + np.power(spec_im, 2) + 1e-8).reshape(time_ruler, -1)[:, 512:]
    spec_po = np.clip((spec_po + 5) / 10, -1.0, 1.0)
    return spec_po
