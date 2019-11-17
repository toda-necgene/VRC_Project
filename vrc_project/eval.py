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
    def __init__(self, _trainer, _args, data, _sp_input_length, only_score):
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
        only_score: str
            画像・音声を出力するか否か
        """
        self.model_name = _args["version"]
        mpl.rcParams["agg.path.chunksize"] = 100000
        self.dir = _args["wave_otp_dir"]
        self.model_en = _trainer.updater.gen_ab
        self.model_f0 = _trainer.updater.disa
        self.target = data[1]
        source_f0, source_sp, source_ap = wave2world(data[0].astype(np.float64))
        self.l_f0, self.source_sp_l, _ = wave2world(data[1].astype(np.float64))
        self.image_power_l = fft(data[1])
        self.source_ap = source_ap
        self.length = source_f0.shape[0]
        padding_size = abs(_sp_input_length - source_sp.shape[0] % _sp_input_length)
        ch = source_sp.shape[1]
        source_sp = np.pad(source_sp, ((padding_size, 0), (0, 0)), "edge").reshape(-1, _sp_input_length, ch)
        source_sp = source_sp.astype(np.float32).reshape(-1, _sp_input_length, ch, 1)
        source_ap = np.pad(source_ap, ((padding_size, 0), (0, 0)), "edge").reshape(-1, _sp_input_length, 1025)
        source_ap = np.transpose(source_ap, [0, 2, 1]).astype(np.float32).reshape(-1, 1025, _sp_input_length, 1)
        padding_size = abs(_sp_input_length - self.source_sp_l.shape[0] % _sp_input_length)
        self.source_sp_l = np.pad(self.source_sp_l, ((padding_size, 0), (0, 0)), "edge").reshape(-1, _sp_input_length, ch)
        self.source_sp_l = self.source_sp_l.astype(np.float32).reshape(-1, ch)
        self.source_pp = chainer.backends.cuda.to_gpu(source_sp)
        self.source_f0 = np.exp(np.log(source_f0 + 1.0) * data[2]["log_rate"]) - 1.0
        self.wave_len = data[0].shape[0]
        self.only_score = only_score
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
        result = self.model_en(self.source_pp)
        _, pr = self.model_f0(self.source_pp)
        _, po = self.model_f0(result)
        result = chainer.backends.cuda.to_cpu(result.data)
        pr = chainer.backends.cuda.to_cpu(pr.data)
        pr = np.exp(pr.reshape(-1)[-self.length:]) - 1
        po = chainer.backends.cuda.to_cpu(po.data)
        po = np.exp(po.reshape(-1)[-self.length:]) - 1
        ch = result.shape[2]
        result = result.reshape(-1, ch)
        score_m = np.mean((result[-self.length:] - self.source_sp_l[-self.length:])**2)
        result_wave = world2wave(self.source_f0, result[-self.length:], self.source_ap)
        otp = result_wave.reshape(-1)
        head_cut_num = otp.shape[0]-self.wave_len
        if head_cut_num > 0:
            otp = otp[head_cut_num:]
        chainer.using_config("train", True)
        return otp, score_m, pr, po
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
        out_put, score_raw, pre_f0, post_f0 = self.convert()
        chainer.report({"env_test_loss": score_raw})
        if self.only_score is not None:
            out_puts = (out_put*32767).astype(np.int16)
            image_power_spec = fft(out_put)
            # calculating power spectrum
            image_power_spec = fft(out_put)
            n = min(image_power_spec.shape[0], self.image_power_l.shape[0])
            score_fft = np.mean((image_power_spec[:n]-self.image_power_l[:n]) ** 2)
            chainer.report({"test_loss": score_fft})
            #saving fake power-spec image
            figure = plt.figure(figsize=(8, 5))
            gs = mpl.gridspec.GridSpec(nrows=6, ncols=2)
            plt.subplots_adjust(hspace=0)
            figure.add_subplot(gs[:3, :])
            plt.subplots_adjust(top=0.95)
            plt.title(self.model_name, pad=0.2)
            _insert_image = np.transpose(image_power_spec, (1, 0))
            plt.tick_params(labelbottom=False)
            plt.imshow(_insert_image, vmin=-1.0, vmax=1.0, aspect="auto")
            ax = figure.add_subplot(gs[3:4, :])
            plt.tick_params(labeltop=False, labelbottom=False, labelleft=False)
            plt.margins(x=0)
            plt.ylim(-1, 1)
            ax.grid(which="major", axis="x", color="blue", alpha=0.8, linestyle="--", linewidth=1)
            _t = out_put.shape[0] / 44100
            _x = np.linspace(0, _t, out_put.shape[0])
            plt.plot(_x, out_put)
            figure.add_subplot(gs[4:5, :])
            plt.plot(np.abs(np.mean(image_power_spec, axis=0)-np.mean(self.image_power_l, axis=0)))
            plt.plot(np.abs(np.std(image_power_spec, axis=0)-np.std(self.image_power_l, axis=0)))
            plt.tick_params(labelbottom=False)
            figure.add_subplot(gs[5:6, :])
            plt.margins(x=0)
            plt.plot(self.l_f0)
            plt.plot(pre_f0)
            plt.plot(post_f0)
            plt.tick_params(labelbottom=False)
            table = plt.table(cellText=[["iteraiton", "fft_diff", "spenv_diff"], ["%d (%s)" % (_trainer.updater.iteration, self.only_score), "%f" % score_fft, "%f" % score_raw]])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            plt.savefig("%s%05d.png" % (self.dir, _trainer.updater.iteration))
            plt.savefig("./latest.png")
            #saving fake waves
            path_save = self.dir + str(self.model_name)+"_"+ str(_trainer.updater.iteration).zfill(5)
            voiced = out_puts.astype(np.int16)
            wave_data = wave.open(path_save + ".wav", 'wb')
            wave_data.setnchannels(1)
            wave_data.setsampwidth(2)
            wave_data.setframerate(44100)
            wave_data.writeframes(voiced.reshape(-1).tobytes())
            wave_data.close()
            wave_data = wave.open("latest.wav", 'wb')
            wave_data.setnchannels(1)
            wave_data.setsampwidth(2)
            wave_data.setframerate(44100)
            wave_data.writeframes(voiced.reshape(-1).tobytes())
            wave_data.close()
            plt.clf()
def fft(_data, nfft=2048):
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
    shift = nfft //2
    time_ruler = _data.shape[0] // shift
    if _data.shape[0] % shift == 0:
        time_ruler -= 1
    window = np.hamming(nfft)
    pos = 0
    wined = np.zeros([time_ruler, nfft])
    for fft_index in range(time_ruler):
        frame = _data[pos:pos + nfft]
        padding_size = nfft-frame.shape[0]
        if padding_size > 0:
            frame = np.pad(frame, (0, padding_size), "constant")
        wined[fft_index] = frame * window
        pos += shift
    fft_r = np.fft.fft(wined, n=nfft, axis=1)
    spec_re = fft_r.real.reshape(time_ruler, -1)
    spec_im = fft_r.imag.reshape(time_ruler, -1)
    spec_po = np.log(np.power(spec_re, 2) + np.power(spec_im, 2) + 1e-8).reshape(time_ruler, -1)[:, -512:]
    spec_po = np.clip((spec_po + 5) / 10, -1.0, 1.0)
    return spec_po
