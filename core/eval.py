"""
製作者:TODA

学習時に逐次行うテストクラスの定義
"""
import os
import wave
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime as dt
from core.world_and_wave import wave2world_lofi, world2wave, fft
class TestModel():
    """
    テストを行うExtention
    """
    def __init__(self, args, data, name_ad=""):
        """
        変数の初期化と事前処理
        Parameters
        ----------
        : Generator
            評価用トレーナ
        args: str
            ファイル出力ディレクトリパス
        data : tuple or list
            [0]: np.ndarray
                変換元音声
                range: [-1.0, 1.0]
                dtype: float
            [1]: np.ndarray
                目標話者パラレルテストデータのパワースペクトラム
                Noneならば比較を省略
            [2]: dict of int
                f0に関するデータ
        """
        self.dir = args["wave_otp_dir"]
        if not os.path.exists(self.dir):
            os.makedirs(args["wave_otp_dir"])
    
        self.model_name = args["version"]
        self.name_ad = name_ad
        
        source_f0, source_sp, self.source_ap = wave2world_lofi(data[0].astype(np.float64))
        self.target = data[1]
        ch = source_sp.shape[1]
        padding_size = abs(args["length_sp"] - source_sp.shape[0] % args["length_sp"])
        source_sp = np.pad(source_sp, ((padding_size, 0), (0, 0)), "edge").reshape(-1, args["length_sp"], ch, 1).astype(np.float32)
        source_sp = source_sp.transpose([0, 2, 1, 3])
        Tensor = torch.cuda.FloatTensor if args["gpu"] >= 0 else torch.Tensor
        self.source_pp = Tensor(source_sp)
        self.source_f0 = (source_f0 - data[2]["pre_sub"]) * np.sign(source_f0) * data[2]["pitch_rate"] + data[2]["post_add"] * np.sign(source_f0)
        self.wave_len = data[0].shape[0]
        _, self.target_sp, _ = wave2world_lofi(data[1].astype(np.float64))
        self.target_sp = np.pad(self.target_sp, ((padding_size, 0), (0, 0)), "edge").reshape(-1, ch)
        padding_size = abs(args["length_sp"] - self.target_sp.shape[0] % args["length_sp"])
        self.target_sp = self.target_sp.astype(np.float32)
        self.length = source_f0.shape[0]
        self.image_power_l = fft(data[1])
    def __call__(self, iteration, writer, model):
        """
        評価関数
        やっていること
        パワースペクトラム・スペクトラム包絡の比較評価
        音声/画像の保存
        Parameters
        ----------
        iteration: int
            イテレーション数
        """
        # test convert
        result = model(self.source_pp)
        result = result.cpu().detach().numpy()
        ch = result.shape[1]
        result = result.transpose([0, 2, 1, 3]).reshape(-1, ch)
        n = min(result.shape[0], self.target_sp.shape[0])
        score_env = np.mean((result[:n] - self.target_sp[:n])**2)
        result_wave = world2wave(self.source_f0, result[-self.length:], self.source_ap)
        out_put = result_wave.reshape(-1)
        over_head_num = out_put.shape[0]-self.wave_len
        if over_head_num > 0:
            out_put = out_put[over_head_num:]
        out_puts = (out_put*32767).astype(np.int16)
        image_power_spec = fft(out_put)
        # calculating power spectrum
        image_power_spec = fft(out_put)
        n = min(image_power_spec.shape[0], self.image_power_l.shape[0])
        score_fft = np.mean((image_power_spec[:n]-self.image_power_l[:n]) ** 2)
        # plot fake power-spec image
        figure = plt.figure(figsize=(8, 5))
        gs = mpl.gridspec.GridSpec(nrows=5, ncols=2)
        plt.subplots_adjust(hspace=0)
        ## power-spec image plot
        figure.add_subplot(gs[:3, :])
        plt.subplots_adjust(top=0.95)
        plt.title(self.model_name, pad=0.2)
        _insert_image = np.transpose(image_power_spec, (1, 0))
        plt.tick_params(labelbottom=False)
        plt.imshow(_insert_image, vmin=-1.0, vmax=1.0, aspect="auto")
        ## wave_form plot
        ax = figure.add_subplot(gs[3:4, :])
        plt.tick_params(labeltop=False, labelbottom=False)
        plt.margins(x=0)
        plt.ylim(-1, 1)
        ax.grid(which="major", axis="x", color="blue", alpha=0.8, linestyle="--", linewidth=1)
        _t = out_put.shape[0] / 44100
        _x = np.linspace(0, _t, out_put.shape[0])
        plt.plot(_x, out_put)
        ## differense (separated by frequency) plot
        figure.add_subplot(gs[4:, :])
        plt.plot(np.abs(np.mean(image_power_spec, axis=0)-np.mean(self.image_power_l, axis=0)))
        plt.plot(np.abs(np.std(image_power_spec, axis=0)-np.std(self.image_power_l, axis=0)))
        ## info table plot
        plt.tick_params(labelbottom=False)
        table = plt.table(cellText=[["time stamp", "iteraiton", "fft_diff", "spenv_diff"],
                                    [dt.now().strftime("[%x]%X"),
                                    "%d" % (iteration),
                                    "{:.4f}".format(score_fft),
                                    "{:.4f}".format(score_env)]])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        plt.savefig("%s%s%05d.png" % (self.dir, self.name_ad, iteration))
        plt.savefig("./latest.png")
        #saving fake waves
        path_save = self.dir + str(self.model_name)+"_"+self.name_ad+"_"+ str(iteration).zfill(5)
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
        writer.add_audio("fake", out_put, iteration)   
        writer.add_figure("result", figure, iteration)
        plt.clf()
        plt.close()
        return score_fft, score_env
