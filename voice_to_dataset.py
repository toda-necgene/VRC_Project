import numpy as np
import wave
import time
import glob
import pyworld.pyworld as pw
import os

import util


class VoiceToDataset:
    """
    WAVEデータ群をパッチデータに変換する

    Attributes
    ----------
    voice_dir : str
        WAVEデータ群が格納されているルートディレクトリ
    train_dir : str
        学習用データを保存するディレクトリ
    term : int
        音響特性を計算するときのブロックサイズ
    rate : int
        WAVEデータのサンプリング周波数
    """
    def __init__(self, voice_dir, train_dir, term=4096, rate=16000, fft_size=1024):
        """
        Parameters
        ----------
        voice_dir : str
            WAVEデータ群が格納されているルートディレクトリ
        train_dir : str
            学習用データを保存するディレクトリ
        term : int optional
            音響特性を計算するときのブロックサイズ
        rate : int optional
            WAVEデータのサンプリング周波数
        """
        self.voice_dir = voice_dir
        self.train_dir = train_dir
        self.fft_size = fft_size
        self.term = term
        self.rate = rate

    def convert(self, source, target):
        """
        第一, 第二話者のデータを変換する

        Parameters
        ----------
        source : str
            第一話者名、実際のWAVEデータはこの名前のサブディレクトリに格納されていること
        target : str
            第二話者名、実際のWAVEデータはこの名前のサブディレクトリに格納されていること

        Returns
        -------
        plof : np.array [第一話者と第二話者のピッチ平均の差, 第一話者と第二話者のピッチの標準偏差の比]
            第一話者のf0を第二話者に変換したいときは (f0 - plof[0]) * plof[1]
        """
        pitch = {}
        for name in [source, target]:
            files = sorted(
                glob.glob(os.path.join(self.voice_dir, name, "*.wav")))
            ff = list()
            m = list()
            for file in files:
                print(" [*] パッチデータに変換を開始します。 :", file)
                wf = wave.open(file, 'rb')
                dms = wf.readframes(wf.getnframes())
                data = np.frombuffer(dms, 'int16')
                data_real = (data / 32767).reshape(-1).astype(np.float)
                times = (data_real.shape[0] - 1) // self.term + 1

                endpos = data_real.shape[0] % self.term
                for i in range(times):
                    data_realAb = data_real[max(endpos -
                                                self.term, 0):endpos].copy()
                    shortage = self.term - data_realAb.shape[0]
                    if shortage > 0:
                        data_realAb = np.pad(data_realAb, (shortage, 0),
                                             "constant")

                    f0, psp, _ = util.encode(data_realAb, self.rate, fft_size=self.fft_size, frame_period=5.0)
                    ff.extend(f0[f0 > 0.0])
                    m.append(psp)

                    endpos += self.term * i

            m = np.asarray(m, dtype=np.float32)
            np.save(os.path.join(self.train_dir, name + ".npy"), m)
            print(" [*] " + name + "データ変換完了")
            pitch[name] = {}
            pitch[name]["mean"] = np.mean(ff)
            pitch[name]["var"] = np.std(ff)

        pitch_mean_s = pitch[source]["mean"]
        pitch_var_s = pitch[source]["var"]
        pitch_mean_t = pitch[target]["mean"]
        pitch_var_t = pitch[target]["var"]

        plof = np.asarray([pitch_mean_s - pitch_mean_t, pitch_var_t / pitch_var_s])
        return plof

