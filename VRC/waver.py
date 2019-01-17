import subprocess
import os
from glob import glob
import wave
import numpy as np
from pyworld import pyworld as pw

class Waver():
    def __init__(self, block=4096, fs=16000, fft_size=1024, bit_rate=16):
        self.block = block
        self.fs = fs
        self.fft_size = fft_size
        self.bit_rate = bit_rate

        self._sox = "C:\\Users\\tal\\bin\\sox-14.4.2\\sox.exe"

        if ~(fft_size & (fft_size - 1)) + 1:
            raise Exception("fft size is required power of 2")

        if block % fft_size != 0:
            raise Exception("block be valued n times fft_size")

        self.frame_period = block * 16 // fft_size * 1000 / fs

    def _read(self, file, filter_silent=False):
        temp = "temp.wav"
        res = subprocess.call("%s -r %d -c 1 -b %d %s %s" %
                                (self._sox, self.fs, self.bit_rate, file, temp))
        if res != 0:
            raise Exception("sox command error")

        if filter_silent:
            temp2 = "temp2.wav"
            res = subprocess.call("%s %s %s silence 1 0.08 %f%% 1 0.08 %f%% : restart" % 
                                (self._sox, temp, temp2, filter_silent, filter_silent))
            os.remove(temp)
            temp = temp2

        wf = wave.open(temp, "rb")
        data = wf.readframes(wf.getnframes())
        data = np.frombuffer(data, 'int16')
        wf.close()
        os.remove(temp)

        data = np.array(data / (pow(2, self.bit_rate - 1) - 1)).astype(np.float)
        return data

    def encode_block(self, data):
        _f0, t = pw.dio(data, self.fs, frame_period=self.frame_period)
        f0 = pw.stonemask(data, _f0, t, self.fs)
        sp = pw.cheaptrick(data, f0, t, self.fs, fft_size=self.fft_size)
        ap = pw.d4c(data, f0, t, self.fs, fft_size=self.fft_size)
        psp = np.clip((np.log(sp) + 15) / 20, -1.0, 1.0)
        return f0, sp, ap, psp

    def encode(self, files, filter_silent=False):
        if type(files) is str:
            files = glob(files)
        
        data = np.array([])
        for file in files:
            data = np.append(data, self._read(file, filter_silent=filter_silent))

        shortage = len(data) % self.block
        data = np.pad(data, (0, shortage), "constant")
        time = len(data) // self.block
        
        result_f0 = np.array([])
        result_sp = []
        result_ap = []
        result_psp = []
        for i in range(time):
            d = data[i * self.block:(i+1) * self.block]
            f0, sp, ap, psp = self.encode_block(d)

            result_f0 = np.append(result_f0, f0)
            result_sp.append(sp)
            result_ap.append(ap)
            result_psp.append(psp)

        return (result_f0, 
                np.concatenate(result_sp, axis=0), 
                np.concatenate(result_ap, axis=0), 
                np.array(result_psp))

    def decode(self, f0, ap, sp=None, psp=None, file=None):
        if sp is None and psp is None:
            raise Exception("ValueError: sp or psp is required defines only one")
        
        sp = np.exp(psp * 20 - 15) if sp is None else sp
        data = pw.synthesize(f0, sp, ap, self.fs, self.frame_period)
        if file is None:
            return data

        d = (data * (pow(2, self.bit_rate - 1) - 1))
        if self.bit_rate == 16:
            d = d.astype(np.int16)
        elif self.bit_rate == 8:
            d = d.astype(np.int8)
        else:
            d = d.astype(np.int)
        
        wf = wave.open(file, "wb")
        wf.setparams((
            1, # channel
            self.bit_rate // 8,
            self.fs,
            len(d),
            "NONE", "not compressed"
        ))
        wf.writeframes(d.tobytes())
        wf.close()

        return data

    def get_f0_transfer_params(self, f0_a, f0_b):
        a = f0_a[f0_a > 0]
        b = f0_b[f0_b > 0]
        return np.mean(a) - np.mean(b), np.std(b) / np.std(a)

    def generate_f0_transfer(self, f0_mean_diff, f0_vars_rate):
        def transfer(f0):
            return (f0 - f0_mean_diff) * f0_vars_rate
        return transfer

