
import wave
import pyworld as pw
import numpy as np
from matplotlib import pyplot as plt
if __name__ == "__main__":

    file = "dataset/test/label.wav"

    with wave.open(file, 'rb') as wf:
        _n = wf.getnframes()
        _d = wf.readframes(_n)
        int_wave = np.frombuffer(_d, 'int16')
    raw_wave = int_wave.reshape(-1) / 32767
    f0, sp, ap = pw.wav2world(raw_wave, 44100, frame_period=10.0)
    sp = np.clip((np.log(sp+1e-8) + 10) /20, -1.0, 1.0)
    sp[:, 65:] = -1.0
    sp = np.exp(sp * 20 - 10)
    syn_wave = pw.synthesize(f0, sp, ap, 44100, frame_period=10.0)
    f0_hat, _, _ = pw.wav2world(syn_wave, 44100, frame_period=10.0)
    voiced = np.concatenate([(raw_wave * 32767).astype(np.int16), (syn_wave * 32767).astype(np.int16)])
    wave_data = wave.open("latest.wav", 'wb')
    wave_data.setnchannels(1)
    wave_data.setsampwidth(2)
    wave_data.setframerate(44100)
    wave_data.writeframes(voiced.reshape(-1).tobytes())
    wave_data.close()
    plt.plot(f0)
    plt.plot(f0_hat)
    plt.show()
    