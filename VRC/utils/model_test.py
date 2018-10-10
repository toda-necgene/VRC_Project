import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
import time
import sklearn,librosa.display
from model.model_cpu import Model as model

Add_Effect=True
NFFT=1024
SHIFT=NFFT//2
C1=32.703
rate=16000
Hz=C1*(2**0)
now=317.6
target=563.666
term=4096
upidx=target/now
upidx=1.0
path="setting.json"
net=model(path)
if not net.load():
    print(" [x] load failed...")
    exit(-1)
print(" [*] load success!!")
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_OUTPUT_FILENAME = "./B.wav"
WAVE_OUTPUT_FILENAME2 = "./B2.wav"
file_ll="./datasets/test/label2.wav"
file_l="./datasets/test/test2.wav"
file="./datasets/test/test.wav"

index=0
dms=[]
wf = wave.open(file, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')
data_realA=data.reshape(-1)

dms=[]
wf = wave.open(file_l, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')
data_realB=data.reshape(-1)


tm=time.time()
print(" [*] conversion start!!")
data_C,_,data_F=net.convert(data_realA)
data_D,_,data_E=net.convert(data_realB)
print(" [*] conversion finished in %3.3f!!" % (time.time()-tm))

rate=16000
FORMAT=pyaudio.paInt16

p=pyaudio.PyAudio()
ww = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(data_C.tobytes())
ww.close()


ww = wave.open(WAVE_OUTPUT_FILENAME2, 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(data_D.tobytes())
ww.close()

pl.subplot(7,1,1)
x, fs = librosa.load('./datasets/test/label.wav', sr=16000)
mfccsa =librosa.feature.mfcc(x, sr=fs)
# mfccsa = sklearn.preprocessing.scale(mfccsa, axis=1)
librosa.display.specshow(mfccsa, sr=fs, x_axis='time')
pl.colorbar()

pl.subplot(7,1,4)
mfccs =librosa.feature.melspectrogram(x, sr=fs)
pl.imshow(mfccs,aspect="auto")
pl.clim(0,100)
pl.colorbar()
pl.subplot(7,1,3)
x, fs = librosa.load(WAVE_OUTPUT_FILENAME, sr=16000)
mfccsb = librosa.feature.mfcc(x, sr=fs)
# mfccsb = sklearn.preprocessing.scale(mfccsb, axis=1)
librosa.display.specshow(mfccsb, sr=fs, x_axis='time')
pl.colorbar()
pl.subplot(7,1,6)
mfccs =librosa.feature.melspectrogram(x, sr=fs)
pl.imshow(mfccs,aspect="auto")
pl.clim(0,100)
pl.colorbar()
pl.subplot(7,1,2)
x, fs = librosa.load('./datasets/test/test.wav', sr=16000)
mfccsc =librosa.feature.mfcc(x, sr=fs)
librosa.display.specshow(mfccsc, sr=fs, x_axis='time')
pl.colorbar()
pl.subplot(7,1,5)
mfccs =librosa.feature.melspectrogram(x, sr=fs)
pl.imshow(mfccs,aspect="auto")
pl.clim(0,100)
pl.colorbar()
pl.show()
