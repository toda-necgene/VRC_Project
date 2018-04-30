import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss

    return noisy

def fft(data):
    rate=16000
    NFFT=32
    time_song=float(data.shape[0])/rate
    time_unit=1/rate
    start=0
    stop=time_song
    step=(NFFT//2)*time_unit
    time_ruler=np.arange(start,stop,step)
    window=np.hamming(NFFT)
    spec=np.zeros([len(time_ruler),(NFFT),2])
    pos=0
    for fft_index in range(len(time_ruler)):
        frame=data[pos:pos+NFFT]
        if len(frame)==NFFT:
            wined=frame*window
            fft=np.fft.fft(wined)
            fft_data=np.asarray([fft.real,fft.imag])
            fft_data=np.reshape(fft_data, (32,2))
            for i in range(len(spec[fft_index])):
                spec[fft_index][-i-1]=fft_data[i]
            pos+=NFFT//2
    return spec
def ifft(data):
    data=data[:,:,0]+1j*data[:,:,1]
    time_ruler=data.shape[0]
    window=np.hamming(32)
    spec=np.zeros([])
    pos=0
    for _ in range(time_ruler):
        frame=data[pos]
        fft=np.fft.ifft(frame)
        fft_data=fft.real
        fft_data/=window
        spec=np.append(spec,fft_data)
        pos+=1

    return spec[1:]
FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "../train/Model/datasets/source/"
WAVE_TMP="tmp.wav"
WAVE_OUTPUT_FILENAME = "../train/Model/datasets/train/01/"
file=(WAVE_INPUT_FILENAME+"/結合済み.wav")
index=0
dms=[]
wf = wave.open(file, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')[0:4480000].reshape(2,-1)
data_realA=data[0]
data_realB=data[1]
time=80000
times=data_realA.shape[0]//time
mod=80000//5

data_realA=data_realA[0:8192]
rate=16000

print(data_realA/32767.0)
a=fft(data_realA/32767.0)
b=ifft(a)
# print(a)

print(a.shape)
print(b.shape)
pl.subplot(2,1,1)
pl.plot(data_realA/32767.0)
pl.subplot(2,1,2)
pl.plot(b/5)
pl.show()