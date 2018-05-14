import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
NFFT=128

def fft(data):
    stop=data.shape[0]
    step=(NFFT//2)
    time_ruler=stop//step
    window=np.hamming(NFFT)
    spec=np.zeros([time_ruler,NFFT,2])
    pos=0
    for fft_index in range(time_ruler):
        frame=data[pos:pos+NFFT]
        if len(frame)==NFFT:
            wined=frame*window
            fft=np.fft.fft(wined)
            fft_data=np.asarray([fft.real,fft.imag])
            fft_data=np.transpose(fft_data, (1,0))
            for i in range(len(spec[fft_index])):
                spec[fft_index][i]=fft_data[i]
            pos+=NFFT//2
    return spec

def ifft(data):
    data=data[:,:,0]+1j*data[:,:,1]
    time_ruler=data.shape[0]
    window=np.hamming(NFFT)
    spec=np.zeros([])
    pos=0
    lats = np.zeros([NFFT//2])
    for _ in range(time_ruler):
        frame=data[pos]
        fft=np.fft.ifft(frame)
        fft_data=fft.real
        fft_data/=window
        v = lats + fft_data[:NFFT//2]
        lats = fft_data[NFFT//2:]
        spec=np.append(spec,v)
        pos+=1

    return spec[1:]
FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "../train/Model/datasets/test/"
WAVE_TMP="tmp.wav"
WAVE_OUTPUT_FILENAME = "../train/Model/datasets/train/01/"
file=(WAVE_INPUT_FILENAME+"/test.wav")
index=0
dms=[]
wf = wave.open(file, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')[0:4480000].reshape(2,-1)
data_realA=data.reshape(-1)
time=80000
times=data_realA.shape[0]//time
mod=80000//5

rate=16000
nos=5

b=np.empty([])
times=data_realA.shape[0]//8192+1
if data_realA.shape[0]%8192==0:
    times-=1
for i in range(times):
    data_realAb = data_realA[8192*i:min([8192*(i+1),data_realA.shape[0]-1])]
    r=8192-data_realAb.shape[0]
    if r>0:
        data_realAb=np.pad(data_realAb,(0,r),"constant")
    a=fft(data_realAb/32767.0)
    a+=np.random.normal(0.0,nos,a.shape)
    s=ifft(a)
    print(s.shape)
    b=np.append(b,s)
# print(a)
b=(b[1:]*(32767/8)).astype(np.int16)
pl.subplot(4,1,1)
pl.plot(data_realA/32767.0)
pl.subplot(4,1,2)
pl.plot(b/5)
pl.subplot(4,2,1)
c=np.abs(a[:,:,0]+1j*a[:,:,1])**2
pl.imshow(c)
pl.colorbar()
pl.subplot(4,2,2)
d=np.log(a[:,:,0]**2+a[:,:,1]**2+1e-10)
pl.imshow(d)
pl.colorbar()
p = pyaudio.PyAudio()
ww = wave.open("B.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(b.tobytes())
ww.close()
pl.show()
