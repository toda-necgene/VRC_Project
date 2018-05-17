import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
NFFT=512
SHIFT=NFFT//2
def fft(data):
    time_ruler=data.shape[0]//SHIFT-1
    window=np.hamming(NFFT)
    spec=np.zeros([time_ruler,NFFT,2])
    pos=0
    for fft_index in range(time_ruler):
        frame=data[pos:pos+NFFT]
        if len(frame)==NFFT:
            wined=frame*window
            fft_r=np.fft.fft(wined)
            re=fft_r.real.reshape(-1,1)
            im=fft_r.imag.reshape(-1,1)
            fft_data=np.concatenate((re,im),1)
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

b=np.empty([])
ab=np.empty([0,NFFT,2])
term=16384
times=data_realA.shape[0]//term+1
if data_realA.shape[0]%16384==0:
    times-=1
for i in range(times):
    ind=SHIFT+term
    data_realAb = data_realA[max(term*i-SHIFT,0):min([term*(i+1),data_realA.shape[0]-1])]
    r=ind-data_realAb.shape[0]
    if r>0:
        data_realAb=np.pad(data_realAb,(0,r),"constant")
    a=fft(data_realAb/32767.0)
    ab = np.append(ab, a,axis=0)
    c = np.log(np.power(a[:,:,0], 2) + np.power(a[:,:,1], 2) + 1e-24)
    d = np.arctan2(a[:,:,1],a[:,:,0])
    p=np.sqrt(np.exp(c))
    r=p *(np.cos(d))
    i=p *(np.sin(d))
    ep = np.concatenate((r.reshape(r.shape[0],r.shape[1],1),i.reshape(i.shape[0],i.shape[1],1)),2)
    print(np.mean(a-ep))
    s=ifft(ep)
    b=np.append(b,s)
# print(a)
print(b.shape)
b=(b[1:data_realA.shape[0]+1]/2*32767).astype(np.int16)
c=np.log(np.power(ab[:,:,0],2)+np.power(ab[:,:,1],2)+1e-8)
d=np.arctan2(ab[:,:,1],ab[:,:,0])
e=np.sqrt(np.exp(c))*np.exp(d)
pl.plot(b)
p=pyaudio.PyAudio()
ww = wave.open("B.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(b.tobytes())
ww.close()
pl.show()
