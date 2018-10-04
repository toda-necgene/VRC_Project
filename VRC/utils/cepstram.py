import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
import time
NFFT=1024
SHIFT=NFFT//2
def fft(data):
    time_ruler=data.shape[0]//SHIFT
    if data.shape[0]%SHIFT==0:
        time_ruler-=1
    window=np.hamming(NFFT)
    pos=0
    wined=np.zeros([time_ruler,NFFT])
    for fft_index in range(time_ruler):
        frame=data[pos:pos+NFFT]
        wined[fft_index]=frame*window
        pos += NFFT // 2
    fft_rs=np.fft.fft(wined,n=NFFT,axis=-1)
    return fft_rs.reshape(time_ruler, -1)
def complex_to_pp(fft_r):
    time_ruler=fft_r.shape[0]
    re = fft_r.real
    im = fft_r.imag
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
    d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
    spec = np.concatenate((c, d), 2)
    return spec

def pp_to_complex(frame):
    power = np.sqrt(np.exp(frame[:, :, 0]))
    re = power * (np.cos(frame[:, :, 1]))
    im = power * (np.sin(frame[:, :, 1]))
    ep = re + 1j * im
    return ep

def ifft(data,inp):
    window=np.hamming(NFFT)
    fft_s=np.fft.ifft(data,n=NFFT,axis=-1)
    fft_data=fft_s.real
    fft_data[:]/=window
    v = fft_data[:,:NFFT//2]
    lats = np.roll(fft_data[:,NFFT//2:],(1,0))
    res=lats[0,:]
    lats[0,:]=inp
    spec=np.reshape(v+lats,(-1))
    return spec,res
FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "./"
WAVE_OUTPUT_FILENAME = "./test.wav"
file="../datasets/test/label.wav"
# file=WAVE_OUTPUT_FILENAME
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
# wf = wave.open(file_l, 'rb')
# dds = wf.readframes(CHUNK)
# while dds != b'':
#     dms.append(dds)
#     dds = wf.readframes(CHUNK)
# dms = b''.join(dms)
# data = np.frombuffer(dms, 'int16')
data_realB=data.reshape(-1)
print(data_realB.shape)
rate=16000
b=np.zeros([1])
ab=np.zeros([1,1024,2])
abc=np.zeros([1])

term=8192
times=data_realA.shape[0]//term+1
if data_realA.shape[0]%term==0:
    times-=1
ttm=time.time()
resp=np.zeros([NFFT//2])
for i in range(times):
    ind=SHIFT+term
    startpos=term*(i+1)
    data_realAb = data_realA[max(startpos-ind,0):startpos]
    r=ind-data_realAb.shape[0]
    if r>0:
        data_realAb=np.pad(data_realAb,(0,r),"reflect")
    dmn=data_realAb/32767.0
    a=fft(dmn)
    a=complex_to_pp(a)
    m=a[:,:,0]
    m=np.fft.fft(m)
    f=64
    m[:,f:-f]=0
    m=np.fft.ifft(m)
    a[:,:,0]=m.real
    # a=pp_to_complex(a)
    ab = np.append(ab, a, axis=0)
    # s,resp=ifft(a,resp)
    # b=np.append(b,s)
r=b.shape[0]-data_realA.shape[0]
bbb=b
bbb=(bbb[1:]/2*32767).astype(np.int16)
pl.subplot(4,1,1)
aba=np.transpose(ab[1:,:,0],(1,0))
aba[0,0]=10
pl.imshow(aba,aspect="auto")
pl.clim(-30,10)
pl.subplot(4,1,2)
abn=abc[1:]
pl.plot(abn)
pl.subplot(4,1,3)
aba=np.transpose(ab[1:,:,1],(1,0))
pl.imshow(aba,aspect="auto")
aba[0,0]=10
pl.subplot(4,1,4)
pl.plot(bbb)
pl.clim(-3.141592,3.141592)
p=pyaudio.PyAudio()
ww = wave.open("B.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(bbb.tobytes())
ww.close()
pl.show()
