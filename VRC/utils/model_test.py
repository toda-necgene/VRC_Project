import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
import time
import os

import librosa,librosa.display
import sklearn

from .model_proto_cpu import Model as model
from datetime import datetime
import glob

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
path="./setting.json"
net=model(path)
net.build_model()
if not net.load():
    print(" [x] load failed...")
    exit(-1)
print(" [*] load success!!")
def ank(x,y):
    return  np.mean(np.abs(x-y),axis=0)
def search(x,y,num=51):
    x0=x[:,:,:].copy()
    ym=y.copy()
    cc = np.mean(x0[:, :, 0], axis=1)
    x0[cc < -10] = -10.0
    cc = np.mean(ym[:, :, 0], axis=1)
    ym[cc < -10] = -10.0
    ys = [y[:,:,:].copy()]
    for i in range(-(num-1)//2,0):
        ys.append(np.roll(ym[:,:,:],i))
    for i in range(0, (num - 1) // 2):
        ys.append(np.roll(ym[:,:,:], i))


    a=np.asarray([ank(x0,c) for c in ys]).reshape(num,NFFT,-1)
    b=np.min(a, axis=0)
    return b
def filter_clip(dd,f=1.5):
    dxf=np.maximum(dd,-f)+f+np.minimum(dd,f)-f
    return dxf

def filter_mean(dd):
    dxx1=np.roll(dd,1)
    dxx1[:1]=dd[:1]
    return (dd+dxx1)/2.0

def filter_pes(dd):
    dxx1=np.roll(dd,-1)
    dxx1[:1]=0
    return dd-dxx1*0.27
def mask_const(dd,f,t,power):
    dd[:,f:t,0]-=power
    # dd[:,:,1]=dd[:,:,1]*1.12
    return dd
def mask_scale(dd,f,t,power):
    dd[:,f:t,0]-=(power/100*-dd[:,f:t,0])

    return dd
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
def shift(data_inps,pitch):
    data_inp=data_inps.reshape(-1)
    return scale(time_strech(data_inp,1/pitch),data_inp.shape[0])

def scale(inputs,len_wave):
    x=np.linspace(0.0,inputs.shape[0]-1,len_wave)
    ref_x_n=(x+0.5).astype(int)
    spec=inputs[ref_x_n[...]]
    return spec.reshape(-1)
def time_strech(datanum,speed):
    term_s = int(rate * 0.05)
    fade=term_s//2
    pulus=int(term_s*speed)
    data_s=datanum.reshape(-1)
    spec=np.zeros(1)
    ifs=np.zeros(fade)
    for i_s in np.arange(0.0,data_s.shape[0],pulus):
        st=int(i_s)
        fn=min(int(i_s+term_s+fade),data_s.shape[0])
        dd=data_s[st:fn]
        if i_s + pulus >= data_s.shape[0]:
            spec = np.append(spec, dd)
        else:
            ds_in = np.linspace(0, 0.999, fade)
            ds_out = np.linspace(0.999, 0, fade)
            stock = dd[:fade]
            dd[:fade] = dd[:fade] * ds_in
            if st != 0:
                dd[:fade] += ifs[:fade]
            else:
                dd[:fade] += stock * np.linspace(0.999, 0, fade)
            if fn!=data_s.shape[0]:
                ifs = dd[-fade:] * ds_out
            spec=np.append(spec,dd[:-fade])
    return spec[1:]

def complex_to_pp(fft_r):
    time_ruler=fft_r.shape[0]
    re = fft_r.real
    im = fft_r.imag
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-34).reshape(time_ruler, -1, 1)
    d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
    spec = np.concatenate((c, d), 2)
    return spec

CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_OUTPUT_FILENAME = "./テスト.wav"
WAVE_OUTPUT_FILENAME2 = "./テスト-2.wav"
WAVE_OUTPUT_FILENAME3 = "./天才.wav"
WAVE_OUTPUT_FILENAME4 = "./天才-2.wav"
file_l="./train/Model/datasets/test/label.wav"
file_l2 = "./テスト.wav"
file_l3="./train/Model/datasets/test/label.wav"
file3="./train/Model/datasets/test/B2.wav"
file4="./train/Model/datasets/test/B.wav"
file="./train/Model/datasets/test/test.wav"

index=0
dms=[]
wf = wave.open(file, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')
data_realA=data.reshape(-1).astype(np.float32)


dms=[]
wf = wave.open(file3, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')
data_realC=data.reshape(-1).astype(np.float32)


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
# data_realA=filter_pes (data_realA)
data_realA=data_realA.reshape(1,-1,1)
data_realC=data_realC.reshape(1,-1,1)
print(" [*] conversion start!!")
data_C,_,data_E=net.convert(data_realA)
data_G,_,_=net.convert(data_realC)

print(" [*] conversion finished in %3.3f!!" % (time.time()-tm))
data_C=data_C.reshape(-1)
data_D=(data_C.copy()*2.5).astype(np.int16)
timee=80000
times=data_realB.shape[0]//timee

rate=16000

ab=np.zeros([NFFT,2])
abc=np.zeros([1,NFFT,2])
abb=np.zeros([1,NFFT,2])
vvr=np.zeros([1])
times=data_realB.shape[0]//term+1
if data_realB.shape[0]%term==0:
    times-=1
ttm=time.time()
resp=np.zeros([NFFT//2])
print("----------------------------------------------------------------")
for i in range(times):
    ind=term+SHIFT
    startpos=term*i+data_realB.shape[0]%term
    data_realAb = data_realA[max(startpos - ind, 0):startpos]
    data_realBb = data_realB[max(startpos - ind, 0):startpos]
    r=ind-data_realBb.shape[0]
    if r>0:
        data_realBb=np.pad(data_realBb,(r,0),"constant")
    ddms=data_realBb.astype(np.float32)/32767.0
    bss=fft(ddms)
    bss=complex_to_pp(bss)
    abc = np.append(abc, bss, axis=0)
data_E=data_E.reshape([-1,NFFT,2])

FORMAT=pyaudio.paInt16
p=pyaudio.PyAudio()
ww = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(data_C.tobytes())
ww.close()
FORMAT=pyaudio.paInt16
p=pyaudio.PyAudio()
ww = wave.open(WAVE_OUTPUT_FILENAME2, 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(data_D.tobytes())
ww.close()

p=pyaudio.PyAudio()
ww = wave.open(WAVE_OUTPUT_FILENAME3, 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(data_G.tobytes())
ww.close()


mix=np.append(data_C,data_realA).astype(np.int16)

ww = wave.open("ミックス.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(mix.tobytes())
ww.close()
pl.subplot(6,1,5)
aba=abc[1:,:,0].transpose((1,0))
pl.imshow(aba,aspect="auto")
pl.clim(-30,10)
pl.colorbar()
pl.subplot(6,1,3)
abn=data_E[1:,:,0].transpose((1,0))
pl.imshow(abn,aspect="auto")
pl.clim(-30,10)
pl.colorbar()
pl.subplot(6,1,6)
aba=abc[1:,:,1].transpose((1,0))
pl.imshow(aba,aspect="auto")
pl.clim(-3.141592,3.141592)
pl.colorbar()
pl.subplot(6,1,4)
abn=data_E[1:,:,1].transpose((1,0))
pl.imshow(abn,aspect="auto")
pl.clim(-3.141592,3.141592)
pl.colorbar()


data_E=data_E.reshape([-1,NFFT,2])
x,xfs = librosa.load(file_l2, sr=16000)
y,yfs = librosa.load(file_l3, sr=16000)

mfccsx = librosa.feature.mfcc(x, sr=xfs)
mfccsy = librosa.feature.mfcc(y, sr=yfs)
mfccsx = sklearn.preprocessing.scale(mfccsx, axis=1)
mfccsy = sklearn.preprocessing.scale(mfccsy, axis=1)

pl.subplot(6,1,1)
librosa.display.specshow(mfccsx, sr=xfs, x_axis='time')
pl.colorbar()

pl.subplot(6,1,2)
librosa.display.specshow(mfccsy, sr=yfs, x_axis='time')
pl.colorbar()

pl.show()
