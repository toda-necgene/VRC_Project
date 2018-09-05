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
def ank(x,y):
    return  np.mean(np.abs(x-y),axis=0)
def search(x,y,num=51):
    x0=x[:,:,:].copy()
    ys = [y[:,:,:].copy()]
    for i in range(-(num-1)//2,0):
        ys.append(np.roll(y[:,:,:],i))
    for i in range(0, (num - 1) // 2):
        ys.append(np.roll(y[:,:,:], i))
    a=np.asarray([ank(x0,c) for c in ys]).reshape(num,NFFT,2)
    b=np.min(a, axis=0)
    return b
def filter_clip(dd,f=1.5):
    dxf=np.maximum(dd,-f)+f+np.minimum(dd,f)-f
    return dxf

def filter_mean(dd,t=4):
    sum_dig=(t+1)*t//2
    nn=dd.copy()/sum_dig*(t+1)
    for i in range(1,t+1):
        dxx1=np.roll(dd,i)
        dxx1[:i]=dd[:i]
        nn+=dxx1/sum_dig*(t+1-i)
    return nn.astype(np.int16)

def filter_pes(dd):
    dxx1=np.roll(dd,-1)
    dxx1[:1]=0
    return dd-dxx1*0.59
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
    res = fft_data[-1, NFFT//2 :].copy()
    lats = np.roll(fft_data[:,NFFT//2:],1,axis=0)
    lats[0,:]=inp
    spec=np.reshape(v+lats,(-1))

    return spec,res
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
# data_N=filter_mean(data_C,4)

data_D,_,data_E=net.convert(data_realB)
data_D=filter_mean(data_D)

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

data_E=data_E.reshape([-1,NFFT,2])
data_F=data_F.reshape([-1,NFFT,2])
pl.subplot(7,1,1)
x, fs = librosa.load('./datasets/test/label.wav', sr=16000)
mfccsa =librosa.feature.mfcc(x, sr=fs)
mfccsa = sklearn.preprocessing.scale(mfccsa, axis=1)
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
mfccsb = sklearn.preprocessing.scale(mfccsb, axis=1)
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
mfccsc = sklearn.preprocessing.scale(mfccsc, axis=1)
librosa.display.specshow(mfccsc, sr=fs, x_axis='time')
pl.colorbar()
pl.subplot(7,1,5)
mfccs =librosa.feature.melspectrogram(x, sr=fs)
pl.imshow(mfccs,aspect="auto")
pl.clim(0,100)
pl.colorbar()
pl.subplot(7,1,7)
pl.imshow(mfccsa[:,2:]-mfccsb[:,:])
pl.colorbar()
rnx=min(mfccsa.shape[1],mfccsb.shape[1])
print(np.sum(np.abs(mfccsa[:,-rnx:]-mfccsb[:,-rnx:])))
print(np.sum(np.abs(mfccsc[:,:]-mfccsb[:,:])))
pl.show()
