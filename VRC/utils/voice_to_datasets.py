import pyaudio
import numpy as np
import wave
import time
import glob
import cupy
import matplotlib.pyplot as plt
NFFT=128
SHIFT=NFFT//2
C1=32.703
rate=16000
Hz=C1*(2**0)
now=317.6
target=563.666
term = 4096
upidx=target/now

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
    wined=cupy.asarray(wined, dtype=cupy.float64)
    fft_rs=cupy.fft.fft(wined,n=NFFT,axis=-1)
    fft_rs=cupy.asnumpy(fft_rs)
    return fft_rs.reshape(time_ruler, -1)
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
    c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(time_ruler, -1, 1)
    d = np.arctan2(im, re).reshape(time_ruler, -1, 1)
    spec = np.concatenate((c, d), 2)
    return spec

FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "../train/Model/datasets/source/02"
files=glob.glob(WAVE_INPUT_FILENAME+"/*.wav")
name="4096-128-2/Answer_data"
cnt=0
for file in files:
    print(file)
    index=0
    dms=[]
    wf = wave.open(file, 'rb')
    dds = wf.readframes(CHUNK)
    while dds != b'':
        dms.append(dds)
        dds = wf.readframes(CHUNK)
    dms = b''.join(dms)
    data = np.frombuffer(dms, 'int16')
    data_real=data.reshape(-1)
    data_realA=data_real
    timee=data_realA.shape[0]

    rate=16000

    b=np.zeros([1])
    ab=np.zeros([1,128,2])
    abc=np.zeros([1,128,2])

    times=data_realA.shape[0]//term+1
    if data_realA.shape[0]%term==0:
        times-=1
    ttm=time.time()
    resp=np.zeros([NFFT//2])
    for i in range(times):
        ind=term+SHIFT
        startpos=term*i+data_realA.shape[0]%term
        data_realAb = data_realA[max(startpos-ind,0):startpos]
        r=ind-data_realAb.shape[0]
        if r>0:
            data_realAb=np.pad(data_realAb,(r,0),"constant")
        dmn=data_realAb/32767.0
        r=SHIFT-dmn.shape[0]%SHIFT
        if r!=SHIFT:
            dmn=np.pad(dmn,(0,r),"reflect")
        a=fft(dmn)
        a=complex_to_pp(a[:,:SHIFT])
        c=a[:,:,0]
        v=1/np.sqrt(np.var(c,axis=1)+1e-36)
        # a[:, :, 0] -= np.tile(np.mean(c, axis=1).reshape(-1, 1), (1, SHIFT))
        # a[:, :, 0]= np.einsum("ij,i->ij",a[:, :, 0],v)
        bb=np.isnan(np.mean(a))
        if bb:
            print("NAN!!")
        np.save("../train/Model/datasets/train/"+str(name)+"/"+str(cnt) +".data", a)
        cnt+=1
plt.subplot(211)
plt.imshow(a[:,:,0],aspect="auto")
plt.colorbar()
plt.subplot(212)
plt.imshow(a[:,:,1],aspect="auto")
plt.colorbar()
plt.show()