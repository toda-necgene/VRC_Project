import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import time
NFFT=1024
SHIFT=NFFT//2
C1=32.703
rate=16000
Hz=C1*(2**0)
now=317.6
target=563.666
upidx=target/now
print(upidx)

def filter_clip(dd,f=1.5):
    dxf=np.maximum(dd,f)-f+np.minimum(dd,-f)+f
    return -dxf*0.5

def filter_eps(dd,f=1.5):
    dp=np.roll(dd,1)
    dp[-1]=0
    ds=dd-dp
    dxf=np.maximum(ds,f)-f+np.minimum(ds,-f)+f
    dp = np.roll(dd, -1)
    dp[-1] = 0
    ds = dd - dp
    dxf2 = np.maximum(ds, f) - f + np.minimum(ds, -f) + f
    return -(dxf+dxf2)*1.2

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
file_l=("../train/Model/datasets/test/label.wav")
file=WAVE_OUTPUT_FILENAME
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
print(data_realA.shape)

dms=[]
wf = wave.open(file_l, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')
data_realB=data.reshape(-1)
print(data_realB.shape)

timee=80000
times=data_realA.shape[0]//timee

rate=16000

b=np.zeros([1])
ab=np.zeros([1,1024,2])
abc=np.zeros([1,1024,2])

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
    data_realBb = data_realB[max(startpos - ind, 0):startpos]
    r=ind-data_realAb.shape[0]
    if r>0:
        data_realAb=np.pad(data_realAb,(0,r),"reflect")
        data_realBb=np.pad(data_realBb,(0,r),"reflect")
    dmn=data_realAb/32767.0
    ddms=data_realBb/32767.0
    dmn=shift(dmn,upidx)
    r=SHIFT-dmn.shape[0]%SHIFT
    if r!=SHIFT:
        dmn=np.pad(dmn,(0,r),"reflect")
    a=fft(dmn)
    bss=fft(ddms)
    a=complex_to_pp(a)
    bss=complex_to_pp(bss)
    abc = np.append(abc, bss, axis=0)
    ab = np.append(ab, a, axis=0)
    # a[:,:,0]=bss[:,:,0]
    a=pp_to_complex(a)
    s,resp=ifft(a,resp)
    b=np.append(b,s)
# print(a)
r=b.shape[0]-data_realA.shape[0]
bsd=filter_clip(b,0.75)
bbb=b
# bbb+=bsd
# bsd=filter_eps(bbb,0.5)
# bbb+=bsd
# bsd=filter_eps(bbb,0.5)
# bbb+=bsd
# bsd=filter_eps(bbb,0.5)
# bbb+=bsd
bbb=(bbb[1:]/2*32767).astype(np.int16)
pl.subplot(4,1,1)
aba=np.transpose(ab[1:,:,0],(1,0))
aba[0,0]=10
pl.imshow(aba,aspect="auto")
pl.clim(-30,10)
pl.colorbar()
pl.subplot(4,1,2)
abn=np.transpose(abc[1:,:,0],(1,0))
abn[0,0]=10
pl.imshow(abn,aspect="auto")
pl.clim(-30,10)
pl.colorbar()
pl.subplot(4,1,3)
aba=np.transpose(ab[1:,:,1],(1,0))
aba[0,0]=10
pl.imshow(aba,aspect="auto")
pl.clim(-3.141592,3.141592)
pl.colorbar()
pl.subplot(4,1,4)
abn=np.transpose(abc[1:,:,1],(1,0))
pl.imshow(abn,aspect="auto")
pl.clim(-3.141592,3.141592)
pl.colorbar()
p=pyaudio.PyAudio()
ww = wave.open("B.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(bbb.tobytes())
ww.close()
pl.show()
