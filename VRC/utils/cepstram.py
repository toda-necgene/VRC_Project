import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
import time
NFFT=1024
SHIFT=NFFT//2
fs = 16000
chs = 100


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
def hz_to_mel(fss):
    return 1127.01048*np.log(fss/700+1.0)
def mel_to_hz(fss):
    return (np.exp(fss/1127.01048)-1.0)*700.0
def mel_freq_encode(datanum):
    fmax=fs/2
    mel_f=np.zeros([datanum.shape[0],chs])
    melmax=hz_to_mel(fmax)
    nmax=NFFT/2
    df=fs/NFFT
    dmel=melmax/(chs+1)
    melcenters=np.arange(1,chs+1)*dmel
    fcenters=mel_to_hz(melcenters)
    indexc=np.round(fcenters/df)
    indexl=np.hstack(([0],indexc[0:chs-1]))
    indexr=np.hstack((indexc[1:chs],[nmax]))
    indexrange=indexr-indexl
    for g in range(chs):
        b=np.bartlett(indexrange[g])
        b=np.pad(b,(int(indexl[g]),int(nmax-indexr[g])),"constant")
        mel_f[:,g]=np.sum(datanum*b,axis=-1)
    return mel_f
def mel_freq_decode(datanum):
    fmax=fs/2
    mel_f=np.zeros([datanum.shape[0],SHIFT])
    melmax=hz_to_mel(fmax)
    nmax=NFFT/2
    df=fs/NFFT
    dmel=melmax/(chs+1)
    melcenters=np.arange(1,chs+1)*dmel
    fcenters=mel_to_hz(melcenters)
    indexc=np.round(fcenters/df)
    indexl=np.hstack(([0],indexc[0:chs-1]))
    indexr=np.hstack((indexc[1:chs],[nmax]))
    indexrange=indexr-indexl
    for g in range(chs):
        p = datanum[:, g]
        b=np.tile(np.bartlett(indexrange[g]),[p.shape[0],1])
        for h in range(b.shape[0]):
            b[h]*=p[h]
        b=np.pad(b,((0,0),(int(indexl[g]),int(nmax-indexr[g]))),"constant")
        mel_f[:]+=b
    return mel_f

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
    m=a[:,:SHIFT,0]
    m=mel_freq_encode(m)
    m = np.concatenate([m, m[:, ::-1]], axis=-1)

    m=np.fft.fft(m)
    f=50
    m[:,f:-f]=0

    m=np.fft.ifft(m).real
    m = m[:, :SHIFT]
    m = mel_freq_decode(m)
    a[:,:,0]=np.concatenate([m,m[:,::-1]],axis=-1)

    ab = np.append(ab, a, axis=0)
    a=pp_to_complex(a)
    s,resp=ifft(a,resp)
    b=np.append(b,s)
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
