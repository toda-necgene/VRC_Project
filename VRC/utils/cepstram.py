import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
import time
import pyworld as pw
NFFT=1024
SHIFT=NFFT//2
fs = 16000
chs = 100




def encode(data):
    fs=16000
    _f0,t=pw.dio(data,fs)
    f0=pw.stonemask(data,_f0,t,fs)
    sp=pw.cheaptrick(data,f0,t,fs)
    ap=pw.d4c(data,f0,t,fs)
    sp=sp
    return f0,sp,ap
def decode(f0,sp,ap):
    return pw.synthesize(f0,sp,ap,16000)

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
ab=np.zeros([1,513])
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
    dmn=(data_realAb/32767.0).astype(np.double)
    f0,a,ap=encode(dmn)
    a=a.reshape(109,513)
    ab = np.append(ab, a, axis=0)
    s=decode(f0,a,ap)
    b=np.append(b,s)
r=b.shape[0]-data_realA.shape[0]
bbb=b
bbb=(bbb[1:]/2*32767).astype(np.int16)
aba=np.transpose(np.log(ab[1:,:513]),(1,0))
pl.imshow(aba,aspect="auto")
pl.colorbar()
p=pyaudio.PyAudio()
ww = wave.open("B.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(bbb.tobytes())
ww.close()
pl.show()
