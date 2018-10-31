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

FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "./"
WAVE_OUTPUT_FILENAME = "./test.wav"
file="../datasets/test/label.wav"
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
data_realB=data.reshape(-1)
rate=16000
b=np.zeros([1])
ab=np.zeros([1,513])
abc=np.zeros([1])

term=4096
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
    dmn=data_realAb.astype(np.double)
    f0,a,ap=encode(dmn)
    ap = ap[::4]
    ap = np.tile(ap.reshape(-1, 1, 513), (1, 4, 1))
    ap = ap.reshape(-1, 513)

    f0=f0[::4]
    f0 = np.tile(f0.reshape(-1, 1), (1, 4))
    f0 = f0.reshape(-1)

    a=a[::4,:]
    a=np.tile(a.reshape(-1,1,513),(1,4,1))
    a=a.reshape(-1,513)
    ab = np.append(ab, a, axis=0)
    s=decode(f0,a,ap)
    b=np.append(b,s)
r=b.shape[0]-data_realA.shape[0]
bbb=b
bbb=bbb[1:].astype(np.int16)
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
