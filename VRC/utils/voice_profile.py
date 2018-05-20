import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
NFFT=8192
SHIFT=NFFT//2
C1=32.703
rate=16000
Hz=C1*(2**0)
cut=-1.0
target=int(1/Hz*rate)
print(target)
if target>NFFT:
    print("Cannot to convert")
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
            c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(-1,1)
            d = np.arctan2(im, re).reshape(-1,1)
            fft_data=np.concatenate((c,d),1)
            for i in range(len(spec[fft_index])):
                spec[fft_index][i]=fft_data[i]
            pos+=NFFT//2
    return spec
def shift(data):
    ist=np.empty(0)
    spd=np.empty(0)
    terms=data.shape[1]
    for s in range(data.shape[0]):
        ds=data[s,:,:]
        # noise_cutting
        spec=np.zeros(ds.shape)-100
        dds_s = np.clip(ds.reshape(-1,2),-100.0,5)
        dds_s_a = dds_s[:SHIFT, 0][::-1]
        dds_s_b = dds_s[SHIFT:, 0]
        obs=np.min([np.abs(np.argmax(dds_s_a)-SHIFT),np.abs(SHIFT-np.argmax(dds_s_b))])
        iss=ds[obs, 0]
        obtainHz=obs
        if iss>cut:
            ist = np.append(ist, obtainHz)
        else:
            spec[:, 0]=-100
        spd=np.append(spd,spec)
    spd=spd.reshape(data.shape)
    return spd,ist
def ifft(data):
    time_ruler=data.shape[0]
    window=np.hamming(NFFT)
    spec=np.zeros([])
    pos=0
    lats = np.zeros([NFFT//2])
    for _ in range(time_ruler):
        frame=data[pos]
        p = np.sqrt(np.exp(frame[:,0]))
        re = p * (np.cos(frame[:,1]))
        im = p * (np.sin(frame[:,1]))
        ep = re+1j*im
        fft_s=np.fft.ifft(ep)
        fft_data=fft_s.real
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
file=(WAVE_INPUT_FILENAME+"/label.wav")
index=0
dms=[]
wf = wave.open(file, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')[0:160000].reshape(2,-1)
data_realA=data.reshape(-1)
time=80000
times=data_realA.shape[0]//time

rate=16000

b=np.empty([])
bb=np.empty([])
ab=np.empty([0,NFFT,2])
term=16384
times=data_realA.shape[0]//term+1
if data_realA.shape[0]%term==0:
    times-=1
for i in range(times):
    ind=SHIFT+term
    startpos=term*(i+1)
    data_realAb = data_realA[max(startpos-ind,0):startpos]
    r=ind-data_realAb.shape[0]
    if r>0:
        data_realAb=np.pad(data_realAb,(0,r),"constant")
    a=fft(data_realAb/32767.0)
    ss,iid=shift(a)
    ab = np.append(ab, a, axis=0)
    bb=np.append(bb,iid)
    s=ifft(ss)
    b=np.append(b,s)
# print(a)
b=(b[:data_realA.shape[0]]/1.5*32767).astype(np.int16)
# c=np.log(np.power(ab[:,:,0],2)+np.power(ab[:,:,1],2)+1e-8)
# d=np.arctan2(ab[:,:,1],ab[:,:,0])
# e=np.sqrt(np.exp(c))*np.exp(d)
# sa=np.transpose(ab[:,:,0],(1,0))

pl.plot(data_realA)
print(np.mean(bb))
# asss=np.transpose(ab[:,:,0],[1,0])
# pl.imshow(asss,aspect="auto")
# pl.colorbar()
p=pyaudio.PyAudio()
ww = wave.open("B.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(b.tobytes())
ww.close()
pl.show()
