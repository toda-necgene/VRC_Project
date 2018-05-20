import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as pl
NFFT=128
SHIFT=NFFT//2
C1=32.703
rate=16000
Hz=C1*(2**0)
now=317.6
target=563.666
upidx=target/now
print(upidx)
def fft(data):
    time_ruler=data.shape[0]//SHIFT
    if data.shape[0]%SHIFT==0:
        time_ruler-=1
    window=np.hamming(NFFT)
    spec=np.zeros([time_ruler,NFFT,2])
    pos=0
    for fft_index in range(time_ruler):
        frame=data[pos:pos+NFFT]
        wined=frame*window
        fft_r=np.fft.fft(wined)
        re=fft_r.real.reshape(-1)
        im=fft_r.imag.reshape(-1)
        c = np.log(np.power(re, 2) + np.power(im, 2) + 1e-24).reshape(-1,1)
        d = np.arctan2(im, re).reshape(-1,1)
        fft_data=np.concatenate((c,d),1)
        for ik in range(len(spec[fft_index])):
            spec[fft_index][ik]=fft_data[ik]
        pos+=NFFT//2
    return spec
def shift(data_inps,pitch):
    data_inp=data_inps.reshape(-1)
    return scale(time_strech(data_inp,1/pitch),data_inp.shape[0])
    # return time_strech(data_inp,1/pitch)
    # return scale(data_inp,data_inp.shape[0]/2)

def scale(inputs,len_wave):
    x=np.linspace(0.0,inputs.shape[0]-1,len_wave)
    ref_x_n=(x+0.5).astype(int)
    spec=inputs[ref_x_n[...]]
    return spec.reshape(-1)
def time_strech(datanum,speed):
    term_s = int(rate * 0.05)
    pulus=int(term_s*speed)
    data_s=datanum.reshape(-1)
    spec=np.zeros(1)
    ifs=np.zeros(pulus//2)
    for i_s in np.arange(0.0,data_s.shape[0],pulus):
        dd=data_s[int(i_s):int(i_s+term_s)]
        fade = min(int(pulus/2),dd.shape[0])
        ds_in= np.linspace(0,1,fade)
        ds_out =  np.linspace(1,0,fade)
        stock=dd[:fade]
        dd[:fade] = dd[:fade] * ds_in
        if  i_s!=0:
            dd[:fade]+=ifs[:fade]
        else :
            dd[:fade] += stock * np.linspace(1, 0, fade)
        ifs=dd[-fade:]*ds_out
        if i_s+pulus>=data_s.shape[0]:
            spec = np.append(spec, dd)
        else:
            spec=np.append(spec,dd[:-fade])
    return spec[1:]

def ifft(data):
    time_ruler=data.shape[0]
    window=np.hamming(NFFT)
    spec=np.zeros([])
    pos=0
    lats = np.zeros([NFFT//2])
    for _ in range(time_ruler):
        frame=data[pos]
        power = np.sqrt(np.exp(frame[:,0]))
        re = power * (np.cos(frame[:,1]))
        im = power * (np.sin(frame[:,1]))
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
file=(WAVE_INPUT_FILENAME+"/test.wav")
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

b=np.zeros([1])
ab=np.zeros([1,128,2])
abc=np.zeros([1,128,2])

term=8192
times=data_realA.shape[0]//term+1
if data_realA.shape[0]%term==0:
    times-=1
for i in range(times):
    ind=SHIFT+term
    startpos=term*(i+1)
    data_realAb = data_realA[max(startpos-ind,0):startpos]
    r=ind-data_realAb.shape[0]
    if r>0:
        data_realAb=np.pad(data_realAb,(0,r),"reflect")
    dmn=data_realAb/32767.0
    dmn=shift(dmn,upidx)
    r=SHIFT-dmn.shape[0]%SHIFT
    if r!=SHIFT:
        dmn=np.pad(dmn,(0,r),"reflect")
    a=fft(dmn)
    ab = np.append(ab, a, axis=0)
    s=ifft(a)
    ac=fft(s/2)
    abc=np.append(abc,ac,axis=0)
    s=ifft(ac)
    b=np.append(b,s)
# print(a)

r=b.shape[0]-data_realA.shape[0]
b=(b[1:]/2*32767).astype(np.int16)
# c=np.log(np.power(ab[:,:,0],2)+np.power(ab[:,:,1],2)+1e-8)
# d=np.arctan2(ab[:,:,1],ab[:,:,0])
# e=np.sqrt(np.exp(c))*np.exp(d)
# sa=np.transpose(ab[:,:,0],(1,0))

pl.subplot(2,1,1)
aba=np.transpose(ab[1:,:,0],(1,0))
pl.imshow(aba,aspect="auto")
pl.colorbar()
pl.subplot(2,1,2)
abn=np.transpose(ab[1:,:,0],(1,0))
pl.imshow(abn,aspect="auto")
pl.colorbar()

p=pyaudio.PyAudio()
ww = wave.open("B.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(b.tobytes())
ww.close()
pl.show()
