import pyaudio
import numpy as np
import wave
import time


Add_Effect=True
NFFT=1024
SHIFT=NFFT//2
C1=32.703
rate=16000
Hz=C1*(2**0)
now=317.6
target=563.666
upidx=target/now
path="../setting.json"
def filter_clip(dd,f=1.5):
    dxf=np.maximum(dd,-f)+f+np.minimum(dd,f)-f
    return -dxf*0.5

def filter_mean(dd):
    dxx1=np.roll(dd,1)
    dxx1[:1]=dd[:1]
    dxx2=np.roll(dd,2)
    dxx2[:2] = dd[:2]
    dxx3= np.roll(dd, 3)
    dxx3[:3] = dd[:3]
    dxx4 = np.roll(dd, 4)
    dxx4[:4] = dd[:4]
    return (dd+dxx1+dxx2+dxx3+dxx4)/5.0
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
    # fft_data/=window
    v = fft_data[:,:NFFT//2]
    res = fft_data[-1, NFFT//2 :].copy()
    lats = np.roll(fft_data[:,NFFT//2:],1,axis=0)
    spec=np.reshape(v+lats,(-1))
    return spec,res
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_OUTPUT_FILENAME = "./B.wav"
WAVE_OUTPUT_FILENAME2 = "./B2.wav"
file="../train/Model/datasets/test/label.wav"
file_l="../train/Model/datasets/test/test.wav"

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
wf = wave.open(file_l, 'rb')
dds = wf.readframes(CHUNK)
while dds != b'':
    dms.append(dds)
    dds = wf.readframes(CHUNK)
dms = b''.join(dms)
data = np.frombuffer(dms, 'int16')
data_realB=data.reshape(-1)

timee=80000
times=data_realB.shape[0]//timee

rate=16000

ab=np.zeros([1,1024,2])
abc=np.zeros([1,1024,2])
b=np.array([],dtype=np.int16)
b2=np.array([],dtype=np.int16)
term=8192
times=data_realA.shape[0]//term+1
if data_realA.shape[0]%term==0:
    times-=1
ttm=time.time()
resp=np.zeros([NFFT//2])
resp2=np.zeros([NFFT//2])
print("convert_start")
for i in range(times):
    ind=term+SHIFT
    startpos=term*i+data_realB.shape[0]%term
    data_realAb= data_realA[max(startpos - ind, 0):startpos]
    data_realBb = data_realB[max(startpos - ind, 0):startpos]
    r=ind-data_realBb.shape[0]
    if r>0:
        data_realBb=np.pad(data_realBb,(r,0),"constant")
    r = ind - data_realAb.shape[0]
    if r > 0:
        data_realAb = np.pad(data_realAb, (r, 0), "constant")
    dms = data_realAb / 32767.0
    dms=shift(dms,upidx)
    ddms=data_realBb/32767.0
    bss=fft(ddms)
    res=fft(dms)
    res=complex_to_pp(res)
    bss=complex_to_pp(bss)
    a=bss.copy()
    a[:,:,1]=res[:,:,1]
    x = res.copy()
    x[:, :, 1] = bss[:, :, 1]

    a=pp_to_complex(a)
    x = pp_to_complex(x)

    ss,resp=ifft(a,resp)
    ss=(np.clip(ss,-1.0,1.0)*32767).astype(np.int16)
    b = np.append(b, ss)
    ss2, resp2 = ifft(x, resp2)
    ss2 =( np.clip(ss2, -1.0, 1.0) * 32767).astype(np.int16)
    b2 = np.append(b2, ss2)
# print(a)
b=b[-data_realB.shape[0]:]
FORMAT=pyaudio.paInt16
p=pyaudio.PyAudio()
ww = wave.open("AB.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(b.tobytes())
ww.close()


b2=b2[-data_realB.shape[0]:]
FORMAT=pyaudio.paInt16
p=pyaudio.PyAudio()
ww = wave.open("BA.wav", 'wb')
ww.setnchannels(1)
ww.setsampwidth(p.get_sample_size(FORMAT))
ww.setframerate(RATE)
ww.writeframes(b2.tobytes())
ww.close()

print(" [*] Finished!!")
