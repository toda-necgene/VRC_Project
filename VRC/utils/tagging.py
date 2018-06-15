import pyaudio
import numpy as np
import wave
import time
import glob
import os
import matplotlib.pyplot as plt
NFFT=128
SHIFT=NFFT//2
C1=32.703
rate=16000
Hz=C1*(2**0)
now=317.6
target=563.666
term = 4096
length=term//SHIFT+1
cutoff=5
effect_ranges=1024
align=True
eef=1200
upidx=target/now
p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,channels=1,rate=16000,output=True)
print(stream.is_active())
lss={"a":0,"i":1,"u":2,"e":3,"o":4,"k":5,"s":6,"t":7,"n":8,"h":9,"m":10,"y":11,"r":12,"w":13,"g":14,"z":15,"d":16,"b":17,"p":18,"H":19,"Q":20,"N":21,"f":22,"R":23,"?":24}
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
def play(d):
    stream.write(d.tobytes())
FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "../train/Model/datasets/source/02/"
files=glob.glob(WAVE_INPUT_FILENAME+"*.wav")
filestri=glob.glob(WAVE_INPUT_FILENAME+"*.str")
stri=""
name="Answer_data"
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
    data = np.frombuffer(dms, np.int16)
    data_real=data.reshape(-1)
    data_realA=data_real
    print(np.max(data_realA[0:8192]))
    timee=data_realA.shape[0]
    ppt=file+".txt"
    seg = 0

    with open(ppt,"rb") as fos:
        b=fos.readline()
        while b is not b'':
            stri+=b.decode('utf-8')
            b=fos.readline()
    rate=16000
    b=np.zeros([1])
    ab=np.zeros([1,128,2])
    abc=np.zeros([1,128,2])
    times=data_realA.shape[0]//term+1
    if data_realA.shape[0]%term==0:
        times-=1
    ttm=time.time()
    mod=data_realA.shape[0]%term
    resp=np.zeros([NFFT//2])
    for i in range(times):
        ind=term+SHIFT
        startpos=term*i+mod
        data_realAb = data_realA[max(startpos-ind,0):startpos]
        r=ind-data_realAb.shape[0]
        if r>0:
            data_realAb=np.pad(data_realAb,(r,0),"constant")
        dddd = data_realAb.copy()
        dmn=data_realAb.copy()/32767.0
        r=SHIFT-dmn.shape[0]%SHIFT
        if r!=SHIFT:
            dmn=np.pad(dmn,(0,r),"reflect")
        a=fft(dmn)
        a=complex_to_pp(a[:,:SHIFT])
        c=a[:,:,0]
        a[:, :, 0] -= np.tile(np.mean(c, axis=1).reshape(-1, 1), (1, SHIFT))
        v = 1 / np.sqrt(np.var(c, axis=1) + 1e-36)
        a[:, :, 0]= np.einsum("ij,i->ij",a[:, :, 0],v)
        bb=np.isnan(np.mean(a))
        #音素アラインメントの実行
        ttms=600//SHIFT
        ttm_pp = int(ttms * 0.2)
        ttm_main=ttms*2-ttm_pp*2
        if align and not os.path.exists("../train/Model/datasets/train/" + str(name) + "/" + str(cnt) + "-stri.npy"):
            stridatra=np.zeros([length-cutoff,24],dtype=np.float32)
            flag=True
            tts=0
            segs=seg-1
            while True:
                print("play"+str(startpos))
                play(dddd)
                print("きこえた言葉をアルファベットで入力 %d/%d"%(i+1,times))
                print(stri[max(seg-1,0):min(seg+5,len(stri))])
                ins=input().encode("utf-8").decode("utf-8")
                deins=""
                if ins=="":
                    flag=False
                for n in ins:
                    if n=="?":
                        deins += "?"
                    elif n in lss:
                        deins += n
                        segs+=1
                    else:
                        flag=False
                if flag:
                    pss=(length-cutoff)//(len(deins)+1)
                    r=1
                    for n in deins:
                        ff=lss[n]
                        mainpos=pss*r
                        ts=np.linspace(0.1,1.0,ttm_pp)
                        tm=np.ones([ttm_main])
                        te=np.linspace(1.0,0.1,ttm_pp)
                        filt=np.append(ts,tm).reshape(-1)
                        filt=np.append(filt,te).reshape(-1)
                        s=mainpos-ttms
                        e=mainpos+ttms
                        sk=0
                        se=ttms*2
                        if s<0:
                            s=0
                            sk=-s-1
                        if e>=stridatra.shape[0]:
                            e=stridatra.shape[0]
                            se=stridatra.shape[0]-e
                        if ff!=24:
                            stridatra[s:e,ff]+=filt[sk:se]
                        else:
                            stridatra[s:e, :]=0.0
                        r+=1
                    seg+=len(ins)
                    break
                print("しっぱい")
                flag = True
                seg = segs
            np.save("../train/Model/datasets/train/" + str(name) + "/" + str(cnt) + "-stri", stridatra)
        if bb:
            print("NAN!!")
        np.save("../train/Model/datasets/train/"+str(name)+"/"+str(cnt) +"-wave", a)
        cnt+=1
plt.subplot(211)
plt.imshow(a[:,:,0],aspect="auto")
plt.colorbar()
plt.subplot(212)
plt.imshow(a[:,:,1],aspect="auto")
plt.colorbar()
plt.show()