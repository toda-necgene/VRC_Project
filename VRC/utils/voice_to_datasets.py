import pyaudio
import numpy as np
import wave
import time
import glob
NFFT=1024
SHIFT=NFFT//2
rate=16000
term = 4096
cut=60
target=750

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

FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "../train/Model/datasets/source/02"
files=glob.glob(WAVE_INPUT_FILENAME+"/*.wav")
name="4096-128-2/Answer_data"
cnt=0
cnt_ns=0
freq=np.fft.fftfreq(NFFT,d=1.0/16000)
file_freq_list=np.zeros(len(files))
print(" [*] 変換プロセスを実行します。")
for file in files:
    print(" [*] アナライズを開始します :", file)
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
    ab=np.zeros([1,NFFT,2])
    abc=np.zeros([1,NFFT,2])

    times=data_realA.shape[0]//term+1
    if data_realA.shape[0]%term==0:
        times-=1
    ttm=time.time()
    resp=np.zeros([NFFT//2])
    ala=np.zeros(NFFT)
    ca=0
    file_freq=0.0
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
            dmn=np.pad(dmn,(0,r),"constant")
        a=fft(dmn)
        a=complex_to_pp(a)
        c=a[:,:,0]
        m=np.max(c,axis=1)>3.5
        me = np.mean(c, axis=1)>-6.5
        cs=list()
        for sd in range(c.shape[0]):
            if m[sd] and me[sd]:
                cs.append(c[sd])
        if len(cs)!=0:
            for v in cs:
                ala = np.fft.ifft(v).real
                ala[cut:-cut] = 0
                ala = np.fft.fft(ala).real[:SHIFT]
                f = 0
                ed = 450
                n = 10

                alan=ala[f:ed]
                # bla = np.roll(alan.copy(), -1, axis=0)
                # ala = alan-bla
                # alas=[]
                # ank=0
                # ask=0
                # alas.append(-np.inf)
                # for l in range(ala.shape[0]-1):
                #     if ala[l]>= 0 and ala[l+1]<0 and l!=0:
                #         alas.append(alan[l]-ank)
                #         ask=l
                #     elif ala[l]<= 0 and ala[l+1]>0 or l==0:
                #         ank=alan[l]
                #         if ask!=0 and alas[ask]<alan[ask]-ank:
                #             alas[ask] = alan[ask] - ank
                #         alas.append(-np.inf)
                #     else:
                #         alas.append(-np.inf)
                # alas=np.asarray(alan[:-1])
                ana = np.argsort(alan)[-n:]
                aws=np.exp(alan[ana])
                ana_weight = aws+np.exp((np.min(alan)))
                ang=0
                ttm=0
                for l in range(ana.shape[0]):
                    ang += (ana[l])*ana_weight[l]
                    ttm += ana_weight[l]
                if ttm!=0:
                    file_freq += freq[int(ang/ttm) + f]
                    ca+=1
            cnt+=1
    if ca==0:
        print("Error!!")
    file_freq/=ca
    print(" [i] 基本周波数を算出しました。:",file_freq)
    file_freq_list[cnt_ns]=file_freq
    cnt_ns+=1
print(" [*] アナライズ完了")
print(file_freq_list)
"""
cnt=0
cnt_ns=0
for file in files:
    print(" [*] パッチデータに変換を開始します。 :",file)
    delta=target/file_freq_list[cnt_ns]
    print(" [i] ピッチの倍率　:", delta)
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
        dmn=shift(dmn,delta)
        a=fft(dmn)
        a=complex_to_pp(a[:,:SHIFT])
        np.save("../train/Model/datasets/train/"+str(name)+"/"+str(cnt) +".data", a)
        cnt+=1
    cnt_ns+=1
print(" [*] プロセス完了!!　プログラムを終了します。")
"""