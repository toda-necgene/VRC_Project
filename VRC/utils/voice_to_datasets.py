import pyaudio
import numpy as np
import wave
import time
import glob
NFFT=1024
SHIFT=NFFT//2
rate=16000
dilations=15
term = 4096
cut=200
target=150
save_num=100
FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
RECORD_SECONDS = 5 #録音する時間の長さ
WAVE_INPUT_FILENAME = "../train/Model/datasets/source/02"
files=glob.glob(WAVE_INPUT_FILENAME+"/*.wav")
name="v4/Source_data"
sample_name="v4/Samples"

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



cnt=0
cnt_ns=0
freq=np.fft.fftfreq(NFFT,d=1.0/16000)
file_freq_list=np.zeros(len(files))
scales_list=list()
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
    file_freqs=list()
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
        m=np.max(c,axis=1)>3.0
        me = np.mean(c, axis=1)>-5.5
        cs=list()
        for sd in range(c.shape[0]):
            if m[sd] and me[sd]:
                cs.append(c[sd])
        if len(cs)!=0:
            for v in cs:
                ala = np.fft.ifft(v).real
                ala[cut:-cut] = 0
                ala = np.fft.fft(ala).real[:SHIFT]
                f = 12
                ed = 450
                n = 6
                alan=ala[f:ed]
                alas=list()
                alaa=alan.copy()
                alab=np.roll(alan.copy(),1)
                alad=alaa-alab
                for k in range(alad.shape[0]-1):
                    if alad[k]>=0 and alad[k+1]<=0:
                        alas.append(v[k+f])
                    else:
                        alas.append(-np.inf)
                alas=np.asarray(alas)
                maximam_form=np.max(alas)
                ana = np.argsort(alas)[-n:]
                ana = np.sort(ana)
                mini=np.nan
                formants=list()
                for k in ana:
                    if k!=0 and not np.isnan(alas[k]) and alas[k]>maximam_form-3.5:
                        formants.append(k+f)
                for k in range(len(formants)-1):
                    file_freqs.append(freq[formants[k+1]-formants[k]])

            cnt+=1
    file_freqs=np.asarray(file_freqs)
    scales=np.round(np.log2(file_freqs/27.5)*12).astype(np.int16)
    counts=np.bincount(scales)
    mode=np.argmax(counts)
    file_freq=np.power(2,mode/12)*27.5
    print(" [i] 基本周波数を算出しました。:",file_freq,scales.shape)
    file_freq_list[cnt_ns]=file_freq
    scales_list.append(scales[:100])
    cnt_ns+=1
scales_list=np.asarray(scales_list)

with open("hist.csv","w") as f:
    for hsm in file_freq_list:
        f.write(str(hsm)+",")
    f.writelines("\n")
    for hsm in scales_list:
        for hmm in hsm:
            f.write(str(hmm)+",")
        f.writelines("\n")

print(" [*] アナライズ完了")

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
    data_real=data.reshape(-1)/32767.0
    data_realA=dmn=shift(data_real.copy(),delta)
    timee=data_realA.shape[0]
    rate=16000
    b=np.zeros([1])

    times=data_realA.shape[0]//term+1
    if data_realA.shape[0]%term==0:
        times-=1
    ttm=time.time()
    resp=np.zeros([NFFT//2])
    for i in range(times):
        if i < save_num:
            ind=term+SHIFT*dilations+SHIFT
            startpos=term*i+data_realA.shape[0]%term
            data_realAb = data_realA[max(startpos-ind,0):startpos]
            r=ind-data_realAb.shape[0]
            if r>0:
                data_realAb=np.pad(data_realAb,(r,0),"constant")
            dmn=data_realAb
            a=fft(dmn)
            a=complex_to_pp(a[:,:SHIFT])
            a=np.append(a,cnt_ns)
            np.save("../train/Model/datasets/train/"+str(name)+"/"+str(cnt) +".data", a)
            cnt+=1
    p=pyaudio.PyAudio()
    data_realA=(data_realA*32767).astype(np.int16)
    ww = wave.open("../train/Model/datasets/train/"+str(sample_name)+"/"+str(cnt_ns) +".wav", 'wb')
    ww.setnchannels(1)
    ww.setsampwidth(p.get_sample_size(FORMAT))
    ww.setframerate(RATE)
    ww.writeframes(data_realA.tobytes())
    ww.close()
    cnt_ns+=1
print(" [*] プロセス完了!!　プログラムを終了します。")
