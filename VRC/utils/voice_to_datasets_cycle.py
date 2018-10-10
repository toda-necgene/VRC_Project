import pyaudio
import numpy as np
import wave
import time
import glob
import pyworld as pw

NFFT=1024
SHIFT=NFFT//2
dilations=0
term = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
WAVE_INPUT_FILENAME = "./datasets/source/01"
WAVE_INPUT_FILENAME2 = "./datasets/source/02"



files=glob.glob(WAVE_INPUT_FILENAME+"/*.wav")
name="/Source_data"
cnt=0
ff=list()
for file in files:
    print(" [*] パッチデータに変換を開始します。 :",file)
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
    data_realA=dmn=data_real.copy()
    timee=data_realA.shape[0]
    rate=16000
    b=np.zeros([1])

    times=data_realA.shape[0]//term+1
    if data_realA.shape[0]%term==0:
        times-=1
    ttm=time.time()
    resp=np.zeros([NFFT//2])
    for i in range(times):

        ind=term+SHIFT*dilations+SHIFT
        startpos=term*i+data_realA.shape[0]%term
        data_realAb = data_realA[max(startpos-ind,0):startpos].copy()/32767.
        r=ind-data_realAb.shape[0]
        if r>0:
            data_realAb=np.pad(data_realAb,(r,0),"constant")
        p = np.random.randint(30, 60) / 40
        _f0, t = pw.dio(data_realAb,16000)
        f0=pw.stonemask(data_realAb,_f0,t,16000)
        sp=pw.cheaptrick(data_realAb,f0,t,16000)
        if np.random.randint(1,5)>=3:
            p=1.0
        if p !=1.0:
            f0*=p
        a = sp
        f0=f0[f0>0.0]
        if len(f0)!=0:
            ff.extend(f0)
        np.save("./datasets/train/"+str(name)+"/"+str(cnt) +".npy", a)
        cnt+=1
print(" [*] ソースデータ変換完了")
print(cnt,np.mean(ff))

files=glob.glob(WAVE_INPUT_FILENAME2+"/*.wav")
name="/Answer_data"
cnt=0
ff=list()
for file in files:
    print(" [*] パッチデータに変換を開始します。 :",file)
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
    data_realA=dmn=data_real.copy()
    timee=data_realA.shape[0]
    rate=16000
    b=np.zeros([1])

    times=data_realA.shape[0]//term+1
    if data_realA.shape[0]%term==0:
        times-=1
    ttm=time.time()
    resp=np.zeros([NFFT//2])
    for i in range(times):
        ind=term+SHIFT*dilations+SHIFT
        startpos=term*i+data_realA.shape[0]%term
        data_realAb = data_realA[max(startpos-ind,0):startpos]
        r=ind-data_realAb.shape[0]
        if r>0:
            data_realAb=np.pad(data_realAb,(r,0),"constant")
        dmn=data_realAb
        _f0, t = pw.dio(data_realAb,16000)
        f0=pw.stonemask(data_realAb,_f0,t,16000)
        sp=pw.cheaptrick(data_realAb,f0,t,16000)
        a=sp
        f0=f0[f0>0.0]
        if len(f0)!=0:
            ff.extend(f0)
        np.save("./datasets/train/"+str(name)+"/"+str(cnt) +".npy", a)
        cnt+=1

print(" [*] アンサーデータ変換完了")
print(cnt,np.mean(ff))
print(" [*] プロセス完了!!　プログラムを終了します。")
