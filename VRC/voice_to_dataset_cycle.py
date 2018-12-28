import numpy as np
import wave
import glob
import pyworld.pyworld as pw

term = 4096
CHANNELS = 1        #モノラル
RATE = 16000       #サンプルレート
CHUNK = 1024     #データ点数
WAVE_INPUT_FILENAME = "./dataset/source/A"
WAVE_INPUT_FILENAME2 = "./dataset/source/B"

def create_dataset():
    files=glob.glob(WAVE_INPUT_FILENAME+"/*.wav")
    ff=list()
    m=list()
    for file in files:
        print(" [*] パッチデータに変換を開始します。 :",file)
        dms=[]
        wf = wave.open(file, 'rb')
        dds = wf.readframes(CHUNK)
        while dds != b'':
            dms.append(dds)
            dds = wf.readframes(CHUNK)
        dms = b''.join(dms)
        data = np.frombuffer(dms, 'int16')
        data_real=(data/32767).reshape(-1).astype(np.float)
        data_realA=data_real.copy()
        times=data_realA.shape[0]//term+1
        if data_realA.shape[0]%term==0:
            times-=1
        for i in range(times):

            ind=term
            startpos=term*i+data_realA.shape[0]%term
            data_realAb = data_realA[max(startpos-ind,0):startpos].copy()
            r=ind-data_realAb.shape[0]
            if r>0:
                data_realAb=np.pad(data_realAb,(r,0),"constant")
            _f0, t = pw.dio(data_realAb,16000)
            f0=pw.stonemask(data_realAb,_f0,t,16000)
            sp=pw.cheaptrick(data_realAb,f0,t,16000)
            f0=f0[f0>0.0]
            if len(f0)!=0:
                ff.extend(f0)
            m.append(np.clip((np.log(sp)+15.0)/20,-1.0,1.0))
    m=np.asarray(m,dtype=np.float32)
    np.save("./dataset/train/A.npy", m)
    print(" [*] ソースデータ変換完了")
    pitch_mean_s=np.mean(ff)
    pitch_var_s=np.std(ff)
    files=glob.glob(WAVE_INPUT_FILENAME2+"/*.wav")
    ff=list()
    m=list()
    for file in files:
        print(" [*] パッチデータに変換を開始します。 :",file)
        dms=[]
        wf = wave.open(file, 'rb')
        dds = wf.readframes(CHUNK)
        while dds != b'':
            dms.append(dds)
            dds = wf.readframes(CHUNK)
        dms = b''.join(dms)
        data = np.frombuffer(dms, 'int16')
        data_real=(data/32767).reshape(-1).astype(np.float)
        data_realA=data_real.copy()
        times=data_realA.shape[0]//term+1
        if data_realA.shape[0]%term==0:
            times-=1
        for i in range(times):
            ind = term
            startpos=term*i+data_realA.shape[0]%term
            data_realAb = data_realA[max(startpos-ind,0):startpos]
            r=ind-data_realAb.shape[0]
            if r>0:
                data_realAb=np.pad(data_realAb,(r,0),"constant")
            _f0, t = pw.dio(data_realAb,16000)
            f0=pw.stonemask(data_realAb,_f0,t,16000)
            sp=pw.cheaptrick(data_realAb,f0,t,16000)
            a  = sp
            f0=f0[f0>0.0]
            if len(f0)!=0:
                ff.extend(f0)
            m.append(np.clip((np.log(a)+15.0)/20,-1.0,1.0))
    m=np.asarray(m,dtype=np.float32)
    np.save("./dataset/train/B.npy", m)
    print(" [*] アンサーデータ変換完了")
    pitch_mean_t=np.mean(ff)
    pitch_var_t=np.std(ff)
    plof=np.asarray([pitch_mean_s,pitch_mean_t,pitch_var_t/pitch_var_s])
    np.save("./voice_profile.npy",plof)
